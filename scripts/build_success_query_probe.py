#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_verl_shoppingbench_data import (  # noqa: E402
    build_product_cache,
    build_system_prompt,
    convert_query_row,
    product_ids_from_query_rows,
    read_jsonl,
    token_len,
)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def slack_bucket(slack: float, budget: float) -> str:
    if budget <= 0 or math.isnan(slack):
        return "unknown"
    ratio = slack / budget
    if slack <= 20 or ratio <= 0.05:
        return "tight"
    if slack <= 80 or ratio <= 0.12:
        return "medium"
    return "loose"


def threshold_bucket(price_after_voucher: float, threshold: float) -> str:
    if threshold <= 0 or math.isnan(price_after_voucher):
        return "unknown"
    ratio = price_after_voucher / threshold
    if ratio < 1.25:
        return "near_threshold"
    if ratio < 2.0:
        return "mid_threshold"
    return "far_threshold"


def row_features(index: int, row: dict[str, Any]) -> dict[str, Any]:
    voucher = row.get("voucher") or {}
    budget = as_float(voucher.get("budget"))
    price_after = as_float(voucher.get("price_after_voucher"))
    threshold = as_float(voucher.get("threshold"))
    slack = budget - price_after
    return {
        "row_index": index,
        "query": row.get("query"),
        "voucher_type": voucher.get("voucher_type"),
        "discount_type": voucher.get("discount_type"),
        "product_count": len(row.get("reward") or []),
        "budget": budget,
        "threshold": threshold,
        "price_after_voucher": price_after,
        "budget_slack": slack,
        "budget_slack_ratio": slack / budget if budget else None,
        "slack_bucket": slack_bucket(slack, budget),
        "threshold_bucket": threshold_bucket(price_after, threshold),
        "reward_product_ids": [
            str(item["product_id"]) for item in row.get("reward", []) if item.get("product_id") is not None
        ],
    }


def feature_keys(features: dict[str, Any]) -> set[str]:
    return {
        f"voucher:{features['voucher_type']}",
        f"discount:{features['discount_type']}",
        f"count:{features['product_count']}",
        f"slack:{features['slack_bucket']}",
        f"threshold:{features['threshold_bucket']}",
        f"combo:{features['voucher_type']}:{features['discount_type']}",
    }


def select_rows(features: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: Counter[str] = Counter()
    remaining = sorted(
        features,
        key=lambda item: (
            item["slack_bucket"] != "tight",
            -int(item["product_count"]),
            item["budget_slack"],
            item["row_index"],
        ),
    )
    while remaining and len(selected) < sample_size:
        best = None
        best_key = None
        for item in remaining:
            novelty = sum(1.0 / (1.0 + covered[key]) for key in feature_keys(item))
            hard_bonus = 0.5 if item["slack_bucket"] == "tight" else 0.2 if item["slack_bucket"] == "medium" else 0.0
            count_bonus = 0.1 * int(item["product_count"])
            score = novelty + hard_bonus + count_bonus
            sort_key = (-score, item["budget_slack"], item["row_index"])
            if best_key is None or sort_key < best_key:
                best = item
                best_key = sort_key
        assert best is not None
        selected.append(best)
        for key in feature_keys(best):
            covered[key] += 1
        remaining = [item for item in remaining if item["row_index"] != best["row_index"]]
    return sorted(selected, key=lambda item: item["row_index"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RL rollout probe set from successful teacher queries.")
    parser.add_argument("--input", default="data/tmp/teacher_gpt55medium_success_merged_20260709/clean_success_queries.jsonl")
    parser.add_argument("--manifest", default="data/tmp/teacher_gpt55medium_success_merged_20260709/clean_success_manifest.jsonl")
    parser.add_argument("--output-dir", default="dataset/probe/sft_clean924_success8_state_local")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--model-name", default="model/Qwen3-4B")
    parser.add_argument("--prompt-file", default="src/agent/prompt/rollout.state_local.md")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--documents-file", default="resources/documents.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(ROOT / args.input)
    manifests = read_jsonl(ROOT / args.manifest) if args.manifest else [None] * len(rows)
    if len(rows) != len(manifests):
        raise ValueError(f"query/manifest count mismatch: {len(rows)} vs {len(manifests)}")

    features = [row_features(index, row) for index, row in enumerate(rows)]
    selected_features = select_rows(features, args.sample_size)
    selected_indices = [item["row_index"] for item in selected_features]
    selected_rows = [rows[index] for index in selected_indices]
    selected_manifests = [manifests[index] for index in selected_indices]

    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_name, trust_remote_code=True)
    system_prompt = build_system_prompt(ROOT / args.prompt_file)
    converted = []
    dropped = []
    for out_idx, (row, manifest, feature) in enumerate(zip(selected_rows, selected_manifests, selected_features)):
        item = convert_query_row(row, out_idx, "probe", system_prompt, manifest)
        item["extra_info"].update(
            {
                "clean_success_index": feature["row_index"],
                "probe_selection": {k: v for k, v in feature.items() if k != "query"},
            }
        )
        length = token_len(tokenizer, item["prompt"])
        if length <= args.max_prompt_length:
            item["extra_info"]["prompt_tokens"] = length
            converted.append(item)
        else:
            dropped.append({"clean_success_index": feature["row_index"], "prompt_tokens": length})

    if dropped:
        raise RuntimeError(f"selected rows exceeded max prompt length: {dropped}")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "probe.parquet"
    jsonl_path = output_dir / "probe_queries.jsonl"
    cache_path = output_dir / "product_cache.json"
    report_path = output_dir / "report.json"

    pd.DataFrame(converted).to_parquet(parquet_path, index=False)
    write_jsonl(jsonl_path, selected_rows)
    build_product_cache(ROOT / args.documents_file, product_ids_from_query_rows(selected_rows), cache_path)

    report = {
        "input": args.input,
        "manifest": args.manifest,
        "output_dir": args.output_dir,
        "prompt_file": args.prompt_file,
        "sample_size": len(converted),
        "max_prompt_length": args.max_prompt_length,
        "distribution": {
            "voucher_type": dict(Counter(item["voucher_type"] for item in selected_features)),
            "discount_type": dict(Counter(item["discount_type"] for item in selected_features)),
            "product_count": dict(Counter(str(item["product_count"]) for item in selected_features)),
            "slack_bucket": dict(Counter(item["slack_bucket"] for item in selected_features)),
            "threshold_bucket": dict(Counter(item["threshold_bucket"] for item in selected_features)),
        },
        "selected": selected_features,
        "files": {
            "parquet": str(parquet_path),
            "queries": str(jsonl_path),
            "product_cache": str(cache_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
