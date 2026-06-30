#!/usr/bin/env python3
"""Build a small, stratified Coupon/Budget probe set from RL train queries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


CORE_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info", "agent_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/shoppingbench_query/train.parquet")
    parser.add_argument("--output-parquet", default="dataset/probe/qwen3_4b_voucher_probe_20260629.parquet")
    parser.add_argument("--output-jsonl", default="dataset/probe/qwen3_4b_voucher_probe_20260629.jsonl")
    parser.add_argument("--report", default="reports/voucher_probe_set_20260629.json")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, tuple) else [value]


def load_ground_truth(row: pd.Series) -> dict:
    ground_truth = row["reward_model"]["ground_truth"]
    return json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth


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


def row_features(index: int, row: pd.Series) -> dict:
    extra = row["extra_info"]
    gt = load_ground_truth(row)
    voucher = gt.get("voucher") or {}
    budget = float(voucher.get("budget") or extra.get("budget") or 0)
    price_after = float(voucher.get("price_after_voucher") or 0)
    threshold = float(voucher.get("threshold") or 0)
    slack = budget - price_after
    return {
        "row_index": int(index),
        "query_index": int(extra.get("index", index)),
        "query": extra.get("query"),
        "voucher_type": voucher.get("voucher_type") or extra.get("voucher_type"),
        "discount_type": voucher.get("discount_type"),
        "product_count": int(extra.get("product_count") or len(as_list(extra.get("reward_product_ids")))),
        "budget": budget,
        "threshold": threshold,
        "price_after_voucher": price_after,
        "budget_slack": slack,
        "budget_slack_ratio": slack / budget if budget else None,
        "slack_bucket": slack_bucket(slack, budget),
        "threshold_bucket": threshold_bucket(price_after, threshold),
        "prompt_tokens": int(extra.get("prompt_tokens") or 0),
        "reward_product_ids": [str(item) for item in as_list(extra.get("reward_product_ids"))],
        "voucher": voucher,
    }


def feature_keys(features: dict) -> set[str]:
    return {
        f"voucher:{features['voucher_type']}",
        f"discount:{features['discount_type']}",
        f"count:{features['product_count']}",
        f"slack:{features['slack_bucket']}",
        f"threshold:{features['threshold_bucket']}",
        f"combo:{features['voucher_type']}:{features['discount_type']}",
        f"hard:{features['voucher_type']}:{features['discount_type']}:{features['slack_bucket']}",
    }


def select_rows(features: list[dict], sample_size: int) -> list[dict]:
    selected: list[dict] = []
    covered: Counter[str] = Counter()
    remaining = sorted(
        features,
        key=lambda item: (
            item["slack_bucket"] != "tight",
            -item["product_count"],
            item["budget_slack"],
            item["query_index"],
        ),
    )

    while remaining and len(selected) < sample_size:
        best = None
        best_score = None
        for item in remaining:
            keys = feature_keys(item)
            novelty = sum(1.0 / (1.0 + covered[key]) for key in keys)
            hard_bonus = 0.5 if item["slack_bucket"] == "tight" else 0.2 if item["slack_bucket"] == "medium" else 0.0
            count_bonus = 0.1 * item["product_count"]
            score = novelty + hard_bonus + count_bonus
            tie_break = (-score, item["budget_slack"], item["query_index"])
            if best_score is None or tie_break < best_score:
                best = item
                best_score = tie_break
        assert best is not None
        selected.append(best)
        for key in feature_keys(best):
            covered[key] += 1
        remaining = [item for item in remaining if item["row_index"] != best["row_index"]]
    return sorted(selected, key=lambda item: item["query_index"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input)
    missing = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected parquet columns: {missing}")

    features = [row_features(index, row) for index, row in df.iterrows()]
    selected = select_rows(features, args.sample_size)
    selected_indices = [item["row_index"] for item in selected]
    probe_df = df.iloc[selected_indices].copy()

    output_parquet = Path(args.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    probe_df.to_parquet(output_parquet, index=False)
    write_jsonl(Path(args.output_jsonl), selected)

    distribution = {
        "voucher_type": dict(Counter(item["voucher_type"] for item in selected)),
        "discount_type": dict(Counter(item["discount_type"] for item in selected)),
        "product_count": dict(Counter(str(item["product_count"]) for item in selected)),
        "slack_bucket": dict(Counter(item["slack_bucket"] for item in selected)),
        "threshold_bucket": dict(Counter(item["threshold_bucket"] for item in selected)),
    }
    report = {
        "input": args.input,
        "output_parquet": str(output_parquet),
        "output_jsonl": args.output_jsonl,
        "sample_size": len(selected),
        "selection_policy": (
            "deterministic greedy coverage over voucher type, discount type, product count, "
            "budget slack bucket, threshold bucket; tie-breaks favor tight budgets and more products"
        ),
        "distribution": distribution,
        "selected": selected,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
