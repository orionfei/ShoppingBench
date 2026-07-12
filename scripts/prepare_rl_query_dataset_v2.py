#!/usr/bin/env python3
"""Build the frozen Step108 RL query split without regenerating query text.

The existing 750-query corpus remains the source of truth.  This builder:

* preserves the 16-query sampling calibration and online-validation panels;
* repairs product-id leakage in the 75-query final test by deterministic swaps;
* assigns every remaining query to the 643-query RL train split;
* enriches extra_info with stable ids and curriculum/audit features; and
* refuses to write a split that is not query-disjoint and test-product-disjoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info", "agent_name"]
SPLIT_ORDER = ("train", "validation", "calibration", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="dataset/shoppingbench_query")
    parser.add_argument("--probe-dir", default="dataset/probe/step108_outcome_sampling")
    parser.add_argument("--product-cache", default="dataset/shoppingbench_query/product_cache.json")
    parser.add_argument("--output-dir", default="dataset/shoppingbench_query_rl_v2")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_query_id(query: str) -> str:
    normalized = " ".join(query.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def row_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    ground_truth = plain(row["reward_model"])["ground_truth"]
    return json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)


def row_query(row: dict[str, Any]) -> str:
    return str(plain(row["extra_info"]).get("query") or "")


def row_product_ids(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["product_id"]) for item in row_ground_truth(row).get("reward") or [])


def constraint_count(rewards: list[dict[str, Any]]) -> int:
    total = 0
    for reward in rewards:
        total += len(reward.get("title") or [])
        total += len(reward.get("service") or [])
        for field in ("attributes", "sku_options", "price"):
            for item in reward.get(field) or []:
                if not isinstance(item, dict):
                    continue
                for value in item.values():
                    total += len(value) if isinstance(value, list) else 1
    return total


def feature_record(row: dict[str, Any], products: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query = row_query(row)
    extra = plain(row["extra_info"])
    ground_truth = row_ground_truth(row)
    rewards = list(ground_truth.get("reward") or [])
    voucher = dict(ground_truth.get("voucher") or {})
    product_ids = tuple(str(item["product_id"]) for item in rewards)
    budget = float(voucher.get("budget") or 0)
    price_after = float(voucher.get("price_after_voucher") or 0)
    slack = budget - price_after
    slack_ratio = slack / budget if budget else math.nan
    total_price = sum(float(products[product_id]["price"]) for product_id in product_ids)
    threshold = float(voucher.get("threshold") or 0)
    threshold_ratio = threshold / total_price if total_price else math.nan
    if slack_ratio <= 0.03:
        budget_difficulty = "hard"
    elif slack_ratio <= 0.06:
        budget_difficulty = "medium"
    else:
        budget_difficulty = "easy"
    if threshold_ratio >= 0.80:
        threshold_difficulty = "hard"
    elif threshold_ratio >= 0.50:
        threshold_difficulty = "medium"
    else:
        threshold_difficulty = "easy"
    constraints = constraint_count(rewards)
    complexity = "low" if constraints <= 5 else "medium" if constraints <= 9 else "high"
    return {
        "query_id": stable_query_id(query),
        "query": query,
        "voucher_type": str(voucher.get("voucher_type") or "unknown"),
        "discount_type": str(voucher.get("discount_type") or "unknown"),
        "product_count": len(rewards),
        "reward_product_ids": list(product_ids),
        "constraint_count": constraints,
        "constraint_complexity": complexity,
        "budget": budget,
        "price_after_voucher": price_after,
        "budget_slack": slack,
        "budget_slack_ratio": slack_ratio,
        "budget_difficulty": budget_difficulty,
        "ground_truth_total_price": total_price,
        "voucher_threshold": threshold,
        "threshold_ratio": threshold_ratio,
        "threshold_difficulty": threshold_difficulty,
        "prompt_tokens": int(extra.get("prompt_tokens") or 0),
        "source_file": str((extra.get("source_meta") or {}).get("source_file") or "unknown"),
        "source_index": int((extra.get("source_meta") or {}).get("source_index") or 0),
        "original_split": str(extra.get("split") or "unknown"),
        "original_split_index": int(extra.get("index") or 0),
    }


def replacement_stratum(feature: dict[str, Any]) -> tuple[Any, ...]:
    return (
        feature["voucher_type"],
        feature["discount_type"],
        feature["product_count"],
        feature["budget_difficulty"],
        feature["source_file"],
    )


def fallback_distance(target: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(target["voucher_type"] != candidate["voucher_type"]),
        int(target["discount_type"] != candidate["discount_type"]),
        abs(target["product_count"] - candidate["product_count"]),
        int(target["budget_difficulty"] != candidate["budget_difficulty"]),
        int(target["source_file"] != candidate["source_file"]),
        abs(target["constraint_count"] - candidate["constraint_count"]),
    )


def tie(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def ideal_budget_success(feature: dict[str, Any], voucher: dict[str, Any], products: dict[str, dict[str, Any]]) -> bool:
    selected = [products[product_id] for product_id in feature["reward_product_ids"]]
    total_price = sum(float(product["price"]) for product in selected)
    budget = float(voucher["budget"])
    if total_price <= budget:
        return True
    shop_ids = {str(product.get("shop_id")) for product in selected}
    applicable = voucher.get("voucher_type") == "platform" or (
        voucher.get("voucher_type") == "shop" and len(shop_ids) == 1
    )
    if not applicable or total_price < float(voucher["threshold"]):
        return False
    if voucher.get("discount_type") == "fixed":
        payable = total_price - float(voucher["face_value"])
    elif voucher.get("discount_type") == "percentage":
        payable = max(
            total_price * (1.0 - float(voucher["discount"])),
            total_price - float(voucher["cap"]),
        )
    else:
        return False
    return payable <= budget


def distribution(features: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dimensions = (
        "voucher_type", "discount_type", "product_count", "budget_difficulty",
        "threshold_difficulty", "constraint_complexity", "source_file",
    )
    return {
        dimension: dict(sorted(Counter(str(feature[dimension]) for feature in features).items()))
        for dimension in dimensions
    }


def enrich_row(row: dict[str, Any], split: str, index: int, feature: dict[str, Any]) -> dict[str, Any]:
    result = {column: plain(row[column]) for column in EXPECTED_COLUMNS}
    extra = dict(result["extra_info"])
    extra.update({
        "split": split,
        "index": index,
        "query_id": feature["query_id"],
        "original_split": feature["original_split"],
        "original_split_index": feature["original_split_index"],
        "voucher_type": feature["voucher_type"],
        "discount_type": feature["discount_type"],
        "product_count": feature["product_count"],
        "reward_product_ids": feature["reward_product_ids"],
        "constraint_count": feature["constraint_count"],
        "constraint_complexity": feature["constraint_complexity"],
        "budget": feature["budget"],
        "price_after_voucher": feature["price_after_voucher"],
        "budget_slack": feature["budget_slack"],
        "budget_slack_ratio": feature["budget_slack_ratio"],
        "budget_difficulty": feature["budget_difficulty"],
        "ground_truth_total_price": feature["ground_truth_total_price"],
        "voucher_threshold": feature["voucher_threshold"],
        "threshold_ratio": feature["threshold_ratio"],
        "threshold_difficulty": feature["threshold_difficulty"],
    })
    result["extra_info"] = extra
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    probe_dir = Path(args.probe_dir)
    product_cache_path = Path(args.product_cache)
    output_dir = Path(args.output_dir)
    outputs = [*(output_dir / f"{name}.parquet" for name in SPLIT_ORDER), output_dir / "manifest.json", output_dir / "report.json"]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit("Refusing to replace existing RL v2 outputs without --force: " + ", ".join(map(str, existing)))

    source_frames = {
        "train": pd.read_parquet(source_dir / "train.parquet"),
        "test": pd.read_parquet(source_dir / "test.parquet"),
    }
    for name, frame in source_frames.items():
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise SystemExit(f"Unexpected {name} columns: {list(frame.columns)}")
    if {name: len(frame) for name, frame in source_frames.items()} != {"train": 675, "test": 75}:
        raise SystemExit("Expected the frozen 675/75 source split")

    rows: dict[str, dict[str, Any]] = {}
    original_split: dict[str, str] = {}
    for split, frame in source_frames.items():
        for record in frame.to_dict("records"):
            record = plain(record)
            query_id = stable_query_id(row_query(record))
            if query_id in rows:
                raise AssertionError(f"Duplicate source query: {query_id}")
            rows[query_id] = record
            original_split[query_id] = split

    products = {str(key): value for key, value in json.loads(product_cache_path.read_text(encoding="utf-8")).items()}
    features = {query_id: feature_record(row, products) for query_id, row in rows.items()}
    missing_products = sorted({product_id for feature in features.values() for product_id in feature["reward_product_ids"] if product_id not in products})
    if missing_products:
        raise AssertionError(f"Missing products in cache: {missing_products[:10]}")

    probe_ids: dict[str, set[str]] = {}
    for output_name, source_name in (("calibration", "calibration16"), ("validation", "validation16")):
        frame = pd.read_parquet(probe_dir / f"{source_name}.parquet")
        probe_ids[output_name] = {stable_query_id(row_query(plain(record))) for record in frame.to_dict("records")}
        if len(probe_ids[output_name]) != 16 or not probe_ids[output_name] <= set(rows):
            raise AssertionError(f"Invalid frozen {source_name} panel")
    if probe_ids["calibration"] & probe_ids["validation"]:
        raise AssertionError("Calibration and validation overlap")

    product_occurrences = Counter(product_id for feature in features.values() for product_id in feature["reward_product_ids"])
    current_test = {query_id for query_id, split in original_split.items() if split == "test"}
    current_non_test = set(rows) - current_test
    non_test_products = {product_id for query_id in current_non_test for product_id in features[query_id]["reward_product_ids"]}
    conflicting_test = {
        query_id for query_id in current_test
        if set(features[query_id]["reward_product_ids"]) & non_test_products
    }
    original_overlap_ids = sorted({
        product_id
        for query_id in conflicting_test
        for product_id in features[query_id]["reward_product_ids"]
        if product_id in non_test_products
    })
    repaired_test = current_test - conflicting_test
    used_replacements: set[str] = set()
    swaps: list[dict[str, Any]] = []
    eligible_replacements = [
        query_id for query_id in current_non_test - probe_ids["calibration"] - probe_ids["validation"]
        if all(product_occurrences[product_id] == 1 for product_id in features[query_id]["reward_product_ids"])
    ]
    for removed_id in sorted(conflicting_test):
        target = features[removed_id]
        candidates = [query_id for query_id in eligible_replacements if query_id not in used_replacements]
        exact = [query_id for query_id in candidates if replacement_stratum(features[query_id]) == replacement_stratum(target)]
        pool = exact or candidates
        replacement_id = min(
            pool,
            key=lambda query_id: (fallback_distance(target, features[query_id]), tie(args.seed, query_id)),
        )
        used_replacements.add(replacement_id)
        repaired_test.add(replacement_id)
        swaps.append({
            "removed_from_test_query_id": removed_id,
            "removed_source_index": target["source_index"],
            "removed_product_ids": target["reward_product_ids"],
            "overlapping_product_ids": sorted(set(target["reward_product_ids"]) & non_test_products),
            "replacement_query_id": replacement_id,
            "replacement_source_index": features[replacement_id]["source_index"],
            "replacement_product_ids": features[replacement_id]["reward_product_ids"],
            "exact_stratum_match": replacement_stratum(features[replacement_id]) == replacement_stratum(target),
            "stratum": list(replacement_stratum(target)),
        })

    split_ids = {
        "calibration": probe_ids["calibration"],
        "validation": probe_ids["validation"],
        "test": repaired_test,
    }
    split_ids["train"] = set(rows) - set().union(*split_ids.values())
    expected_counts = {"train": 643, "validation": 16, "calibration": 16, "test": 75}
    if {name: len(ids) for name, ids in split_ids.items()} != expected_counts:
        raise AssertionError(f"Unexpected split counts: { {name: len(ids) for name, ids in split_ids.items()} }")
    if set().union(*split_ids.values()) != set(rows) or sum(map(len, split_ids.values())) != len(rows):
        raise AssertionError("Query partitions do not form a disjoint cover")

    development_ids = split_ids["train"] | split_ids["validation"] | split_ids["calibration"]
    development_products = {product_id for query_id in development_ids for product_id in features[query_id]["reward_product_ids"]}
    final_test_products = {product_id for query_id in split_ids["test"] for product_id in features[query_id]["reward_product_ids"]}
    product_overlap = sorted(development_products & final_test_products)
    if product_overlap:
        raise AssertionError(f"Final test product leakage remains: {product_overlap}")

    ideal_failures = []
    for query_id, row in rows.items():
        voucher = row_ground_truth(row).get("voucher") or {}
        if not ideal_budget_success(features[query_id], voucher, products):
            ideal_failures.append(query_id)
    if ideal_failures:
        raise AssertionError(f"Ideal ground truth fails budget for {len(ideal_failures)} queries")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    manifest_members: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_ORDER:
        ordered_ids = sorted(split_ids[split], key=lambda query_id: tie(args.seed + SPLIT_ORDER.index(split), query_id))
        enriched = [enrich_row(rows[query_id], split, index, features[query_id]) for index, query_id in enumerate(ordered_ids)]
        path = output_dir / f"{split}.parquet"
        pd.DataFrame(enriched, columns=EXPECTED_COLUMNS).to_parquet(path, index=False)
        member_records = [{key: value for key, value in features[query_id].items() if key != "query"} for query_id in ordered_ids]
        manifest_members[split] = member_records
        split_reports[split] = {
            "rows": len(ordered_ids),
            "parquet": str(path),
            "sha256": sha256_file(path),
            "unique_products": len({product_id for query_id in ordered_ids for product_id in features[query_id]["reward_product_ids"]}),
            "max_prompt_tokens": max(features[query_id]["prompt_tokens"] for query_id in ordered_ids),
            "distribution": distribution([features[query_id] for query_id in ordered_ids]),
        }

    manifest = {
        "schema_version": 2,
        "objective": "Step108 outcome-only GRPO query partitions",
        "seed": args.seed,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        },
        "source": {
            "train": str(source_dir / "train.parquet"),
            "train_sha256": sha256_file(source_dir / "train.parquet"),
            "test": str(source_dir / "test.parquet"),
            "test_sha256": sha256_file(source_dir / "test.parquet"),
            "product_cache": str(product_cache_path),
            "product_cache_sha256": sha256_file(product_cache_path),
            "probe_report": str(probe_dir / "report.json"),
            "probe_report_sha256": sha256_file(probe_dir / "report.json"),
        },
        "members": manifest_members,
    }
    write_json(output_dir / "manifest.json", manifest)

    report = {
        "schema_version": 2,
        "decision": "reuse existing queries; repartition and enrich without reward/query regeneration",
        "seed": args.seed,
        "counts": expected_counts,
        "total_queries": len(rows),
        "total_unique_products": len({product_id for feature in features.values() for product_id in feature["reward_product_ids"]}),
        "query_partition_is_disjoint_cover": True,
        "final_test_development_query_overlap": 0,
        "final_test_development_product_overlap": 0,
        "current_sft_query_overlap_inherited_from_source_audit": 0,
        "missing_product_cache_ids": 0,
        "ideal_exact_answer_paper_asr": len(rows),
        "ideal_exact_answer_total": len(rows),
        "test_repair": {
            "original_test_product_overlap_count": len(original_overlap_ids),
            "original_test_product_overlap_ids": original_overlap_ids,
            "conflicting_test_rows": len(conflicting_test),
            "swaps": swaps,
        },
        "splits": split_reports,
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "product_cache": str(product_cache_path),
        "notes": [
            "calibration is an archive for sampling-selection evidence and is excluded from training",
            "validation is the fixed online RL monitor and is excluded from training",
            "test is an internal 75-query holdout and remains untouched until a trained checkpoint exists",
            "all added fields are diagnostics/curriculum metadata and never enter the outcome reward",
        ],
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
