#!/usr/bin/env python3
"""Build deterministic, disjoint calibration/validation probes for sampling search.

The two panels are selected jointly from the RL training split.  The greedy
objective tracks marginal and interaction strata in both panels, while stable
SHA-256 tie breaks make the result independent of pandas/Python hash ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info", "agent_name"}
SPLIT_NAMES = ("calibration16", "validation16")
DIMENSION_WEIGHTS = {
    "voucher": 2.0,
    "discount": 2.0,
    "count": 1.5,
    "slack": 1.5,
    "threshold": 1.5,
    "voucher_discount": 2.5,
    "voucher_count": 1.0,
    "discount_slack": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/shoppingbench_query/train.parquet")
    parser.add_argument("--output-dir", default="dataset/probe/step108_outcome_sampling")
    parser.add_argument("--sample-size", type=int, default=16, help="Rows in each of the two panels.")
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument(
        "--expected-input-size",
        type=int,
        default=675,
        help="Fail if the source row count changes; use 0 to disable.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(row: pd.Series) -> dict[str, Any]:
    value = row["reward_model"]["ground_truth"]
    return json.loads(value) if isinstance(value, str) else dict(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(value, tuple):
        return list(value)
    return [value]


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
        return "near"
    if ratio < 2.0:
        return "mid"
    return "far"


def row_features(position: int, row: pd.Series) -> dict[str, Any]:
    extra = dict(row["extra_info"])
    ground_truth = load_ground_truth(row)
    voucher = dict(ground_truth.get("voucher") or {})
    budget = float(voucher.get("budget") or extra.get("budget") or 0)
    price_after = float(voucher.get("price_after_voucher") or 0)
    threshold = float(voucher.get("threshold") or 0)
    slack = budget - price_after
    voucher_type = str(voucher.get("voucher_type") or extra.get("voucher_type") or "unknown")
    discount_type = str(voucher.get("discount_type") or "unknown")
    product_count = int(extra.get("product_count") or len(as_list(extra.get("reward_product_ids"))))
    return {
        "row_position": int(position),
        "query_index": int(extra.get("index", position)),
        "query": str(extra.get("query") or ""),
        "voucher_type": voucher_type,
        "discount_type": discount_type,
        "product_count": product_count,
        "budget": budget,
        "price_after_voucher": price_after,
        "budget_slack": slack,
        "budget_slack_ratio": slack / budget if budget else None,
        "slack_bucket": slack_bucket(slack, budget),
        "threshold": threshold,
        "threshold_bucket": threshold_bucket(price_after, threshold),
        "prompt_tokens": int(extra.get("prompt_tokens") or 0),
        "reward_product_ids": [str(item) for item in as_list(extra.get("reward_product_ids"))],
    }


def strata(feature: dict[str, Any]) -> dict[str, str]:
    voucher = feature["voucher_type"]
    discount = feature["discount_type"]
    count = str(feature["product_count"])
    slack = feature["slack_bucket"]
    threshold = feature["threshold_bucket"]
    return {
        "voucher": f"voucher:{voucher}",
        "discount": f"discount:{discount}",
        "count": f"count:{count}",
        "slack": f"slack:{slack}",
        "threshold": f"threshold:{threshold}",
        "voucher_discount": f"voucher_discount:{voucher}:{discount}",
        "voucher_count": f"voucher_count:{voucher}:{count}",
        "discount_slack": f"discount_slack:{discount}:{slack}",
    }


def stable_tie(seed: int, split: str, feature: dict[str, Any]) -> str:
    payload = f"{seed}:{split}:{feature['query_index']}:{feature['row_position']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_joint_panels(features: list[dict[str, Any]], size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    if 2 * size > len(features):
        raise ValueError(f"Need {2 * size} distinct rows, but input has only {len(features)}")

    keyed = {item["row_position"]: strata(item) for item in features}
    population: Counter[str] = Counter(key for keys in keyed.values() for key in keys.values())
    targets = {key: size * count / len(features) for key, count in population.items()}
    selected = {name: [] for name in SPLIT_NAMES}
    counts = {name: Counter() for name in SPLIT_NAMES}
    used: set[int] = set()

    def candidate_score(split: str, feature: dict[str, Any]) -> float:
        other = SPLIT_NAMES[1] if split == SPLIT_NAMES[0] else SPLIT_NAMES[0]
        score = 0.0
        for dimension, key in keyed[feature["row_position"]].items():
            weight = DIMENSION_WEIGHTS[dimension]
            target = targets[key]
            current = counts[split][key]
            peer = counts[other][key]
            deficit = max(0.0, target - current) / max(1.0, target)
            coverage = 0.30 if current == 0 and target >= 0.35 else 0.0
            rarity = 0.04 * math.sqrt(len(features) / population[key])
            before_gap = abs(current - peer)
            after_gap = abs((current + 1) - peer)
            balance = 0.18 * (before_gap - after_gap)
            score += weight * (deficit + coverage + rarity + balance)
        return score

    for round_index in range(size):
        order = SPLIT_NAMES if round_index % 2 == 0 else tuple(reversed(SPLIT_NAMES))
        for split in order:
            candidates = [item for item in features if item["row_position"] not in used]
            best = min(
                candidates,
                key=lambda item: (
                    -candidate_score(split, item),
                    stable_tie(seed, split, item),
                    item["query_index"],
                    item["row_position"],
                ),
            )
            selected[split].append(best)
            used.add(best["row_position"])
            counts[split].update(keyed[best["row_position"]].values())

    for name in SPLIT_NAMES:
        selected[name].sort(key=lambda item: (item["query_index"], item["row_position"]))
    return selected


def distribution(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dimensions = ("voucher_type", "discount_type", "product_count", "slack_bucket", "threshold_bucket")
    return {
        dimension: dict(sorted(Counter(str(item[dimension]) for item in items).items()))
        for dimension in dimensions
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir)
    outputs = [
        *(output_dir / f"{name}.parquet" for name in SPLIT_NAMES),
        *(output_dir / f"{name}.json" for name in SPLIT_NAMES),
        output_dir / "report.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit("Refusing to replace existing outputs without --force: " + ", ".join(map(str, existing)))

    frame = pd.read_parquet(input_path)
    missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing expected parquet columns: {missing}")
    if args.expected_input_size and len(frame) != args.expected_input_size:
        raise SystemExit(f"Expected {args.expected_input_size} input rows, found {len(frame)}")

    features = [row_features(position, row) for position, (_, row) in enumerate(frame.iterrows())]
    panels = select_joint_panels(features, args.sample_size, args.seed)
    calibration_ids = {item["row_position"] for item in panels[SPLIT_NAMES[0]]}
    validation_ids = {item["row_position"] for item in panels[SPLIT_NAMES[1]]}
    if calibration_ids & validation_ids:
        raise AssertionError("Calibration and validation panels overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        positions = [item["row_position"] for item in panels[name]]
        split_frame = frame.iloc[positions].copy()
        parquet_path = output_dir / f"{name}.parquet"
        json_path = output_dir / f"{name}.json"
        split_frame.to_parquet(parquet_path, index=False)
        split_payload = {
            "name": name,
            "seed": args.seed,
            "source": str(input_path),
            "source_sha256": sha256_file(input_path),
            "rows": len(panels[name]),
            "distribution": distribution(panels[name]),
            "selected": panels[name],
        }
        write_json(json_path, split_payload)
        split_reports[name] = {
            "parquet": str(parquet_path),
            "parquet_sha256": sha256_file(parquet_path),
            "json": str(json_path),
            "json_sha256": sha256_file(json_path),
            "rows": len(panels[name]),
            "distribution": split_payload["distribution"],
            "query_indices": [item["query_index"] for item in panels[name]],
            "row_positions": positions,
        }

    report = {
        "schema_version": 1,
        "selection": "joint deterministic greedy stratification",
        "seed": args.seed,
        "sample_size_per_split": args.sample_size,
        "source": str(input_path),
        "source_rows": len(frame),
        "source_sha256": sha256_file(input_path),
        "stratification_dimensions": list(DIMENSION_WEIGHTS),
        "dimension_weights": DIMENSION_WEIGHTS,
        "disjoint": not bool(calibration_ids & validation_ids),
        "splits": split_reports,
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
