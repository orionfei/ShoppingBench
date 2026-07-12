#!/usr/bin/env python3
"""Paired Step108/step23 test250 analysis with task-bucket breakdowns."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def query_from_input(text: str) -> str:
    return text.rsplit("\nuser\n", 1)[1].rsplit("\nassistant\n", 1)[0]


def load_groups(path: Path) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            groups[query_from_input(row["input"])].append(float(row["terminal_asr"]))
    if len(groups) != 250 or any(len(values) != 8 for values in groups.values()):
        raise ValueError(f"expected 250 complete G8 groups in {path}")
    return groups


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.parquet)
    meta = {row.extra_info["query"]: row.extra_info for _, row in frame.iterrows()}
    models = {
        "Step108": load_groups(args.test_root / "untouched_step108/0.jsonl"),
        "step23": load_groups(args.test_root / "best_step23/0.jsonl"),
    }
    queries = sorted(models["Step108"])
    differences = np.array([np.mean(models["step23"][q]) - np.mean(models["Step108"][q]) for q in queries])
    rng = np.random.default_rng(108)
    draws = np.mean(rng.choice(differences, (20_000, len(differences)), replace=True), axis=1)
    paired = {
        "mean_difference": float(np.mean(differences)),
        "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "bootstrap_probability_positive": float(np.mean(draws > 0)),
    }

    bucket_rows = []
    dimensions = {"product_count": [1, 2, 3, 4], "voucher_type": ["platform", "shop"]}
    for dimension, values in dimensions.items():
        for value in values:
            selected = [q for q in queries if meta[q][dimension] == value]
            for model, groups in models.items():
                scores = [score for q in selected for score in groups[q]]
                bucket_rows.append({"dimension": dimension, "bucket": str(value), "model": model,
                                    "queries": len(selected), "trajectories": len(scores),
                                    "terminal_asr": float(np.mean(scores))})
    with (args.output_dir / "bucket_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bucket_rows[0].keys())
        writer.writeheader(); writer.writerows(bucket_rows)
    (args.output_dir / "paired_test_summary.json").write_text(
        json.dumps({"paired_terminal_asr": paired, "buckets": bucket_rows}, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (dimension, values) in zip(axes, dimensions.items()):
        x = np.arange(len(values)); width = .36
        for offset, model in ((-.5, "Step108"), (.5, "step23")):
            ys = [next(row["terminal_asr"] for row in bucket_rows
                       if row["dimension"] == dimension and row["bucket"] == str(value) and row["model"] == model)
                  for value in values]
            ax.bar(x + offset * width, ys, width, label=model)
        ax.set_xticks(x, [str(value) for value in values]); ax.set_xlabel(dimension); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("terminal ASR"); axes[0].legend()
    savefig(fig, args.figure_dir / "test250_bucket_asr")
    print(json.dumps(paired))


if __name__ == "__main__":
    main()
