#!/usr/bin/env python3
"""Summarize a state-local stage report with optional harness skip cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUMMARY_KEYS = [
    "success",
    "progress",
    "find_correct",
    "view_confirmed",
    "budget_correct",
    "recommend_correct",
    "terminate_complete",
    "workflow_valid",
    "format",
    "tool_valid",
    "steps",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: mean(rows, key) for key in SUMMARY_KEYS}


def failure_modes(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("structured_failure_mode") or "missing") for row in rows))


def summarize_by_failure_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("structured_failure_mode") or "missing")].append(row)

    output: dict[str, dict[str, Any]] = {}
    for mode in sorted(grouped):
        mode_rows = grouped[mode]
        output[mode] = {"count": len(mode_rows), **summarize(mode_rows)}
    return output


def parse_skip_cases(skip_list: dict[str, Any]) -> dict[int, dict[str, Any]]:
    cases: dict[int, dict[str, Any]] = {}
    for item in skip_list.get("skip_cases") or []:
        if "idx" not in item:
            raise ValueError(f"skip case missing idx: {item!r}")
        idx = int(item["idx"])
        if idx in cases:
            raise ValueError(f"duplicate skip idx: {idx}")
        cases[idx] = dict(item)
    return cases


def build_report(stage_report: dict[str, Any], skip_list: dict[str, Any]) -> dict[str, Any]:
    rows = list(stage_report.get("per_query") or [])
    skip_cases = parse_skip_cases(skip_list)
    row_indices = {int(row["idx"]) for row in rows if "idx" in row}
    missing = sorted(idx for idx in skip_cases if idx not in row_indices)
    if missing:
        raise ValueError(f"skip idx not found in stage report: {missing}")

    kept_rows = [row for row in rows if int(row.get("idx", -1)) not in skip_cases]
    skipped_rows = [row for row in rows if int(row.get("idx", -1)) in skip_cases]

    skipped_cases = []
    for row in skipped_rows:
        idx = int(row["idx"])
        skipped_cases.append(
            {
                "idx": idx,
                "label": skip_cases[idx].get("label"),
                "reason": skip_cases[idx].get("reason"),
                "structured_failure_mode": row.get("structured_failure_mode"),
                "success": row.get("success"),
                "steps": row.get("steps"),
                "query": row.get("query"),
                "expected_ids": row.get("expected_ids"),
                "recommended_ids": row.get("recommended_ids"),
            }
        )

    return {
        "stage_report": stage_report.get("rollout_file"),
        "skip_scope": skip_list.get("scope"),
        "skip_indices": sorted(skip_cases),
        "counts": {
            "full": len(rows),
            "kept": len(kept_rows),
            "skipped": len(skipped_rows),
        },
        "full_summary": summarize(rows),
        "kept_summary": summarize(kept_rows),
        "skipped_summary": summarize(skipped_rows),
        "full_failure_modes": failure_modes(rows),
        "kept_failure_modes": failure_modes(kept_rows),
        "skipped_failure_modes": failure_modes(skipped_rows),
        "kept_summary_by_failure_mode": summarize_by_failure_mode(kept_rows),
        "skipped_cases": skipped_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-report", required=True, type=Path)
    parser.add_argument("--skip-list", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(load_json(args.stage_report), load_json(args.skip_list))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
