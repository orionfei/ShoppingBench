#!/usr/bin/env python3
"""Summarize grouped ShoppingBench query rollout dumps from verl validation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_METRICS = [
    "score",
    "reward",
    "protocol",
    "format",
    "tool_valid",
    "task",
    "progress",
    "outcome",
    "final_success",
    "search_gold_recall",
    "select_gold_f1",
    "select_attribute_f1",
    "select_relevance_f1",
    "verify_gold_f1",
    "verify_attribute_f1",
    "verify_relevance_f1",
    "budget_attempted",
    "budget_attribute_f1",
    "budget_relevance_f1",
    "budget_recomputed_correct",
    "budget_numeric_alignment",
    "within_budget_correct",
    "recommend_gold_f1",
    "recommend_attribute_f1",
    "recommend_relevance_f1",
    "recommend_count_match",
    "set_exact",
    "attribute_set_exact",
    "semantic_set_exact",
    "terminate_after_valid_recommend",
    "steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSONL files or directories containing verl dumps.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--selection-summary", action="store_true")
    return parser.parse_args()


def expand_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)
    return files


def read_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def var(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def short_query(input_text: str) -> str:
    marker = "\nuser\n"
    if marker in input_text:
        text = input_text.split(marker)[-1]
    else:
        text = input_text
    if "\nassistant\n" in text:
        text = text.split("\nassistant\n")[0]
    text = " ".join(text.strip().split())
    return text[:220]


def summarize(rows: list[dict[str, Any]], group_size: int) -> dict[str, Any]:
    by_input: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_input[str(row.get("input", ""))].append(row)

    query_summaries = []
    for input_text, items in sorted(by_input.items(), key=lambda item: short_query(item[0])):
        summary: dict[str, Any] = {
            "query": short_query(input_text),
            "samples": len(items),
            "complete_group": len(items) >= group_size,
        }
        for metric in CORE_METRICS:
            vals = [as_float(item.get(metric)) for item in items]
            vals = [item for item in vals if item is not None]
            if vals:
                summary[f"{metric}_mean"] = mean(vals)
                summary[f"{metric}_var"] = var(vals)
                summary[f"{metric}_min"] = min(vals)
                summary[f"{metric}_max"] = max(vals)
        recommended = [str(item.get("recommended_ids") or "") for item in items]
        summary["recommended_unique"] = len(set(recommended))
        summary["recommended_nonempty_rate"] = mean([1.0 if item else 0.0 for item in recommended])
        query_summaries.append(summary)

    aggregate: dict[str, Any] = {
        "rows": len(rows),
        "queries": len(query_summaries),
        "complete_groups": sum(1 for item in query_summaries if item["complete_group"]),
        "group_size": group_size,
    }
    for metric in CORE_METRICS:
        metric_means = [item[f"{metric}_mean"] for item in query_summaries if f"{metric}_mean" in item]
        metric_vars = [item[f"{metric}_var"] for item in query_summaries if f"{metric}_var" in item]
        aggregate[f"{metric}_mean"] = mean(metric_means)
        aggregate[f"{metric}_group_var_mean"] = mean(metric_vars)
    aggregate["recommended_unique_mean"] = mean([float(item["recommended_unique"]) for item in query_summaries])
    aggregate["recommended_nonempty_rate_mean"] = mean(
        [float(item["recommended_nonempty_rate"]) for item in query_summaries]
    )

    return {
        "aggregate": aggregate,
        "queries": query_summaries,
    }


def checkpoint_name(path: Path) -> str:
    parts = path.parts
    for part in reversed(parts):
        if part.startswith("global_step_"):
            return part
    return path.name


def selection_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for report in reports:
        agg = report.get("aggregate", {})
        protocol_mean = float(agg.get("protocol_mean", 0.0))
        protocol_var = float(agg.get("protocol_group_var_mean", 1e9))
        task_var = float(agg.get("task_group_var_mean", 0.0))
        task_mean = float(agg.get("task_mean", 0.0))
        progress_mean = float(agg.get("progress_mean", 0.0))
        eligible = protocol_mean >= 0.90 and protocol_var <= 0.05
        candidates.append(
            {
                "checkpoint": report.get("checkpoint"),
                "eligible": eligible,
                "protocol_mean": protocol_mean,
                "protocol_group_var_mean": protocol_var,
                "task_mean": task_mean,
                "task_group_var_mean": task_var,
                "progress_mean": progress_mean,
                "recommend_gold_f1_mean": agg.get("recommend_gold_f1_mean", 0.0),
                "within_budget_correct_mean": agg.get("within_budget_correct_mean", 0.0),
                "final_success_mean": agg.get("final_success_mean", 0.0),
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["task_group_var_mean"], item["task_mean"])) if eligible else None
    return {
        "rule": "protocol_mean >= 0.90 and protocol_group_var_mean <= 0.05; among eligible choose max task_group_var_mean, tie task_mean",
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    reports = []
    for raw in args.paths:
        path = Path(raw)
        files = expand_paths([raw])
        rows = read_rows(files)
        report = {
            "path": raw,
            "checkpoint": checkpoint_name(path),
            "files": [str(item) for item in files],
            **summarize(rows, args.group_size),
        }
        reports.append(report)

    payload: dict[str, Any]
    if len(reports) == 1 and not args.selection_summary:
        payload = reports[0]
    else:
        payload = {"reports": reports}
        if args.selection_summary:
            payload["selection"] = selection_summary(reports)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        output.write_text(text + "\n", encoding="utf-8")
        if "reports" in payload:
            compact = {
                "output": str(output),
                "selection": payload.get("selection"),
                "checkpoints": [
                    {
                        "checkpoint": item.get("checkpoint"),
                        "rows": item.get("aggregate", {}).get("rows"),
                        "protocol_mean": item.get("aggregate", {}).get("protocol_mean"),
                        "protocol_group_var_mean": item.get("aggregate", {}).get("protocol_group_var_mean"),
                        "task_mean": item.get("aggregate", {}).get("task_mean"),
                        "task_group_var_mean": item.get("aggregate", {}).get("task_group_var_mean"),
                        "progress_mean": item.get("aggregate", {}).get("progress_mean"),
                        "final_success_mean": item.get("aggregate", {}).get("final_success_mean"),
                    }
                    for item in payload["reports"]
                ],
            }
        else:
            agg = payload.get("aggregate", {})
            compact = {
                "output": str(output),
                "checkpoint": payload.get("checkpoint"),
                "rows": agg.get("rows"),
                "queries": agg.get("queries"),
                "protocol_mean": agg.get("protocol_mean"),
                "protocol_group_var_mean": agg.get("protocol_group_var_mean"),
                "task_mean": agg.get("task_mean"),
                "task_group_var_mean": agg.get("task_group_var_mean"),
                "progress_mean": agg.get("progress_mean"),
                "final_success_mean": agg.get("final_success_mean"),
            }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
