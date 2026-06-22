#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RL_SRC = ROOT / "src" / "rl"
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from verl.utils.reward_score import shoppingbench_query  # noqa: E402


def load_ground_truth_by_expected_ids(probe_parquet: Path) -> dict[str, str]:
    df = pd.read_parquet(probe_parquet)
    ground_truth_by_expected_ids = {}
    for _, row in df.iterrows():
        ground_truth = row.reward_model["ground_truth"]
        parsed = json.loads(ground_truth)
        expected_ids = ",".join(str(item["product_id"]) for item in parsed.get("reward") or [])
        ground_truth_by_expected_ids[expected_ids] = ground_truth
    return ground_truth_by_expected_ids


def iter_rollout_rows(rollout_root: Path):
    files = sorted(
        rollout_root.glob("global_step_*/0.jsonl"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    for path in files:
        step = int(path.parent.name.split("_")[-1])
        with path.open(encoding="utf-8") as fin:
            for row_index, line in enumerate(fin):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_step"] = step
                row["_row"] = row_index
                row["_path"] = str(path)
                yield row


def score_rows(rollout_root: Path, probe_parquet: Path) -> list[dict]:
    ground_truth_by_expected_ids = load_ground_truth_by_expected_ids(probe_parquet)
    rows = []
    for row in iter_rollout_rows(rollout_root):
        expected_ids = row.get("expected_ids")
        ground_truth = ground_truth_by_expected_ids.get(expected_ids)
        if ground_truth is None:
            raise KeyError(f"No ground truth for expected_ids={expected_ids!r}")
        result = shoppingbench_query.compute_score(
            row.get("output", ""),
            ground_truth,
            extra_info={
                "global_step": row["_step"],
                "total_training_steps": 256,
            },
        )
        rows.append(
            {
                "checkpoint": f"global_step_{row['_step']}",
                "step": row["_step"],
                "row": row["_row"],
                "old_score": row.get("score"),
                "old_task": row.get("task"),
                "new_score": result["score"],
                "new_task": result["task"],
                "new_progress": result["progress"],
                "new_outcome": result["outcome"],
                "protocol": result["protocol"],
                "format": result["format"],
                "tool_valid": result["tool_valid"],
                "search_gold_recall": result["search_gold_recall"],
                "select_gold_f1": result["select_gold_f1"],
                "verify_gold_f1": result["verify_gold_f1"],
                "shop_constraint_correct": result["shop_constraint_correct"],
                "budget_attempted": result["budget_attempted"],
                "budget_attempt_quality": result.get("budget_attempt_quality"),
                "budget_recomputed_correct": result["budget_recomputed_correct"],
                "budget_numeric_alignment": result.get("budget_numeric_alignment"),
                "within_budget_correct": result["within_budget_correct"],
                "recommend_gold_overlap": result["recommend_gold_overlap"],
                "recommend_gold_precision": result["recommend_gold_precision"],
                "recommend_gold_f1": result["recommend_gold_f1"],
                "recommend_count_match": result["recommend_count_match"],
                "exact": result["exact"],
                "ordered_exact": result.get("ordered_exact"),
                "set_exact": result.get("set_exact"),
                "budget": result["budget"],
                "success": result["success"],
                "terminate": result["terminate"],
                "terminate_quality": result.get("terminate_quality"),
                "wrong_recommend_penalty": result.get("wrong_recommend_penalty"),
                "count_penalty": result.get("count_penalty"),
                "premature_terminate_penalty": result.get("premature_terminate_penalty"),
                "invalid_tool_penalty": result.get("invalid_tool_penalty"),
                "penalties": result.get("penalties"),
                "steps": result["steps"],
                "state_count": result["state_count"],
                "event_count": result["event_count"],
                "recommended_count": result["recommended_count"],
                "expected_count": result["expected_count"],
                "recommended_ids": result["recommended_ids"],
                "expected_ids": result["expected_ids"],
                "source_file": row["_path"],
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def var(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict]) -> dict:
    tasks = [float(row["new_task"]) for row in rows]
    old_tasks = [float(row["old_task"]) for row in rows if row.get("old_task") is not None]
    wrong_final_high = [
        row
        for row in rows
        if row["recommended_count"] > 0 and row["recommend_gold_f1"] == 0 and row["new_task"] > 0.20
    ]
    long_list_high = [
        row
        for row in rows
        if row["recommended_count"] > row["expected_count"]
        and row["recommend_gold_f1"] > 0
        and row["new_task"] > 0.35
    ]
    partial_signal_lost = [
        row
        for row in rows
        if (
            row["search_gold_recall"] > 0
            or row["select_gold_f1"] > 0
            or row["verify_gold_f1"] > 0
            or row["recommend_gold_f1"] > 0
        )
        and row["new_task"] < -0.05
    ]
    high_old_wrong = [
        row
        for row in rows
        if row.get("old_task") is not None
        and row["recommended_count"] > 0
        and row["recommend_gold_f1"] == 0
        and float(row["old_task"]) > 1.0
    ]
    return {
        "rows": len(rows),
        "task_mean": mean(tasks),
        "task_min": min(tasks) if tasks else None,
        "task_max": max(tasks) if tasks else None,
        "task_var": var(tasks),
        "old_task_mean": mean(old_tasks),
        "wrong_final_high_count": len(wrong_final_high),
        "long_list_high_count": len(long_list_high),
        "partial_signal_lost_count": len(partial_signal_lost),
        "old_wrong_final_high_count": len(high_old_wrong),
        "success_count": sum(1 for row in rows if row["success"]),
        "exact_count": sum(1 for row in rows if row["exact"]),
        "budget_success_count": sum(1 for row in rows if row["budget"]),
        "top_new_task": sorted(rows, key=lambda row: row["new_task"], reverse=True)[:12],
        "wrong_final_high": wrong_final_high[:12],
        "long_list_high": long_list_high[:12],
        "partial_signal_lost": partial_signal_lost[:12],
        "old_wrong_final_high_examples": high_old_wrong[:12],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay ShoppingBench rollout JSONL files with the current task reward.")
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--probe-parquet", default="dataset/probe/sft_probe_query_8_statefolded_20260620.parquet")
    parser.add_argument("--output-prefix", default="plots/task_reward_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = score_rows(ROOT / args.rollout_root, ROOT / args.probe_parquet)
    summary = summarize(rows)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    output_prefix.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
