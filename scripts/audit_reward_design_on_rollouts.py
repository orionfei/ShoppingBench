#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RL_SRC = ROOT / "src" / "rl"
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from verl.utils.reward_score import shoppingbench_query  # noqa: E402


DEFAULT_ROLLOUT_FILES = [
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptonly_t08_n4_probe16_8192_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptonly_t06_n4_probe16_8192_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptfix_tokenstop_t01_probe16_8192_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptfix_t08_n4_probe16_8192_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptfix_t01_probe16_8192_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_step414_promptfix_t01_probe16_1024_2gpu_20260701/global_step_414/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_steps138_276_414_probe16_semreward_10240_2gpu_20260701/global_step_138/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_steps138_276_414_probe16_semreward_10240_2gpu_20260701/global_step_276/0.jsonl",
    "rollouts/qwen3_4b_hybrid_findbatch_steps138_276_414_probe16_semreward_10240_2gpu_20260701/global_step_414/0.jsonl",
]

PROGRESS_WEIGHTS = {
    "find_correct": 0.20,
    "view_confirmed": 0.20,
    "budget_correct": 0.20,
    "recommend_correct": 0.30,
    "terminate_complete": 0.10,
}

NUMERIC_FIELDS = [
    "score",
    "task",
    "progress",
    "final_success",
    "format",
    "tool_valid",
    "protocol",
    "exact",
    "budget",
    "terminate",
    "penalties",
    "search_gold_recall",
    "verify_relevance_f1",
    "shop_constraint_correct",
    "budget_attempt_quality",
    "budget_recomputed_correct",
    "budget_numeric_alignment",
    "within_budget_correct",
    "recommend_relevance_f1",
    "terminate_after_valid_recommend",
]


def clean_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def short_text(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_ground_truth_by_expected_ids(probe_parquets: list[Path]) -> dict[str, str]:
    ground_truth_by_expected_ids: dict[str, str] = {}
    for probe_parquet in probe_parquets:
        df = pd.read_parquet(probe_parquet)
        for _, row in df.iterrows():
            ground_truth = row.reward_model["ground_truth"]
            parsed = json.loads(ground_truth)
            expected_ids = ",".join(str(item["product_id"]) for item in parsed.get("reward") or [])
            ground_truth_by_expected_ids[expected_ids] = ground_truth
    return ground_truth_by_expected_ids


def iter_rows(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as fin:
            for row_index, line in enumerate(fin):
                if not line.strip():
                    continue
                row = json.loads(line)
                yield path, row_index, row


def rollout_name(path: Path) -> str:
    try:
        return path.relative_to(ROOT).parts[1]
    except Exception:
        return path.parent.parent.name


def rollout_step(path: Path) -> int:
    name = path.parent.name
    if name.startswith("global_step_"):
        return int(name.rsplit("_", 1)[-1])
    return int(clean_float(name, -1))


def approximate_components(row: dict[str, Any]) -> dict[str, float]:
    find_correct = clean_float(row.get("search_gold_recall"))
    view_confirmed = min(find_correct, clean_float(row.get("verify_relevance_f1")))
    budget_correct = min(
        view_confirmed,
        clean_float(row.get("shop_constraint_correct")),
        clean_float(row.get("budget_attempt_quality")),
        clean_float(row.get("budget_recomputed_correct")),
        clean_float(row.get("budget_numeric_alignment")),
        clean_float(row.get("within_budget_correct")),
    )
    recommend_correct = min(budget_correct, clean_float(row.get("recommend_relevance_f1")))
    terminate_complete = min(recommend_correct, clean_float(row.get("terminate_after_valid_recommend")))
    components = {
        "find_correct": find_correct,
        "view_confirmed": view_confirmed,
        "budget_correct": budget_correct,
        "recommend_correct": recommend_correct,
        "terminate_complete": terminate_complete,
    }
    components["progress"] = sum(PROGRESS_WEIGHTS[key] * components[key] for key in PROGRESS_WEIGHTS)
    return components


def score_output_only(row: dict[str, Any], step: int, ground_truth_by_expected_ids: dict[str, str]) -> dict[str, float]:
    ground_truth = ground_truth_by_expected_ids.get(row.get("expected_ids", ""))
    if ground_truth is None:
        return {"missing_ground_truth": 1.0}
    result = shoppingbench_query.compute_score(
        row.get("output", ""),
        ground_truth,
        extra_info={"global_step": step, "total_training_steps": 256},
    )
    keep = [
        "score",
        "task",
        "progress",
        "format",
        "tool_valid",
        "protocol",
        "workflow_valid",
        "find_correct",
        "view_confirmed",
        "budget_correct",
        "recommend_correct",
        "terminate_complete",
        "event_count",
        "observed_event_count",
        "message_count",
    ]
    return {key: clean_float(result.get(key)) for key in keep}


def build_record(path: Path, row_index: int, row: dict[str, Any], ground_truth_by_expected_ids: dict[str, str]) -> dict[str, Any]:
    step = rollout_step(path)
    approx = approximate_components(row)
    output_only = score_output_only(row, step, ground_truth_by_expected_ids)
    old_penalties = clean_float(row.get("penalties"))
    approx_task_before_hard_cap = approx["progress"] - old_penalties
    old_protocol = clean_float(row.get("protocol"))
    approx_task_with_protocol_cap = min(approx_task_before_hard_cap, 0.0) if old_protocol < 1.0 else approx_task_before_hard_cap
    record = {
        "run": rollout_name(path),
        "checkpoint": path.parent.name,
        "row": row_index,
        "source_file": str(path.relative_to(ROOT)),
        "expected_ids": row.get("expected_ids", ""),
        "recommended_ids": row.get("recommended_ids", ""),
        "output_len": len(row.get("output", "") or ""),
        "output_snippet": short_text(row.get("output", "")),
        "approx_find_correct": approx["find_correct"],
        "approx_view_confirmed": approx["view_confirmed"],
        "approx_budget_correct": approx["budget_correct"],
        "approx_recommend_correct": approx["recommend_correct"],
        "approx_terminate_complete": approx["terminate_complete"],
        "approx_progress": approx["progress"],
        "approx_task_before_hard_cap": approx_task_before_hard_cap,
        "approx_task_with_protocol_cap": approx_task_with_protocol_cap,
        "output_only_missing_ground_truth": output_only.pop("missing_ground_truth", 0.0),
    }
    for field in NUMERIC_FIELDS:
        record[f"old_{field}"] = clean_float(row.get(field))
    for key, value in output_only.items():
        record[f"output_only_{key}"] = value
    return record


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = {
        "highest_approx_progress": sorted(rows, key=lambda row: row["approx_progress"], reverse=True)[:5],
        "old_final_success_but_low_approx_progress": [
            row for row in rows if row["old_final_success"] >= 1.0 and row["approx_progress"] < 0.8
        ][:10],
        "old_positive_task_but_protocol_bad": [
            row for row in rows if row["old_task"] > 0.0 and row["old_protocol"] < 1.0
        ][:10],
        "recommend_signal_without_budget_stage": [
            row
            for row in rows
            if row["old_recommend_relevance_f1"] > 0.0 and row["approx_budget_correct"] <= 0.0
        ][:10],
        "budget_signal_without_view_stage": [
            row
            for row in rows
            if (
                row["old_budget_attempt_quality"] > 0.0
                or row["old_budget_recomputed_correct"] > 0.0
                or row["old_budget_numeric_alignment"] > 0.0
            )
            and row["approx_view_confirmed"] <= 0.0
        ][:10],
    }
    summary = {
        "rows": len(rows),
        "old_score_mean": mean([row["old_score"] for row in rows]),
        "old_task_mean": mean([row["old_task"] for row in rows]),
        "old_progress_mean": mean([row["old_progress"] for row in rows]),
        "old_final_success_mean": mean([row["old_final_success"] for row in rows]),
        "old_protocol_mean": mean([row["old_protocol"] for row in rows]),
        "old_format_mean": mean([row["old_format"] for row in rows]),
        "old_tool_valid_mean": mean([row["old_tool_valid"] for row in rows]),
        "approx_progress_mean": mean([row["approx_progress"] for row in rows]),
        "approx_task_before_hard_cap_mean": mean([row["approx_task_before_hard_cap"] for row in rows]),
        "approx_task_with_protocol_cap_mean": mean([row["approx_task_with_protocol_cap"] for row in rows]),
        "output_only_protocol_mean": mean([row.get("output_only_protocol", 0.0) for row in rows]),
        "output_only_workflow_valid_mean": mean([row.get("output_only_workflow_valid", 0.0) for row in rows]),
        "output_len_max": max([row["output_len"] for row in rows], default=0),
        "old_final_success_count": sum(1 for row in rows if row["old_final_success"] >= 1.0),
        "old_protocol_bad_count": sum(1 for row in rows if row["old_protocol"] < 1.0),
        "old_positive_task_protocol_bad_count": sum(
            1 for row in rows if row["old_task"] > 0.0 and row["old_protocol"] < 1.0
        ),
        "old_final_success_low_approx_progress_count": sum(
            1 for row in rows if row["old_final_success"] >= 1.0 and row["approx_progress"] < 0.8
        ),
        "recommend_signal_without_budget_stage_count": sum(
            1 for row in rows if row["old_recommend_relevance_f1"] > 0.0 and row["approx_budget_correct"] <= 0.0
        ),
        "budget_signal_without_view_stage_count": sum(
            1
            for row in rows
            if (
                row["old_budget_attempt_quality"] > 0.0
                or row["old_budget_recomputed_correct"] > 0.0
                or row["old_budget_numeric_alignment"] > 0.0
            )
            and row["approx_view_confirmed"] <= 0.0
        ),
        "component_means": {
            "find_correct": mean([row["approx_find_correct"] for row in rows]),
            "view_confirmed": mean([row["approx_view_confirmed"] for row in rows]),
            "budget_correct": mean([row["approx_budget_correct"] for row in rows]),
            "recommend_correct": mean([row["approx_recommend_correct"] for row in rows]),
            "terminate_complete": mean([row["approx_terminate_complete"] for row in rows]),
        },
        "examples": examples,
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether ShoppingBench rollout metrics agree with the staged progress reward.")
    parser.add_argument("--rollout-file", action="append", default=[], help="Rollout JSONL file. Defaults to today's probe files.")
    parser.add_argument(
        "--probe-parquet",
        action="append",
        default=[],
        help="Probe parquet used to map expected_ids to ground truth. Can be repeated.",
    )
    parser.add_argument("--output-json", default="reports/reward_design_audit_20260701.json")
    parser.add_argument("--output-csv", default="reports/reward_design_audit_20260701.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rollout_files = [ROOT / item for item in (args.rollout_file or DEFAULT_ROLLOUT_FILES)]
    probe_parquets = [ROOT / item for item in args.probe_parquet] or [
        ROOT / "dataset/probe/qwen3_4b_voucher_probe_20260629.parquet",
        ROOT / "dataset/probe/qwen3_4b_voucher_probe_promptv3_20260630.parquet",
    ]
    missing = [path for path in rollout_files + probe_parquets if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing input: {path}", file=sys.stderr)
        return 2

    ground_truth_by_expected_ids = load_ground_truth_by_expected_ids(probe_parquets)
    rows = [
        build_record(path, row_index, row, ground_truth_by_expected_ids)
        for path, row_index, row in iter_rows(rollout_files)
    ]

    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_run[row["run"]].append(row)

    report = {
        "limitations": [
            "Today rollout JSONL files do not contain full tool observation messages, so exact current task-progress replay is not possible from these files alone.",
            "approx_* metrics use the real per-row rollout metrics that were dumped at generation time, then apply the new five-stage weighting and stage gates where those dumped metrics are sufficient.",
            "output_only_* metrics run the current reward code on the assistant output string only; they are useful for format/protocol/workflow symptoms, but conservative for task progress because observations are absent.",
            "The new reward has two extra gates that cannot be reconstructed exactly here: budget_ids_viewed and recommended_ids_budgeted.",
        ],
        "inputs": {
            "rollout_files": [str(path.relative_to(ROOT)) for path in rollout_files],
            "probe_parquets": [str(path.relative_to(ROOT)) for path in probe_parquets],
        },
        "overall": summarize_group(rows),
        "by_run": {run: summarize_group(group_rows) for run, group_rows in sorted(rows_by_run.items())},
    }

    output_json = ROOT / args.output_json
    output_csv = ROOT / args.output_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_csv, rows)
    compact_overall = {key: value for key, value in report["overall"].items() if key != "examples"}
    print(json.dumps({"limitations": report["limitations"], "overall": compact_overall}, ensure_ascii=False, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
