#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.modules.setdefault("ujson", json)

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))
RL_SRC = ROOT / "src" / "rl"
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from util.message import Message  # noqa: E402
from verl.utils.reward_score import shoppingbench_query  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe SFT checkpoints for GRPO readiness by scoring grouped ShoppingBench rollouts. "
            "Use it to separate the protocol peak from the task-strategy peak."
        )
    )
    parser.add_argument(
        "--synthesize-file",
        default="data/synthesize_voucher_train.jsonl",
        help="JSONL with query, reward, and voucher fields.",
    )
    parser.add_argument(
        "--rollout-files",
        nargs="*",
        default=[],
        help="One or more rollout JSONL files. Each line must be one full trajectory.",
    )
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--probe-query-output",
        default="data/probe/sft_probe_voucher_16.jsonl",
        help="Where to write the fixed probe query set.",
    )
    parser.add_argument(
        "--report-json",
        default="data/probe/sft_grpo_probe_report.json",
        help="Where to write probe metrics.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def deterministic_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    selected = sorted(indexed[: min(sample_size, len(indexed))], key=lambda item: item[0])
    return [row for _, row in selected]


def synthesize_by_query(rows: list[dict]) -> dict[str, dict]:
    return {row["query"]: row for row in rows if row.get("query")}


def message_from_step(step: dict) -> Message:
    message = step.get("completion", {}).get("message")
    if isinstance(message, dict) and message:
        return Message.from_dict(message)
    completion = step.get("completion", {})
    return Message.from_string(
        completion.get("reasoning_content") or "",
        completion.get("content") or "",
    )


def query_from_row(row: list[dict]) -> str:
    if not row:
        return ""
    return row[0].get("extra_info", {}).get("query", "")


def reward_messages(row: list[dict]) -> list[dict]:
    messages = [{"role": "user", "content": query_from_row(row)}]
    for step in row:
        message = message_from_step(step)
        content = message.to_string(["think", "tool_call", "response"])
        messages.append({"role": "assistant", "content": content, "tool_calls": message.tool_call})
        for obs in message.obs or []:
            result = obs.get("results") if isinstance(obs, dict) else obs
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    return messages


def reward_solution(row: list[dict]) -> str:
    return "\n".join(
        message_from_step(step).to_string(["think", "tool_call", "response"])
        for step in row
    )


def score_row(row: list[dict], meta: dict) -> dict:
    ground_truth = json.dumps(
        {"reward": meta.get("reward") or [], "voucher": meta.get("voucher") or {}},
        ensure_ascii=False,
    )
    result = shoppingbench_query.compute_score(
        reward_solution(row),
        ground_truth,
        extra_info={
            "messages": {"messages": reward_messages(row)},
            "num_turns": len(row),
        },
    )
    protocol = result["protocol"]
    task = result["task"]
    total = protocol + task
    progress_details = {
        "components": {
            key: result[key]
            for key in (
                "search_gold_recall",
                "select_gold_overlap",
                "same_shop",
                "verify_selected_gold",
                "budget_recomputed_correct",
                "within_budget_correct",
                "recommend_gold_overlap",
                "terminate_after_valid_recommend",
            )
        },
        "recommended_ids": result["recommended_ids"].split(",") if result.get("recommended_ids") else [],
        "within_budget": bool(result["within_budget_correct"]),
        "budget_trusted": bool(result["budget_recomputed_correct"]),
        "shop_ok": bool(result["same_shop"]),
    }
    return {
        "format": result["format"],
        "tool_valid": result["tool_valid"],
        "protocol": protocol,
        "progress": result["progress"],
        "outcome": result["outcome"],
        "task": task,
        "total": total,
        "steps": result["steps"],
        "progress_details": progress_details,
    }


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_groups(scored_by_query: dict[str, list[dict]], group_size: int) -> dict:
    component_names = ["format", "tool_valid", "protocol", "progress", "outcome", "task", "total", "steps"]
    query_summaries = []
    for query, scores in scored_by_query.items():
        group = scores[:group_size]
        if not group:
            continue
        summary = {
            "query": query,
            "samples": len(group),
            "complete_group": len(group) == group_size,
        }
        for name in component_names:
            values = [float(item[name]) for item in group]
            summary[f"{name}_mean"] = mean(values)
            summary[f"{name}_var"] = variance(values)
        query_summaries.append(summary)

    aggregate = {
        "queries": len(query_summaries),
        "complete_groups": sum(1 for item in query_summaries if item["complete_group"]),
    }
    for name in component_names:
        means = [item[f"{name}_mean"] for item in query_summaries]
        variances = [item[f"{name}_var"] for item in query_summaries]
        aggregate[f"{name}_mean"] = mean(means)
        aggregate[f"{name}_group_var_mean"] = mean(variances)
    aggregate["protocol_peak_signal"] = aggregate.get("protocol_group_var_mean", 0.0)
    aggregate["task_peak_signal"] = aggregate.get("task_group_var_mean", 0.0)
    return {"aggregate": aggregate, "queries": query_summaries}


def score_rollout_file(path: Path, meta_by_query: dict[str, dict], probe_queries: set[str], group_size: int) -> dict:
    rows = read_jsonl(path)
    scored_by_query = defaultdict(list)
    detailed_rows = []
    for row in rows:
        if not isinstance(row, list):
            continue
        query = query_from_row(row)
        if probe_queries and query not in probe_queries:
            continue
        meta = meta_by_query.get(query)
        if not meta:
            continue
        score = score_row(row, meta)
        score["query"] = query
        detailed_rows.append(score)
        scored_by_query[query].append(score)
    summary = summarize_groups(scored_by_query, group_size)
    summary["rollout_file"] = str(path)
    summary["scored_trajectories"] = len(detailed_rows)
    return summary


def choose_second_peak_candidate(reports: list[dict]) -> dict | None:
    candidates = []
    for report in reports:
        agg = report.get("aggregate", {})
        protocol_mean = agg.get("protocol_mean", 0.0)
        protocol_var = agg.get("protocol_group_var_mean", math.inf)
        task_var = agg.get("task_group_var_mean", 0.0)
        if protocol_mean >= 0.90 and protocol_var <= 0.05:
            candidates.append((task_var, report))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def main() -> int:
    args = parse_args()
    synthesize_path = ROOT / args.synthesize_file
    synthesize_rows = read_jsonl(synthesize_path)
    probe_rows = deterministic_sample(synthesize_rows, args.sample_size, args.seed)
    write_jsonl(ROOT / args.probe_query_output, probe_rows)

    meta_by_query = synthesize_by_query(synthesize_rows)
    probe_queries = {row["query"] for row in probe_rows}
    rollout_reports = []
    for rollout_file in args.rollout_files:
        rollout_reports.append(
            score_rollout_file(ROOT / rollout_file, meta_by_query, probe_queries, args.group_size)
        )

    selected = choose_second_peak_candidate(rollout_reports)
    report = {
        "settings": {
            "synthesize_file": str(synthesize_path),
            "sample_size": args.sample_size,
            "group_size": args.group_size,
            "seed": args.seed,
            "probe_query_output": str(ROOT / args.probe_query_output),
            "reward": {
                "protocol": "0.5 * format + 0.5 * tool_valid",
                "task": "progress + 2.0 * outcome - 0.02 * steps",
                "total": "protocol + task",
            },
            "second_peak_candidate_rule": "protocol_mean >= 0.90 and protocol_group_var_mean <= 0.05, max task_group_var_mean",
        },
        "rollouts": rollout_reports,
        "selected_second_peak_candidate": selected.get("rollout_file") if selected else None,
    }
    write_json(ROOT / args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
