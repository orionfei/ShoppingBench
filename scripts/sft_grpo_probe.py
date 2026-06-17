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

from rewards.prm import format_reward  # noqa: E402
from util.history_compression import build_state_from_history  # noqa: E402
from util.message import Message, OUTPUT_ROLES  # noqa: E402


LEGAL_TOOLS = {
    "find_product",
    "view_product_information",
    "recommend_product",
    "python_execute",
    "web_search",
    "terminate",
}


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


def assistant_history(row: list[dict]) -> list[str]:
    query = query_from_row(row)
    history = [Message(user=query).to_string(["user"])]
    for step in row:
        history.append(message_from_step(step).to_string(["think", "tool_call", "obs", "response"]))
    return history


def final_recommended_ids(row: list[dict]) -> list[str]:
    product_ids = []
    for step in row:
        message = message_from_step(step)
        for call in message.tool_call or []:
            if call.get("name") == "recommend_product":
                raw = call.get("parameters", {}).get("product_ids", "")
                if isinstance(raw, str):
                    product_ids = [part.strip() for part in raw.split(",") if part.strip()]
    return product_ids


def terminated_success(row: list[dict]) -> bool:
    for step in reversed(row):
        message = message_from_step(step)
        for call in message.tool_call or []:
            if call.get("name") == "terminate":
                return call.get("parameters", {}).get("status") == "success"
    return False


def average_format_score(row: list[dict]) -> float:
    if not row:
        return 0.0
    scores = []
    for step in row:
        message = message_from_step(step)
        scores.append(format_reward(message.to_string(OUTPUT_ROLES)))
    return sum(scores) / len(scores)


def tool_call_validity(row: list[dict]) -> float:
    checks = []
    for step in row:
        message = message_from_step(step)
        calls = message.tool_call or []
        if not calls and not message.response:
            checks.append(0.0)
            continue
        obs_by_id = {
            item.get("tool_call_id"): item
            for item in (message.obs or [])
            if isinstance(item, dict)
        }
        for call in calls:
            name = call.get("name")
            params = call.get("parameters")
            ok = name in LEGAL_TOOLS and isinstance(params, dict)
            observation = obs_by_id.get(call.get("tool_call_id"))
            if name in {"find_product", "view_product_information"}:
                ok = ok and isinstance((observation or {}).get("results"), list)
            elif name == "python_execute":
                result = (observation or {}).get("results")
                ok = ok and isinstance(result, dict) and result.get("success") is not False
            checks.append(1.0 if ok else 0.0)
    return sum(checks) / len(checks) if checks else 0.0


def coverage_ratio(values: list, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(len(values), total) / total


def state_progress_score(row: list[dict], meta: dict) -> tuple[float, dict]:
    reward_items = meta.get("reward") or []
    voucher = meta.get("voucher") or {}
    requested = len(reward_items)
    try:
        state = build_state_from_history(assistant_history(row))
    except Exception:
        state = {}

    selected_ids = [str(pid) for pid in state.get("selected_product_ids") or []]
    viewed_ids = {
        str(item.get("product_id"))
        for item in state.get("viewed_products") or []
        if item.get("product_id")
    }
    recommended_ids = final_recommended_ids(row)
    selected_shops = {
        str(item.get("shop_id"))
        for item in state.get("budget_candidates") or []
        if str(item.get("product_id")) in selected_ids and item.get("shop_id") is not None
    }
    shop_ok = voucher.get("voucher_type") != "shop" or len(selected_shops) == 1
    selected_ratio = coverage_ratio(selected_ids, requested)
    viewed_ratio = coverage_ratio([pid for pid in selected_ids if pid in viewed_ids], requested)
    recommend_count_ok = requested > 0 and len(recommended_ids) == requested
    within_budget = state.get("within_budget_if_now") is True
    budget_trusted = state.get("budget_calculation_trusted") is True
    has_search = bool(state.get("searches"))
    terminated = terminated_success(row)

    components = {
        "voucher_parse": 1.0 if state.get("voucher", {}).get("parse_ok") is True else 0.0,
        "search": 1.0 if has_search else 0.0,
        "select": selected_ratio,
        "shop": 1.0 if shop_ok else 0.0,
        "verify": viewed_ratio,
        "budget_trusted": 1.0 if budget_trusted else 0.0,
        "within_budget": 1.0 if within_budget else 0.0,
        "recommend": 1.0 if recommend_count_ok else 0.0,
        "terminate": 1.0 if terminated else 0.0,
    }
    weights = {
        "voucher_parse": 0.10,
        "search": 0.10,
        "select": 0.20,
        "shop": 0.10,
        "verify": 0.15,
        "budget_trusted": 0.15,
        "within_budget": 0.10,
        "recommend": 0.05,
        "terminate": 0.05,
    }
    progress = sum(components[key] * weights[key] for key in weights)
    details = {
        "components": components,
        "selected_product_ids": selected_ids,
        "recommended_ids": recommended_ids,
        "within_budget": within_budget,
        "budget_trusted": budget_trusted,
        "shop_ok": shop_ok,
    }
    return progress, details


def exact_outcome_score(row: list[dict], meta: dict) -> float:
    reward_ids = [str(item.get("product_id")) for item in meta.get("reward") or []]
    recommended_ids = final_recommended_ids(row)
    if not reward_ids or recommended_ids != reward_ids:
        return 0.0
    progress, details = state_progress_score(row, meta)
    return 1.0 if details["within_budget"] and details["shop_ok"] and terminated_success(row) else 0.0


def score_row(row: list[dict], meta: dict) -> dict:
    fmt = average_format_score(row)
    tool = tool_call_validity(row)
    progress, progress_details = state_progress_score(row, meta)
    outcome = exact_outcome_score(row, meta)
    steps = len(row)
    protocol = 0.5 * fmt + 0.5 * tool
    task = progress + 2.0 * outcome - 0.02 * steps
    total = protocol + task
    return {
        "format": fmt,
        "tool_valid": tool,
        "protocol": protocol,
        "progress": progress,
        "outcome": outcome,
        "task": task,
        "total": total,
        "steps": steps,
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
