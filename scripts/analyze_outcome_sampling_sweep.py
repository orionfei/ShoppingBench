#!/usr/bin/env python3
"""Analyze outcome-only GRPO sampling sweeps from rollout JSONL/report JSON.

The analyzer deliberately never treats a dense ``score`` as an outcome.  It uses
explicit ``paper_asr``/``terminal_asr`` fields when present and can reconstruct
the historical equivalent from ``rule``, ``budget`` and a successful terminate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAILURE_KINDS = ("server_error", "json_decode", "truncation", "runaway")


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def first_number(mapping: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = as_float(mapping.get(name))
        if value is not None:
            return value
    return None


def binary(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number >= 1.0 - 1e-9)


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            value = dict(value)
            value["_source_file"] = str(path)
            rows.append(value)
    return rows


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(raw: str, origin: Path) -> Path | None:
    path = Path(raw)
    candidates = [path, origin.parent / path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def extract_rows(value: Any) -> list[dict[str, Any]]:
    """Find embedded trajectory rows without mistaking aggregate rows for them."""
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        keys = set().union(*(item.keys() for item in value[:3]))
        if keys & {"paper_asr", "terminal_asr", "output", "terminate_success"}:
            return [dict(item) for item in value]
    if isinstance(value, dict):
        for key in ("trajectories", "samples", "rollouts", "rows", "records"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                found = extract_rows(candidate)
                if found:
                    return found
    return []


def report_raw_files(value: Any, origin: Path) -> list[Path]:
    raw_paths: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("files"), list):
            raw_paths.extend(str(item) for item in value["files"])
        if isinstance(value.get("path"), str):
            raw_paths.append(value["path"])
        reports = value.get("reports")
        if isinstance(reports, list):
            for report in reports:
                if isinstance(report, dict):
                    if isinstance(report.get("files"), list):
                        raw_paths.extend(str(item) for item in report["files"])
                    if isinstance(report.get("path"), str):
                        raw_paths.append(report["path"])
    files: list[Path] = []
    for raw in raw_paths:
        resolved = resolve_path(raw, origin)
        if resolved is None:
            continue
        if resolved.is_dir():
            files.extend(sorted(resolved.rglob("*.jsonl")))
        elif resolved.suffix == ".jsonl":
            files.append(resolved)
    return list(dict.fromkeys(files))


def find_meta(start: Path) -> dict[str, Any]:
    parents = [start if start.is_dir() else start.parent, *(start.parents[:3])]
    for parent in parents:
        for filename in ("manifest.json", "meta.json"):
            path = parent / filename
            if path.exists():
                value = load_json(path)
                return value if isinstance(value, dict) else {}
    return {}


def parse_name_metadata(name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    patterns = {
        "temperature": (r"(?:^|[_-])t(?:emp)?(\d+)(?:[_-]|$)", 10.0),
        "top_p": (r"(?:^|[_-])top[_-]?p(\d+)(?:[_-]|$)", 10.0),
        "seed": (r"(?:^|[_-])seed(\d+)(?:[_-]|$)", 1.0),
        "group_size": (r"(?:^|[_-])g(\d+)(?:[_-]|$)", 1.0),
    }
    for key, (pattern, scale) in patterns.items():
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            digits = match.group(1)
            if key in {"temperature", "top_p"}:
                result[key] = int(digits) / (10 ** (len(digits) - 1)) if digits.startswith("0") else int(digits) / scale
            else:
                result[key] = int(digits)
    return result


def canonical_metadata(meta: dict[str, Any], source: Path, overrides: argparse.Namespace) -> dict[str, Any]:
    inferred = parse_name_metadata(source.parent.name + "_" + source.stem)
    aliases = {
        "temperature": ("temperature", "train_temperature", "rollout_temperature"),
        "top_p": ("top_p", "train_top_p", "rollout_top_p"),
        "seed": ("seed", "rollout_seed", "sampling_seed"),
        "group_size": ("group_size", "rollout_n", "n", "G"),
    }
    result: dict[str, Any] = {"run_id": str(meta.get("run_id") or source.parent.name or source.stem)}
    for key, names in aliases.items():
        explicit = getattr(overrides, key, None)
        value = explicit
        if value is None:
            value = next((meta[name] for name in names if meta.get(name) is not None), inferred.get(key))
        result[key] = value
    for key in (
        "checkpoint", "checkpoint_path", "dataset", "query_val_files", "max_response_length",
        "max_assistant_turns", "max_user_turns", "rollout_max_num_seqs", "elapsed_seconds",
        "training_step", "candidate", "kl", "entropy", "clip_fraction",
    ):
        if meta.get(key) is not None:
            result[key] = meta[key]
    if result.get("training_step") is None and source.suffix == ".jsonl" and source.stem.isdigit():
        result["training_step"] = int(source.stem)
    return result


def load_run(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    meta = find_meta(path)
    rows: list[dict[str, Any]] = []
    if path.is_dir():
        files = sorted(path.rglob("*.jsonl"))
        if not files:
            reports = sorted(path.glob("*.json"))
            for report_path in reports:
                value = load_json(report_path)
                rows = extract_rows(value)
                files = report_raw_files(value, report_path)
                if rows or files:
                    path = report_path
                    break
        for file in files:
            rows.extend(read_jsonl(file))
    elif path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        value = load_json(path)
        if isinstance(value, dict):
            embedded_meta = value.get("meta") or value.get("manifest") or {}
            if isinstance(embedded_meta, dict):
                meta = {**meta, **embedded_meta}
            for key in ("temperature", "top_p", "rollout_n", "seed", "checkpoint", "run_id"):
                if value.get(key) is not None:
                    meta[key] = value[key]
        rows = extract_rows(value)
        if not rows:
            files = report_raw_files(value, path)
            for file in files:
                rows.extend(read_jsonl(file))
        if not rows:
            warnings.append("No trajectory rows found; aggregate-only report cannot reconstruct mixed groups")
    metadata = canonical_metadata(meta, path, args)
    metadata["source"] = str(path)
    return rows, metadata, warnings


def query_key(row: dict[str, Any]) -> str:
    for key in ("query_id", "prompt_id", "uid", "query", "input"):
        value = row.get(key)
        if value not in (None, ""):
            text = str(value)
            return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return "missing-query"


def outcome(row: dict[str, Any]) -> tuple[int | None, int | None, str, str]:
    paper = binary(row.get("paper_asr"))
    paper_source = "paper_asr"
    if paper is None:
        rule, budget = binary(row.get("rule")), binary(row.get("budget"))
        if rule is not None and budget is not None:
            paper, paper_source = rule * budget, "rule*budget"
        else:
            paper_source = "missing"

    terminal = binary(row.get("terminal_asr"))
    terminal_source = "terminal_asr"
    if terminal is None and paper is not None:
        terminated = binary(row.get("terminate_success"))
        if terminated is None:
            terminated = binary(row.get("terminate"))
            terminal_source = "paper_asr*terminate"
        else:
            terminal_source = "paper_asr*terminate_success"
        if terminated is not None:
            terminal = paper * terminated
        else:
            terminal_source = "missing"
    return paper, terminal, paper_source, terminal_source


def repeated_sentence_runaway(output: Any) -> bool:
    """Detect obvious decoder loops without treating repeated product tables as failures."""
    if not isinstance(output, str) or len(output) < 160:
        return False
    sentences = [
        re.sub(r"\s+", " ", sentence).strip().lower()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", output[-5000:])
    ]
    previous, repeats = None, 0
    for sentence in sentences:
        if len(sentence) < 30:
            previous, repeats = None, 0
            continue
        if sentence == previous:
            repeats += 1
            if repeats >= 4:
                return True
        else:
            previous, repeats = sentence, 1
    return False


def malformed_tool_call_in_output(output: Any) -> bool:
    if not isinstance(output, str):
        return False
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, flags=re.DOTALL):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return True
        values = parsed if isinstance(parsed, list) else [parsed]
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                return True
            parameters = value.get("parameters")
            if not isinstance(parameters, dict):
                return True
    return False


def failures(row: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("structured_failure_mode", "failure_mode", "error", "finish_reason")
    )
    explicit = lambda *names: any(bool(row.get(name)) for name in names)
    output_text = str(row.get("output") or "").lower()
    return {
        "server_error": explicit("server_error", "engine_error") or any(
            word in text or word in output_text
            for word in ("internal server error", "server_error", "engine_error", "connection error")
        ),
        "json_decode": explicit("json_error", "malformed_tool_call")
        or any(word in text for word in ("json", "decode", "malformed"))
        or malformed_tool_call_in_output(row.get("output")),
        "truncation": explicit("truncated", "token_limit", "length_truncated") or any(word in text for word in ("truncate", "token_limit", "max_length", "length")),
        "runaway": explicit("runaway") or "runaway" in text or repeated_sentence_runaway(row.get("output")),
    }


def token_count(row: dict[str, Any]) -> float | None:
    return first_number(row, ("response_tokens", "completion_tokens", "output_tokens", "num_response_tokens", "response_length"))


def bootstrap_ci(values: list[int], draws: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_run(rows: list[dict[str, Any]], metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    group_size = int(metadata.get("group_size") or args.group_size)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[query_key(row)].append(row)
    groups: list[dict[str, Any]] = []
    paper_sources, terminal_sources = Counter(), Counter()
    all_failures = Counter()
    token_values: list[float] = []
    step_values: list[float] = []

    for key, items in sorted(by_query.items()):
        papers, terminals = [], []
        for row in items:
            paper, terminal, p_source, t_source = outcome(row)
            paper_sources[p_source] += 1
            terminal_sources[t_source] += 1
            papers.append(paper)
            terminals.append(terminal)
            flags = failures(row)
            for kind, present in flags.items():
                all_failures[kind] += int(present)
            value = token_count(row)
            if value is not None:
                token_values.append(value)
            step = first_number(row, ("steps", "assistant_message_count", "num_steps"))
            if step is not None:
                step_values.append(step)
        valid_papers = [x for x in papers if x is not None]
        valid_terminals = [x for x in terminals if x is not None]

        def state(values: list[int]) -> str | None:
            if not values:
                return None
            return "all_fail" if max(values) == 0 else "all_success" if min(values) == 1 else "mixed"

        groups.append({
            "query_key": key,
            "samples": len(items),
            "complete": len(items) == group_size,
            "paper_known": len(valid_papers),
            "terminal_known": len(valid_terminals),
            "paper_successes": sum(valid_papers),
            "terminal_successes": sum(valid_terminals),
            "paper_state": state(valid_papers) if len(valid_papers) == len(items) else None,
            "terminal_state": state(valid_terminals) if len(valid_terminals) == len(items) else None,
        })

    paper_states = Counter(group["paper_state"] for group in groups if group["paper_state"])
    terminal_states = Counter(group["terminal_state"] for group in groups if group["terminal_state"])
    paper_known = sum(group["paper_known"] for group in groups)
    terminal_known = sum(group["terminal_known"] for group in groups)
    complete = sum(group["complete"] for group in groups)
    terminal_mixed = [int(group["terminal_state"] == "mixed") for group in groups if group["terminal_state"]]
    paper_mixed = [int(group["paper_state"] == "mixed") for group in groups if group["paper_state"]]
    rng = np.random.default_rng(args.bootstrap_seed + int(metadata.get("seed") or 0))
    terminal_ci = bootstrap_ci(terminal_mixed, args.bootstrap, rng)
    paper_ci = bootstrap_ci(paper_mixed, args.bootstrap, rng)
    n = len(rows)
    combined_failures = sum(any(failures(row).values()) for row in rows)
    infrastructure_failures = sum(
        failures(row)["server_error"] or failures(row)["json_decode"] or failures(row)["runaway"]
        for row in rows
    )
    format_mean = mean(first_number(row, ("format", "format_score", "format_valid")) for row in rows)
    workflow_mean = mean(first_number(row, ("workflow_valid", "workflow")) for row in rows)
    elapsed = as_float(metadata.get("elapsed_seconds"))
    reward_batch_seconds = max(
        (value for row in rows if (value := first_number(row, ("reward_batch_wall_seconds",))) is not None),
        default=None,
    )

    result: dict[str, Any] = {
        **metadata,
        "group_size": group_size,
        "rows": n,
        "queries": len(groups),
        "complete_groups": complete,
        "rollout_complete_rate": complete / len(groups) if groups else None,
        "paper_asr": sum(group["paper_successes"] for group in groups) / paper_known if paper_known else None,
        "terminal_asr": sum(group["terminal_successes"] for group in groups) / terminal_known if terminal_known else None,
        "paper_asr_known_rate": paper_known / n if n else None,
        "terminal_asr_known_rate": terminal_known / n if n else None,
        "paper_group_counts": dict(paper_states),
        "terminal_group_counts": dict(terminal_states),
        "mixed_paper_asr_group_rate": mean(paper_mixed),
        "mixed_terminal_asr_group_rate": mean(terminal_mixed),
        "mixed_paper_asr_ci95": list(paper_ci),
        "mixed_terminal_asr_ci95": list(terminal_ci),
        "paper_pass_at_g": mean(int(group["paper_successes"] > 0) for group in groups if group["paper_state"]),
        "terminal_pass_at_g": mean(int(group["terminal_successes"] > 0) for group in groups if group["terminal_state"]),
        "format_mean": format_mean,
        "workflow_valid_mean": workflow_mean,
        "failure_counts": {kind: int(all_failures[kind]) for kind in FAILURE_KINDS},
        "failure_rate": combined_failures / n if n else None,
        "infrastructure_failure_rate": infrastructure_failures / n if n else None,
        "token_limit_noncompletion_rate": all_failures["truncation"] / n if n else None,
        "response_tokens_mean": mean(token_values),
        "response_tokens_p50": percentile(token_values, 50),
        "response_tokens_p95": percentile(token_values, 95),
        "response_tokens_max": max(token_values) if token_values else None,
        "steps_mean": mean(step_values),
        "steps_p50": percentile(step_values, 50),
        "steps_p95": percentile(step_values, 95),
        "steps_max": max(step_values) if step_values else None,
        "elapsed_seconds": elapsed,
        "reward_batch_wall_seconds": reward_batch_seconds,
        "reward_wall_fraction": reward_batch_seconds / elapsed if reward_batch_seconds is not None and elapsed else None,
        "title_embedding_cache_size": max(
            (value for row in rows if (value := first_number(row, ("title_embedding_cache_size",))) is not None),
            default=None,
        ),
        "title_embeddings_added": max(
            (value for row in rows if (value := first_number(row, ("title_embeddings_added",))) is not None),
            default=None,
        ),
        "trajectories_per_second": n / elapsed if elapsed and elapsed > 0 else None,
        "outcome_field_sources": {"paper_asr": dict(paper_sources), "terminal_asr": dict(terminal_sources)},
        "groups": groups,
    }
    return result


def apply_gates_and_rank(runs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline_workflow = args.baseline_workflow
    if baseline_workflow is None:
        baselines = [
            run["workflow_valid_mean"] for run in runs
            if run.get("temperature") == 0.2 and run.get("top_p") == 0.9 and run.get("workflow_valid_mean") is not None
        ]
        baseline_workflow = mean(baselines)
    for run in runs:
        reasons: list[str] = []
        if run["rows"] == 0:
            reasons.append("no_trajectory_rows")
        if run.get("terminal_asr_known_rate") != 1.0:
            reasons.append("terminal_asr_incomplete")
        if run.get("format_mean") is None:
            reasons.append("format_missing")
        elif run["format_mean"] < args.format_min:
            reasons.append("format_below_threshold")
        if run.get("rollout_complete_rate") != 1.0:
            reasons.append("incomplete_groups")
        gate_failure_key = "infrastructure_failure_rate" if args.allow_truncation_as_outcome else "failure_rate"
        if run.get(gate_failure_key) is not None and run[gate_failure_key] > args.max_failure_rate:
            reasons.append(f"{gate_failure_key}_above_threshold")
        if run.get("workflow_valid_mean") is None:
            reasons.append("workflow_missing")
        elif baseline_workflow is not None and run["workflow_valid_mean"] < baseline_workflow - args.workflow_drop_max:
            reasons.append("workflow_below_baseline_tolerance")
        run["gate_eligible"] = not reasons
        run["gate_reasons"] = reasons
        run["baseline_workflow_valid_mean"] = baseline_workflow

    eligible = [run for run in runs if run["gate_eligible"]]
    ranked = sorted(
        eligible,
        key=lambda run: (
            run.get("mixed_terminal_asr_ci95", [None])[0] if run.get("mixed_terminal_asr_ci95", [None])[0] is not None else -1,
            run.get("mixed_paper_asr_group_rate") if run.get("mixed_paper_asr_group_rate") is not None else -1,
            run.get("terminal_pass_at_g") if run.get("terminal_pass_at_g") is not None else -1,
            run.get("terminal_asr") if run.get("terminal_asr") is not None else -1,
            -(run.get("response_tokens_mean") if run.get("response_tokens_mean") is not None else float("inf")),
            -(run.get("elapsed_seconds") if run.get("elapsed_seconds") is not None else float("inf")),
        ),
        reverse=True,
    )
    for rank, run in enumerate(ranked, 1):
        run["rank"] = rank
    return ranked


def aggregate_and_rank_configs(runs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Pool seed×query observations for a multi-seed configuration comparison."""
    by_config: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        temperature, top_p = as_float(run.get("temperature")), as_float(run.get("top_p"))
        if temperature is not None and top_p is not None:
            by_config[(temperature, top_p)].append(run)
    configs: list[dict[str, Any]] = []
    for (temperature, top_p), members in sorted(by_config.items()):
        eligible = [run for run in members if run.get("gate_eligible")]
        terminal_mixed = [
            int(group.get("terminal_state") == "mixed")
            for run in eligible for group in run.get("groups", []) if group.get("terminal_state")
        ]
        paper_mixed = [
            int(group.get("paper_state") == "mixed")
            for run in eligible for group in run.get("groups", []) if group.get("paper_state")
        ]
        seed_key = int(abs(temperature * 1000 + top_p * 10000))
        rng = np.random.default_rng(args.bootstrap_seed + seed_key)
        ci = bootstrap_ci(terminal_mixed, args.bootstrap, rng)
        terminal_known = sum(
            group.get("terminal_known", 0) for run in eligible for group in run.get("groups", [])
        )
        terminal_success = sum(
            group.get("terminal_successes", 0) for run in eligible for group in run.get("groups", [])
        )
        configs.append({
            "temperature": temperature,
            "top_p": top_p,
            "run_count": len(members),
            "eligible_run_count": len(eligible),
            "all_runs_eligible": len(eligible) == len(members),
            "seed_count": len({run.get("seed") for run in eligible}),
            "seeds": sorted({run.get("seed") for run in eligible}, key=lambda value: (value is None, str(value))),
            "query_group_observations": len(terminal_mixed),
            "mixed_terminal_asr_group_rate": mean(terminal_mixed),
            "mixed_terminal_asr_ci95": list(ci),
            "mixed_paper_asr_group_rate": mean(paper_mixed),
            "terminal_pass_at_g": mean(run.get("terminal_pass_at_g") for run in eligible),
            "terminal_asr": terminal_success / terminal_known if terminal_known else None,
            "response_tokens_mean": mean(run.get("response_tokens_mean") for run in eligible),
            "elapsed_seconds_mean": mean(run.get("elapsed_seconds") for run in eligible),
        })
    eligible_configs = [config for config in configs if config["all_runs_eligible"] and config["eligible_run_count"] > 0]
    eligible_configs.sort(
        key=lambda config: (
            config["mixed_terminal_asr_ci95"][0] if config["mixed_terminal_asr_ci95"][0] is not None else -1,
            config["mixed_paper_asr_group_rate"] if config["mixed_paper_asr_group_rate"] is not None else -1,
            config["terminal_pass_at_g"] if config["terminal_pass_at_g"] is not None else -1,
            config["terminal_asr"] if config["terminal_asr"] is not None else -1,
            -(config["response_tokens_mean"] if config["response_tokens_mean"] is not None else float("inf")),
            -(config["elapsed_seconds_mean"] if config["elapsed_seconds_mean"] is not None else float("inf")),
        ),
        reverse=True,
    )
    for rank, config in enumerate(eligible_configs, 1):
        config["rank"] = rank
    return eligible_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Rollout dirs, JSONL files, or summary/report JSON files")
    parser.add_argument("--output", default="reports/outcome_sampling_sweep/analysis.json")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=None, help="Override metadata for all inputs")
    parser.add_argument("--top-p", type=float, default=None, help="Override metadata for all inputs")
    parser.add_argument("--seed", type=int, default=None, help="Override metadata for all inputs")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--format-min", type=float, default=0.98)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument(
        "--allow-truncation-as-outcome", action="store_true",
        help="Gate server/JSON/runaway only; report token-limit noncompletion separately as binary outcome failure",
    )
    parser.add_argument("--workflow-drop-max", type=float, default=0.05)
    parser.add_argument("--baseline-workflow", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs, warnings = [], []
    for raw in args.inputs:
        path = Path(raw).resolve()
        if not path.exists():
            warnings.append(f"Missing input: {path}")
            continue
        rows, metadata, run_warnings = load_run(path, args)
        run = summarize_run(rows, metadata, args)
        run["warnings"] = run_warnings
        runs.append(run)
    ranked = apply_gates_and_rank(runs, args)
    config_ranking = aggregate_and_rank_configs(runs, args)
    report = {
        "schema_version": 1,
        "selection_rule": (
            ("hard infrastructure gates with token-limit noncompletion treated as outcome=0, then "
             if args.allow_truncation_as_outcome else "hard combined-failure gates, then ")
            + "mixed_terminal_asr CI95 lower bound, mixed_paper_asr_group_rate, "
            "terminal_pass_at_g, terminal_asr, lower token/time cost"
        ),
        "config": {
            "group_size_default": args.group_size,
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "format_min": args.format_min,
            "max_failure_rate": args.max_failure_rate,
            "allow_truncation_as_outcome": args.allow_truncation_as_outcome,
            "workflow_drop_max": args.workflow_drop_max,
        },
        "warnings": warnings,
        "runs": runs,
        "ranking": [
            {
                "rank": run["rank"], "run_id": run["run_id"], "temperature": run.get("temperature"),
                "top_p": run.get("top_p"), "seed": run.get("seed"),
                "mixed_terminal_asr_ci95_lower": run["mixed_terminal_asr_ci95"][0],
                "mixed_terminal_asr_group_rate": run["mixed_terminal_asr_group_rate"],
            }
            for run in ranked
        ],
        "config_ranking": config_ranking,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(runs)} runs, {len(ranked)} gate-eligible")
    for item in report["ranking"]:
        print(f"#{item['rank']} {item['run_id']} t={item['temperature']} p={item['top_p']} mixed={item['mixed_terminal_asr_group_rate']}")


if __name__ == "__main__":
    main()
