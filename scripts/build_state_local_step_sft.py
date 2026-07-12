#!/usr/bin/env python3
import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
STATE_RE = re.compile(r"<state>\s*(.*?)\s*</state>", re.DOTALL)


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fout.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def completion_content(step: dict[str, Any]) -> str:
    completion = step.get("completion") or {}
    return str(completion.get("content") or "")


def parse_state_from_step(step: dict[str, Any]) -> dict[str, Any]:
    prompt = step.get("prompt") or []
    user_prompt = (prompt[1] if len(prompt) > 1 else {}).get("content") or ""
    match = STATE_RE.search(str(user_prompt))
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def message(step: dict[str, Any]) -> dict[str, Any]:
    return ((step.get("completion") or {}).get("message") or {})


def tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message(step).get("tool_call") or []
    return calls if isinstance(calls, list) else []


def observations(step: dict[str, Any]) -> list[dict[str, Any]]:
    obs = message(step).get("obs") or []
    return [item for item in obs if isinstance(item, dict)]


def obs_by_id(step: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("tool_call_id")): item
        for item in observations(step)
        if item.get("tool_call_id") is not None
    }


def is_error_result(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("error")) or result.get("_tool_success") is False or result.get("_parse_ok") is False
    if isinstance(result, str):
        lowered = result.lower()
        return "error" in lowered or "invalid" in lowered
    return False


def state_has_failure_feedback(state: dict[str, Any]) -> bool:
    return bool(
        state.get("last_errors")
        or state.get("selection_issues")
        or state.get("failed_searches")
        or state.get("failed_retry_searches")
    )


def state_has_recovery_context(state: dict[str, Any]) -> bool:
    return state_has_failure_feedback(state) or bool(state.get("previous_decision"))


def tool_sequence(step: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(call.get("name") or "") for call in tool_calls(step))


def current_step_negative_reasons(
    trajectory: list[dict[str, Any]],
    step_index: int,
    *,
    mask_decision_replacement_searches: bool,
) -> list[str]:
    step = trajectory[step_index]
    calls = tool_calls(step)
    call_obs = obs_by_id(step)
    reasons: list[str] = []

    if not calls:
        reasons.append("no_tool_call")

    for call in calls:
        name = call.get("name")
        obs = call_obs.get(str(call.get("tool_call_id")))
        result = None if obs is None else obs.get("results")
        if obs is None:
            reasons.append(f"missing_observation::{name}")
            continue
        if is_error_result(result):
            reasons.append(f"tool_error::{name}")
        if name == "find_product" and isinstance(result, list) and not result:
            reasons.append("empty_find_product_result")
        if name == "budget_check" and isinstance(result, dict):
            if result.get("within_budget") is False:
                reasons.append("budget_check_not_within_budget")
            if result.get("voucher_used") and result.get("voucher_applied") is False:
                reasons.append("budget_check_voucher_not_applied")

    state_name = (step.get("extra_info") or {}).get("harness_state")
    sequence = tool_sequence(step)
    if mask_decision_replacement_searches and state_name == "DECISION" and "find_product" in sequence:
        reasons.append("decision_replacement_search")

    if step_index + 1 < len(trajectory):
        next_state = parse_state_from_step(trajectory[step_index + 1])
        current_has_selection = any(call.get("name") in {"view_product_information", "budget_check"} for call in calls)
        if current_has_selection and (next_state.get("selection_issues") or next_state.get("last_errors")):
            reasons.append("selection_led_to_failure_feedback")

    # Keep deterministic order while deduplicating.
    return list(dict.fromkeys(reasons))


def chat_token_length(tokenizer, messages: list[dict[str, str]]) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )


def build_messages_for_target_step(
    trajectory: list[dict[str, Any]],
    target_step_idx: int,
    *,
    train_current_assistant: bool,
) -> tuple[list[dict[str, str]], list[int]]:
    first_prompt = trajectory[0].get("prompt") or []
    system_content = (first_prompt[0] if first_prompt else {}).get("content") or ""
    messages: list[dict[str, str]] = [{"role": "system", "content": str(system_content)}]
    assistant_loss_mask: list[int] = []

    for step_idx in range(target_step_idx + 1):
        step = trajectory[step_idx]
        prompt = step.get("prompt") or []
        user_content = (prompt[1] if len(prompt) > 1 else {}).get("content") or ""
        messages.append({"role": "user", "content": str(user_content)})
        messages.append({"role": "assistant", "content": completion_content(step)})
        assistant_loss_mask.append(1 if step_idx == target_step_idx and train_current_assistant else 0)

    return messages, assistant_loss_mask


def split_trajectory_indices(n: int, val_size: float, seed: int) -> tuple[set[int], set[int]]:
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_n = 0 if val_size == 0 else max(1, round(n * val_size))
    val = set(indices[:val_n])
    train = set(indices[val_n:])
    return train, val


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total": int(sum(values)),
        "min": int(min(values)),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(max(values)),
    }


def build_rows(args: argparse.Namespace) -> dict[str, Any]:
    input_path = ROOT / args.input
    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir
    trajectories = read_jsonl(input_path)
    manifests = read_jsonl(manifest_path)
    if len(trajectories) != len(manifests):
        raise ValueError(f"trajectory/manifest length mismatch: {len(trajectories)} vs {len(manifests)}")

    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_name, trust_remote_code=True)
    train_traj, val_traj = split_trajectory_indices(len(trajectories), args.val_size, args.seed)
    all_rows: list[dict[str, Any]] = []
    trainable_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    stats = Counter()
    mask_reason_counts = Counter()
    recovery_counts = Counter()
    tool_counts = Counter()
    token_lengths_all: list[int] = []
    token_lengths_trainable: list[int] = []
    masked_examples: list[dict[str, Any]] = []
    recovery_examples: list[dict[str, Any]] = []

    for trajectory_idx, (trajectory, manifest) in enumerate(zip(trajectories, manifests, strict=True)):
        split = "test" if trajectory_idx in val_traj else "train"
        previous_step_masked = False
        for step_idx, step in enumerate(trajectory):
            state = parse_state_from_step(step)
            reasons = current_step_negative_reasons(
                trajectory,
                step_idx,
                mask_decision_replacement_searches=args.mask_decision_replacement_searches,
            )
            sft_loss_mask = not reasons
            recovery_turn = bool(state_has_recovery_context(state) or previous_step_masked)
            if recovery_turn and sft_loss_mask:
                recovery_counts["trainable_recovery_turn"] += 1
                if len(recovery_examples) < 20:
                    recovery_examples.append(
                        {
                            "trajectory_idx": trajectory_idx,
                            "original_idx": manifest.get("original_idx"),
                            "step_idx": step_idx,
                            "harness_state": (step.get("extra_info") or {}).get("harness_state"),
                            "tool_sequence": list(tool_sequence(step)),
                            "state_failure_feedback": state_has_failure_feedback(state),
                            "state_previous_decision": bool(state.get("previous_decision")),
                            "previous_step_masked": previous_step_masked,
                        }
                    )
            if not sft_loss_mask:
                for reason in reasons:
                    mask_reason_counts[reason] += 1
                if len(masked_examples) < 40:
                    masked_examples.append(
                        {
                            "trajectory_idx": trajectory_idx,
                            "original_idx": manifest.get("original_idx"),
                            "step_idx": step_idx,
                            "harness_state": (step.get("extra_info") or {}).get("harness_state"),
                            "tool_sequence": list(tool_sequence(step)),
                            "mask_reasons": reasons,
                            "next_state": (
                                parse_state_from_step(trajectory[step_idx + 1]).get("state")
                                if step_idx + 1 < len(trajectory)
                                else None
                            ),
                        }
                    )

            if args.context_mode == "full_prefix":
                messages, assistant_loss_mask = build_messages_for_target_step(
                    trajectory,
                    step_idx,
                    train_current_assistant=sft_loss_mask,
                )
            else:
                messages = copy.deepcopy(step.get("prompt") or [])
                messages.append({"role": "assistant", "content": completion_content(step)})
                assistant_loss_mask = [1 if sft_loss_mask else 0]
            token_length = chat_token_length(tokenizer, messages)
            token_lengths_all.append(token_length)
            sequence = tool_sequence(step)
            tool_counts[str(sequence)] += 1
            row_extra = dict(step.get("extra_info") or {})
            row_extra.update(
                {
                    "split": split,
                    "step_sft_index": len(all_rows),
                    "trajectory_index": trajectory_idx,
                    "trajectory_step_index": step_idx,
                    "trajectory_steps": len(trajectory),
                    "original_idx": manifest.get("original_idx"),
                    "source_kind": manifest.get("source_kind"),
                    "source_run": manifest.get("source_run"),
                    "source_idx": manifest.get("source_idx"),
                    "tool_sequence": list(sequence),
                    "sft_loss_mask": int(sft_loss_mask),
                    "mask_reasons": reasons,
                    "recovery_turn": recovery_turn,
                    "state_failure_feedback": state_has_failure_feedback(state),
                    "state_previous_decision": bool(state.get("previous_decision")),
                    "previous_step_masked": previous_step_masked,
                    "token_length": token_length,
                }
            )
            row = {
                "messages": messages,
                "enable_thinking": False,
                "assistant_loss_mask": assistant_loss_mask,
                "sft_loss_mask": int(sft_loss_mask),
                "mask_reasons": reasons,
                "recovery_turn": recovery_turn,
                "token_length": token_length,
                "extra_info": row_extra,
            }
            all_rows.append(row)
            if sft_loss_mask:
                if token_length <= args.max_length:
                    trainable_rows_by_split[split].append(row)
                    token_lengths_trainable.append(token_length)
                else:
                    stats["dropped_trainable_over_max_length"] += 1
            else:
                stats["masked_rows"] += 1
            previous_step_masked = not sft_loss_mask

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_steps_with_mask.jsonl", all_rows)
    trainable_rows = trainable_rows_by_split["train"] + trainable_rows_by_split["test"]
    write_jsonl(output_dir / "trainable_steps.jsonl", trainable_rows)

    split_reports = {}
    for split, rows in trainable_rows_by_split.items():
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet")
        split_reports[split] = {
            "rows": len(rows),
            "token_length": numeric_summary([int(row["token_length"]) for row in rows]),
            "recovery_rows": sum(1 for row in rows if row["recovery_turn"]),
        }

    report = {
        "input": args.input,
        "manifest": args.manifest,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "trajectories": len(trajectories),
        "all_step_rows": len(all_rows),
        "trainable_step_rows": len(trainable_rows),
        "masked_step_rows": stats["masked_rows"],
        "max_length": args.max_length,
        "context_mode": args.context_mode,
        "dropped_trainable_over_max_length": stats["dropped_trainable_over_max_length"],
        "mask_decision_replacement_searches": args.mask_decision_replacement_searches,
        "mask_reason_counts": dict(mask_reason_counts),
        "recovery_counts": dict(recovery_counts),
        "tool_sequence_counts": dict(tool_counts),
        "token_length_all": numeric_summary(token_lengths_all),
        "token_length_trainable": numeric_summary(token_lengths_trainable),
        "splits": split_reports,
        "masked_examples": masked_examples,
        "recovery_examples": recovery_examples,
        "files": {
            "all_steps_with_mask": str(output_dir / "all_steps_with_mask.jsonl"),
            "trainable_steps": str(output_dir / "trainable_steps.jsonl"),
            "train_parquet": str(output_dir / "train.parquet"),
            "test_parquet": str(output_dir / "test.parquet"),
            "report": str(output_dir / "report.json"),
        },
    }
    write_json(output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build step-level state-local SFT rows from successful teacher rollouts.")
    parser.add_argument(
        "--input",
        default="data/tmp/teacher_gpt55medium_success_merged_20260709/clean_success_rollout.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/tmp/teacher_gpt55medium_success_merged_20260709/clean_success_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/shoppingbench_sft_state_local_step_clean924_masked",
    )
    parser.add_argument("--model-name", default="model/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=20480)
    parser.add_argument(
        "--context-mode",
        choices=["full_prefix", "current_step"],
        default="full_prefix",
        help=(
            "full_prefix matches RL rollout prefixes: system plus every previous state/assistant turn "
            "up to the current target assistant. current_step keeps only system/current state/current assistant."
        ),
    )
    parser.add_argument(
        "--mask-decision-replacement-searches",
        action="store_true",
        help=(
            "Also mask DECISION-phase find_product turns. By default these are kept because they are "
            "usually recovery actions after failure feedback; only negative outcomes are masked."
        ),
    )
    return parser.parse_args()


def main() -> None:
    report = build_rows(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
