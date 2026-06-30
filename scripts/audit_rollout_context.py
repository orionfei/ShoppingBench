#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT_DIR / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from util.message import Message, OUTPUT_ROLES  # noqa: E402


DEFAULT_TOKENIZER = "model/Qwen3-4B"
DEFAULT_MAX_PROMPT_LENGTH = 16384
DEFAULT_MAX_RESPONSE_LENGTH = 1024
DEFAULT_MAX_TOTAL_LENGTH = 18000
DEFAULT_SFT_CUTOFF_LEN = 20480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ShoppingBench rollout trajectories against Qwen/AgticRL "
            "prompt, response, and total token limits."
        )
    )
    parser.add_argument("rollout_file", help="Path to a rollout jsonl file.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=(
            "Tokenizer/model path for AutoTokenizer. Prefer the exact Qwen3-4B "
            "training model path; defaults to the local Qwen3 tokenizer fallback."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help="AgticRL/verl data.max_prompt_length.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=DEFAULT_MAX_RESPONSE_LENGTH,
        help="AgticRL/verl data.max_response_length.",
    )
    parser.add_argument(
        "--max-total-length",
        type=int,
        default=DEFAULT_MAX_TOTAL_LENGTH,
        help="Per-sample prompt + normalized output token budget.",
    )
    parser.add_argument(
        "--sft-cutoff-len",
        type=int,
        default=DEFAULT_SFT_CUTOFF_LEN,
        help="SFT cutoff length used for trajectory-level diagnostics.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full audit summary as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to write the per-step audit table as CSV.",
    )
    parser.add_argument(
        "--fail-on-raw-content",
        action="store_true",
        help=(
            "Treat raw completion.content over max response length as a failure. "
            "By default raw content over limit is reported as a warning because "
            "training should use normalized output."
        ),
    )
    parser.add_argument(
        "--fail-on-trajectory-concat",
        action="store_true",
        help=(
            "Treat whole-trajectory concatenation over SFT cutoff as a failure. "
            "By default this is reported as a diagnostic because per-step samples "
            "are the intended training format."
        ),
    )
    return parser.parse_args()


def load_tokenizer(name_or_path: str):
    tokenizer_path = Path(name_or_path)
    if not tokenizer_path.is_absolute():
        candidate = ROOT_DIR / tokenizer_path
        if candidate.exists():
            name_or_path = str(candidate)
    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        local_files_only=Path(name_or_path).exists(),
    )


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def render_prompt(tokenizer, prompt: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return "\n\n".join(
        f"<{message.get('role', '')}>\n{message.get('content', '')}\n</{message.get('role', '')}>"
        for message in prompt
    )


def normalized_message(completion: dict) -> tuple[Message, str]:
    message_dict = completion.get("message")
    if isinstance(message_dict, dict) and message_dict:
        message = Message.from_dict(message_dict)
    else:
        message = Message.from_string(
            completion.get("reasoning_content") or "",
            completion.get("content") or "",
        )
    return message, message.to_string(OUTPUT_ROLES)


def load_rollout_rows(path: Path) -> list[list[dict]]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, list):
                raise ValueError(f"{path}:{line_no}: rollout row must be a list")
            rows.append(row)
    return rows


def audit_rollout(args: argparse.Namespace) -> dict:
    rollout_file = Path(args.rollout_file)
    tokenizer = load_tokenizer(args.tokenizer)
    rows = load_rollout_rows(rollout_file)

    step_records = []
    concatenated_normalized_parts = []
    concatenated_raw_parts = []

    for row_idx, row in enumerate(rows, 1):
        query = ""
        if row:
            query = row[0].get("extra_info", {}).get("query", "")
        for step_idx, step in enumerate(row, 1):
            prompt = step.get("prompt") or []
            completion = step.get("completion") or {}
            rendered_prompt = render_prompt(tokenizer, prompt)
            prompt_json = json.dumps(prompt, ensure_ascii=False)
            message, normalized_output = normalized_message(completion)
            raw_content = completion.get("content") or ""

            prompt_chat_tokens = token_count(tokenizer, rendered_prompt)
            prompt_json_tokens = token_count(tokenizer, prompt_json)
            normalized_output_tokens = token_count(tokenizer, normalized_output)
            raw_content_tokens = token_count(tokenizer, raw_content)
            prompt_plus_normalized_tokens = prompt_chat_tokens + normalized_output_tokens

            record = {
                "row": row_idx,
                "step": step_idx,
                "query": query,
                "prompt_chat_tokens": prompt_chat_tokens,
                "prompt_json_tokens": prompt_json_tokens,
                "normalized_output_tokens": normalized_output_tokens,
                "raw_content_tokens": raw_content_tokens,
                "prompt_plus_normalized_tokens": prompt_plus_normalized_tokens,
                "tool_names": "|".join(
                    call.get("name", "") for call in (message.tool_call or [])
                ),
                "response_chars": len(message.response or ""),
                "passes_prompt_chat_limit": prompt_chat_tokens <= args.max_prompt_length,
                "passes_prompt_json_limit": prompt_json_tokens <= args.max_prompt_length,
                "passes_normalized_response_limit": normalized_output_tokens
                <= args.max_response_length,
                "passes_prompt_plus_normalized_limit": prompt_plus_normalized_tokens
                <= args.max_total_length,
                "raw_content_within_response_limit": raw_content_tokens
                <= args.max_response_length,
            }
            record["passes_step_limits"] = (
                record["passes_prompt_chat_limit"]
                and record["passes_prompt_json_limit"]
                and record["passes_normalized_response_limit"]
                and record["passes_prompt_plus_normalized_limit"]
            )
            if args.fail_on_raw_content:
                record["passes_step_limits"] = (
                    record["passes_step_limits"]
                    and record["raw_content_within_response_limit"]
                )
            step_records.append(record)

            concatenated_normalized_parts.append(rendered_prompt)
            concatenated_normalized_parts.append(normalized_output)
            concatenated_raw_parts.append(rendered_prompt)
            concatenated_raw_parts.append(raw_content)

    concat_normalized_tokens = token_count(
        tokenizer, "\n\n".join(concatenated_normalized_parts)
    )
    concat_raw_tokens = token_count(tokenizer, "\n\n".join(concatenated_raw_parts))
    trajectory_single_sample_allowed = (
        concat_normalized_tokens <= args.sft_cutoff_len
        and concat_raw_tokens <= args.sft_cutoff_len
    )

    failures = [
        record
        for record in step_records
        if not record["passes_step_limits"]
    ]
    raw_content_warnings = [
        record
        for record in step_records
        if not record["raw_content_within_response_limit"]
    ]

    summary = {
        "rollout_file": str(rollout_file),
        "tokenizer": args.tokenizer,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "limits": {
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
            "sft_cutoff_len": args.sft_cutoff_len,
            "fail_on_raw_content": args.fail_on_raw_content,
            "fail_on_trajectory_concat": args.fail_on_trajectory_concat,
        },
        "num_rows": len(rows),
        "num_steps": len(step_records),
        "max_prompt_chat_tokens": max_value(step_records, "prompt_chat_tokens"),
        "max_prompt_json_tokens": max_value(step_records, "prompt_json_tokens"),
        "max_normalized_output_tokens": max_value(
            step_records, "normalized_output_tokens"
        ),
        "max_raw_content_tokens": max_value(step_records, "raw_content_tokens"),
        "max_prompt_plus_normalized_tokens": max_value(
            step_records, "prompt_plus_normalized_tokens"
        ),
        "naive_concat_all_step_prompts_and_normalized_outputs_tokens": concat_normalized_tokens,
        "naive_concat_all_step_prompts_and_raw_outputs_tokens": concat_raw_tokens,
        "trajectory_single_sample_allowed": trajectory_single_sample_allowed,
        "passes_per_step_limits": not failures,
        "raw_content_warning_count": len(raw_content_warnings),
        "failure_count": len(failures),
        "steps": step_records,
    }
    summary["passes_audit"] = summary["passes_per_step_limits"]
    if args.fail_on_trajectory_concat:
        summary["passes_audit"] = (
            summary["passes_audit"] and trajectory_single_sample_allowed
        )
    return summary


def max_value(records: list[dict], key: str) -> dict | None:
    if not records:
        return None
    record = max(records, key=lambda item: item[key])
    return {
        "value": record[key],
        "row": record["row"],
        "step": record["step"],
    }


def write_json(path: str, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def write_csv(path: str, records: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "step",
        "prompt_chat_tokens",
        "prompt_json_tokens",
        "normalized_output_tokens",
        "raw_content_tokens",
        "prompt_plus_normalized_tokens",
        "tool_names",
        "response_chars",
        "passes_prompt_chat_limit",
        "passes_prompt_json_limit",
        "passes_normalized_response_limit",
        "passes_prompt_plus_normalized_limit",
        "raw_content_within_response_limit",
        "passes_step_limits",
    ]
    with target.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})


def print_table(summary: dict) -> None:
    print(
        "row step prompt_chat prompt_json norm_out raw_content prompt+norm "
        "pass_step raw_ok tools"
    )
    for record in summary["steps"]:
        print(
            f"{record['row']:>3} {record['step']:>4} "
            f"{record['prompt_chat_tokens']:>11} "
            f"{record['prompt_json_tokens']:>11} "
            f"{record['normalized_output_tokens']:>8} "
            f"{record['raw_content_tokens']:>11} "
            f"{record['prompt_plus_normalized_tokens']:>11} "
            f"{str(record['passes_step_limits']):>9} "
            f"{str(record['raw_content_within_response_limit']):>6} "
            f"{record['tool_names']}"
        )

    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"steps"}
    }
    print("\nSUMMARY")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    summary = audit_rollout(args)
    print_table(summary)
    if args.output_json:
        write_json(args.output_json, summary)
        print(f"\nWrote JSON audit: {args.output_json}")
    if args.output_csv:
        write_csv(args.output_csv, summary["steps"])
        print(f"Wrote CSV audit: {args.output_csv}")
    return 0 if summary["passes_audit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
