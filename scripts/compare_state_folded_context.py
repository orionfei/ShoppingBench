#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
sys.path.insert(0, str(AGENT_SRC))

from util.history_compression import build_state_folded_user_prompt
from util.message import Message, ASSISTANT_ROLES, USER_ROLES


def load_tokenizer(name_or_path: str):
    path = Path(name_or_path)
    if not path.is_absolute() and (ROOT / path).exists():
        name_or_path = str(ROOT / path)
    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        local_files_only=Path(name_or_path).exists(),
    )


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def render_chat(tokenizer, system_message: dict, user_prompt: str) -> str:
    messages = [system_message, {"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return json.dumps(messages, ensure_ascii=False)


def load_rollouts(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values):
    return statistics.mean(values) if values else 0


def pct_saving(raw: int, folded: int) -> float:
    if raw <= 0:
        return 0.0
    return 1.0 - (folded / raw)


def compare_rollout(row, tokenizer, args):
    records = []
    query = row[0]["extra_info"]["query"] if row else ""
    history_messages = []
    message = Message(user=query)

    for step_idx, step in enumerate(row, 1):
        user_message = message.to_string(USER_ROLES)
        if user_message:
            history_messages.append(user_message)
        assistant_message = message.to_string(ASSISTANT_ROLES)
        if assistant_message:
            history_messages.append(assistant_message)

        raw_user_prompt = "# Dialogue Records History\n" + "\n\n".join(history_messages)
        folded_user_prompt = build_state_folded_user_prompt(
            history_messages,
            max_candidates_per_search=args.max_candidates_per_search,
            max_searches=args.max_searches,
            max_budget_candidates=args.max_budget_candidates,
            max_viewed_products=args.max_viewed_products,
            never_expand=args.never_expand,
            min_char_saving_for_state=args.min_char_saving,
        )
        system_message = (step.get("prompt") or [{}])[0]
        actual_user_prompt = ""
        prompt = step.get("prompt") or []
        if len(prompt) > 1:
            actual_user_prompt = prompt[1].get("content") or ""

        raw_chat_tokens = token_count(tokenizer, render_chat(tokenizer, system_message, raw_user_prompt))
        folded_chat_tokens = token_count(tokenizer, render_chat(tokenizer, system_message, folded_user_prompt))
        actual_chat_tokens = token_count(tokenizer, render_chat(tokenizer, system_message, actual_user_prompt))
        raw_user_tokens = token_count(tokenizer, raw_user_prompt)
        folded_user_tokens = token_count(tokenizer, folded_user_prompt)

        records.append(
            {
                "step": step_idx,
                "raw_user_tokens": raw_user_tokens,
                "folded_user_tokens": folded_user_tokens,
                "raw_chat_tokens": raw_chat_tokens,
                "folded_chat_tokens": folded_chat_tokens,
                "actual_chat_tokens": actual_chat_tokens,
                "chat_token_saving": pct_saving(raw_chat_tokens, folded_chat_tokens),
                "user_token_saving": pct_saving(raw_user_tokens, folded_user_tokens),
                "actual_matches_recomputed_folded": actual_user_prompt == folded_user_prompt,
            }
        )

        completion_message = (step.get("completion") or {}).get("message") or {}
        message = Message.from_dict(completion_message) if completion_message else Message()

    return {"query": query, "steps": len(row), "records": records}


def summarize(rows):
    records = [record for row in rows for record in row["records"]]
    raw_chat = [item["raw_chat_tokens"] for item in records]
    folded_chat = [item["folded_chat_tokens"] for item in records]
    raw_user = [item["raw_user_tokens"] for item in records]
    folded_user = [item["folded_user_tokens"] for item in records]
    savings = [item["chat_token_saving"] for item in records]
    user_savings = [item["user_token_saving"] for item in records]
    return {
        "num_rows": len(rows),
        "num_steps": len(records),
        "mean_raw_chat_tokens": mean(raw_chat),
        "mean_folded_chat_tokens": mean(folded_chat),
        "max_raw_chat_tokens": max(raw_chat) if raw_chat else 0,
        "max_folded_chat_tokens": max(folded_chat) if folded_chat else 0,
        "mean_raw_user_tokens": mean(raw_user),
        "mean_folded_user_tokens": mean(folded_user),
        "max_raw_user_tokens": max(raw_user) if raw_user else 0,
        "max_folded_user_tokens": max(folded_user) if folded_user else 0,
        "mean_chat_token_saving": mean(savings),
        "mean_user_token_saving": mean(user_savings),
        "min_chat_token_saving": min(savings) if savings else 0,
        "all_actual_prompts_match_recomputed_folded": all(
            item["actual_matches_recomputed_folded"] for item in records
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare raw concatenated history with state-folded history.")
    parser.add_argument("rollout_file")
    parser.add_argument("--tokenizer", default="model/Qwen3-4B")
    parser.add_argument("--max-candidates-per-search", type=int, default=10)
    parser.add_argument("--max-searches", type=int, default=12)
    parser.add_argument("--max-budget-candidates", type=int, default=120)
    parser.add_argument("--max-viewed-products", type=int, default=40)
    parser.add_argument("--never-expand", action="store_true")
    parser.add_argument("--min-char-saving", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    rows = [
        compare_rollout(row, tokenizer, args)
        for row in load_rollouts(ROOT / args.rollout_file)
    ]
    report = {
        "rollout_file": args.rollout_file,
        "tokenizer": args.tokenizer,
        "state_limits": {
            "max_candidates_per_search": args.max_candidates_per_search,
            "max_searches": args.max_searches,
            "max_budget_candidates": args.max_budget_candidates,
            "max_viewed_products": args.max_viewed_products,
            "never_expand": args.never_expand,
            "min_char_saving": args.min_char_saving,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    if args.output_json:
        output = ROOT / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fout:
            json.dump(report, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
