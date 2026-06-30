#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from prepare_verl_shoppingbench_data import build_system_prompt, count_chat_tokens


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from util.history_compression import normalize_state_schema  # noqa: E402

STATE_RE = re.compile(r"<state>\s*(.*?)\s*</state>", re.DOTALL)
OLD_STATE_LABELS = (
    "terminate_if_not_done",
    "check_voucher_budget",
    "revise_selection_or_fail",
    "verify_product_information",
    "recommend_products",
    "recommend_products_and_terminate",
    "select_candidates_from_search_results",
    "search_products",
    "find_remaining_products_inside_shop_anchor",
    "find_remaining_products",
)


def messages_to_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def rewrite_state_blocks(text: str, stats: Counter) -> str:
    def replace(match: re.Match) -> str:
        raw_state = match.group(1).strip()
        try:
            state = json.loads(raw_state)
        except Exception as exc:
            raise ValueError(f"Failed to parse <state> JSON: {raw_state[:200]}") from exc
        if isinstance(state, dict) and "pending" in state:
            stats["old_pending_state_count"] += 1
            pending = state.get("pending")
            if isinstance(pending, list):
                for item in pending:
                    stats[f"old_pending_value::{item}"] += 1
        normalized = normalize_state_schema(state)
        if isinstance(normalized, dict) and "decision_hint" in normalized:
            stats["decision_hint_state_count"] += 1
        if isinstance(normalized, dict) and "pending" in normalized:
            stats["new_pending_state_count"] += 1
        return (
            "<state>"
            + json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</state>"
        )

    state_count = len(STATE_RE.findall(text))
    stats["state_count"] += state_count
    return STATE_RE.sub(replace, text)


def rewrite_split(input_path: Path, output_path: Path, system_prompt: str, tokenizer) -> dict:
    df = pd.read_parquet(input_path)
    rows = []
    token_lengths = []
    old_placeholder = 0
    new_placeholder = 0
    rewrite_stats = Counter()
    steps = Counter()

    for row in df.to_dict("records"):
        messages = messages_to_list(row["messages"])
        if not messages or messages[0].get("role") != "system":
            raise ValueError(f"{input_path} has a row whose first message is not system")
        old_text = str(messages[0].get("content") or "")
        if '"tool name"' in old_text or "parameter1" in old_text:
            old_placeholder += 1
        messages[0] = {"role": "system", "content": system_prompt}
        for item in messages:
            if item.get("role") == "user":
                item["content"] = rewrite_state_blocks(str(item.get("content") or ""), rewrite_stats)
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if '"tool name"' in joined or "parameter1" in joined:
            new_placeholder += 1
        if '"pending"' in joined:
            rewrite_stats["new_pending_literal_rows"] += 1
        for label in OLD_STATE_LABELS:
            if label in joined:
                rewrite_stats[f"pseudo_label_literal_rows::{label}"] += 1
        length = count_chat_tokens(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        row["messages"] = messages
        row["token_length"] = length
        token_lengths.append(length)
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            steps[str(extra_info.get("step"))] += 1
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "old_system_placeholder_rows": old_placeholder,
        "new_placeholder_rows": new_placeholder,
        "state_rewrite": dict(rewrite_stats),
        "step": dict(steps),
        "max_tokens": max(token_lengths, default=0),
        "mean_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild state-folded SFT parquets with the current rollout prompt and state schema.")
    parser.add_argument("--input-dir", default="dataset/shoppingbench_sft_state_folded")
    parser.add_argument("--output-dir", default="dataset/shoppingbench_sft_state_folded_schemav3")
    parser.add_argument("--prompt-file", default="src/agent/prompt/rollout.md")
    parser.add_argument("--model-name", default="model/Qwen3-4B")
    parser.add_argument("--report", default="reports/sft_schemav3_data_20260629.json")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_name, trust_remote_code=True)
    system_prompt = build_system_prompt(ROOT / args.prompt_file)
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir

    report = {
        "source_dir": args.input_dir,
        "output_dir": args.output_dir,
        "prompt_file": args.prompt_file,
        "system_prompt_chars": len(system_prompt),
        "splits": {},
    }
    for split in ("train", "test"):
        report["splits"][split] = rewrite_split(
            input_dir / f"{split}.parquet",
            output_dir / f"{split}.parquet",
            system_prompt,
            tokenizer,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
