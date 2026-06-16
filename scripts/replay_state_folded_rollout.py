#!/usr/bin/env python3
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from run_rollout import act, get_system_prompt, get_user_prompt, is_terminate  # noqa: E402
from util.message import Message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one or more voucher trajectories through the online "
            "state-folded rollout prompt path. The policy action is replayed "
            "from an existing trajectory; tools are executed again locally."
        )
    )
    parser.add_argument(
        "--input-rollout",
        default="data/voucher_obs_compression_10_compact.jsonl",
        help="Existing trajectory JSONL used as the replay policy.",
    )
    parser.add_argument(
        "--output-rollout",
        default="data/voucher_state_folded_replay_1.jsonl",
        help="Output replay rollout JSONL.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--system-prompt-file",
        default="src/agent/prompt/rollout.md",
    )
    parser.add_argument("--max-candidates-per-search", type=int, default=10)
    return parser.parse_args()


def read_rows(path: Path) -> list[list[dict]]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_action_message(step: dict) -> Message:
    message = copy.deepcopy(step["completion"]["message"])
    # Tool observations must come from this replay run, not from the source row.
    message.pop("obs", None)
    return Message.from_dict(message)


def extract_state(prompt: str) -> dict:
    match = re.search(r"<state>(.+)</state>", prompt, re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group(1))


def ids_from_params(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def state_candidate_ids(state: dict) -> set[str]:
    ids = set(state.get("selected_product_ids") or [])
    for search in state.get("searches") or []:
        for candidate in search.get("candidates") or []:
            if candidate.get("product_id"):
                ids.add(str(candidate["product_id"]))
    for product in state.get("viewed_products") or []:
        if product.get("product_id"):
            ids.add(str(product["product_id"]))
    return ids


def check_action_supported_by_state(prompt: str, message: Message) -> dict:
    state = extract_state(prompt)
    if not state:
        return {"has_state": False, "checks": [], "ok": True}

    candidate_ids = state_candidate_ids(state)
    checks = []
    for call in message.tool_call or []:
        name = call.get("name")
        params = call.get("parameters", {}) or {}
        if name == "find_product" and params.get("shop_id"):
            shop_id = str(params["shop_id"])
            state_shop_ids = {
                str(candidate.get("shop_id"))
                for search in state.get("searches") or []
                for candidate in search.get("candidates") or []
                if candidate.get("shop_id") is not None
            }
            if state.get("shop_anchor"):
                state_shop_ids.add(str(state["shop_anchor"]))
            checks.append(
                {
                    "tool": name,
                    "field": "shop_id",
                    "value": shop_id,
                    "ok": shop_id in state_shop_ids,
                }
            )
        elif name in {"view_product_information", "recommend_product"}:
            ids = ids_from_params(params.get("product_ids"))
            checks.append(
                {
                    "tool": name,
                    "field": "product_ids",
                    "value": ids,
                    "ok": bool(ids) and set(ids).issubset(candidate_ids),
                }
            )
        elif name == "python_execute":
            checks.append(
                {
                    "tool": name,
                    "field": "budget_inputs",
                    "ok": bool(candidate_ids) and bool(state.get("voucher")),
                }
            )
    return {"has_state": True, "checks": checks, "ok": all(item["ok"] for item in checks)}


def replay_row(source_row: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    query = source_row[0]["extra_info"]["query"]
    system_prompt = get_system_prompt(config)
    history_messages = []
    message = Message(user=query)
    output_row = []
    support_records = []

    for step_idx, source_step in enumerate(source_row, 1):
        user_prompt = get_user_prompt(message, history_messages, config)
        message.clear()
        message = gold_action_message(source_step)
        support = check_action_supported_by_state(user_prompt, message)
        support["step"] = step_idx
        support_records.append(support)

        if message.tool_call:
            message.obs = act(message)

        output_row.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": {
                    "reasoning_content": source_step["completion"].get(
                        "reasoning_content", ""
                    ),
                    "content": source_step["completion"].get("content", ""),
                    "message": copy.deepcopy(message.to_dict()),
                },
                "extra_info": {
                    "step": step_idx,
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "history_compression": "state_folded",
                    "replay_policy": True,
                },
            }
        )
        if is_terminate(message):
            break
    return output_row, support_records


def main() -> None:
    args = parse_args()
    rows = read_rows(ROOT / args.input_rollout)
    selected_rows = rows[args.start_index : args.start_index + args.limit]
    config = {
        "task": "voucher",
        "system_prompt_file": args.system_prompt_file,
        "exclude_tools": ["web_search"],
        "history_compression": "state_folded",
        "state_max_candidates_per_search": args.max_candidates_per_search,
    }
    replayed_rows = []
    support = []
    for idx, row in enumerate(selected_rows, args.start_index + 1):
        replayed, records = replay_row(row, config)
        replayed_rows.append(replayed)
        support.append({"source_row": idx, "records": records})

    output = ROOT / args.output_rollout
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for row in replayed_rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "input_rollout": str(ROOT / args.input_rollout),
        "output_rollout": str(output),
        "rows": len(replayed_rows),
        "support_all_ok": all(
            record["ok"]
            for row_support in support
            for record in row_support["records"]
        ),
        "support": support,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
