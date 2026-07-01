#!/usr/bin/env python3
"""Rebuild voucher SFT data with hybrid or single-action assistant turns.

This consumes the existing state-folded teacher trajectories. It does not call
an LLM teacher again; it only reshapes already verified tool calls and their
stored observations, then rebuilds the online state-folded prompt after each
synthetic tool result.
"""

import argparse
import copy
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer

from prepare_verl_shoppingbench_data import build_system_prompt, count_chat_tokens


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from util.history_compression import build_state_folded_user_prompt, normalize_state_schema  # noqa: E402


USER_ROLES = ["user"]
ASSISTANT_ROLES = ["think", "tool_call", "obs", "response"]


STATE_RE = re.compile(r"<state>\s*(.*?)\s*</state>", re.DOTALL)


def parse_product_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def unique_extend(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value and value not in seen:
            target.append(value)
            seen.add(value)


def sanitize_role_content(content: str) -> str:
    for role in USER_ROLES + ASSISTANT_ROLES:
        content = content.replace(f"<{role}>", f"[{role}]")
        content = content.replace(f"</{role}>", f"[/{role}]")
    return content


@dataclass
class Message:
    user: str = ""
    think: str = ""
    tool_call: list[dict] = field(default_factory=list)
    obs: list[dict] = field(default_factory=list)
    response: str = ""

    def to_string(self, roles: list[str]) -> str:
        parts = []
        for role in roles:
            content = getattr(self, role)
            if not content:
                continue
            if isinstance(content, (dict, list)):
                text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            else:
                text = sanitize_role_content(str(content))
            parts.append(f"<{role}>{text}</{role}>")
        return "\n".join(parts)

    def clear(self) -> None:
        self.user = ""
        self.think = ""
        self.tool_call = []
        self.obs = []
        self.response = ""

    @classmethod
    def from_dict(cls, value: dict) -> "Message":
        return cls(
            think=value.get("think", ""),
            tool_call=copy.deepcopy(value.get("tool_call", []) or []),
            obs=copy.deepcopy(value.get("obs", []) or []),
            response=value.get("response", ""),
        )


def read_jsonl(path: Path) -> list:
    decoder = json.JSONDecoder()
    buffer = ""
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip() and not buffer:
                continue
            buffer += line
            while buffer:
                try:
                    row, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                rows.append(row)
                buffer = buffer[index:].strip()
    if buffer.strip():
        row, index = decoder.raw_decode(buffer)
        rows.append(row)
        if buffer[index:].strip():
            raise ValueError(f"Trailing non-JSON content in {path}")
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def split_trajectories(rows: list, val_size: float, seed: int) -> tuple[list, list]:
    indices = list(range(len(rows)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_n = 0 if val_size == 0 else max(1, round(len(rows) * val_size))
    val_idx = set(indices[:val_n])
    train, val = [], []
    for idx, row in enumerate(rows):
        (val if idx in val_idx else train).append(row)
    return train, val


def output_content(think: str, calls: list[dict]) -> str:
    public_calls = [
        {"name": call.get("name"), "parameters": call.get("parameters", {}) or {}}
        for call in calls
    ]
    return (
        f"<think>{think}</think>\n"
        f"<tool_call>{json.dumps(public_calls, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
    )


class ViewInfoCache:
    def __init__(self, cache_path: Path | None, endpoint: str) -> None:
        self.cache_path = cache_path
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False
        self.items: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            self.items = json.loads(cache_path.read_text(encoding="utf-8"))

    def seed(self, results) -> None:
        if not isinstance(results, list):
            return
        for item in results:
            if not isinstance(item, dict) or not item.get("product_id"):
                continue
            self.items[str(item["product_id"])] = item

    def get_many(self, product_ids: list[str]) -> list[dict]:
        missing = [pid for pid in product_ids if pid not in self.items]
        if missing:
            url = f"{self.endpoint}/view_product_information"
            resp = self.session.get(url, params={"product_ids": ",".join(missing)}, timeout=60)
            resp.raise_for_status()
            self.seed(resp.json())
        return [
            copy.deepcopy(self.items[pid])
            for pid in product_ids
            if pid in self.items
        ]

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.items, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def parse_state_from_prompt(user_prompt: str) -> dict | None:
    match = STATE_RE.search(user_prompt or "")
    if not match:
        return None
    try:
        state = json.loads(match.group(1))
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def shortlist_view_ids(state: dict | None, original_ids: list[str], top_k_per_search: int, max_ids: int) -> list[str]:
    if top_k_per_search <= 0:
        return original_ids
    per_search_ids: list[list[str]] = []
    for search in (state or {}).get("searches", []) or []:
        candidates = search.get("candidates", []) if isinstance(search, dict) else []
        per_search_ids.append(
            [
                str(item.get("product_id"))
                for item in candidates[:top_k_per_search]
                if isinstance(item, dict) and item.get("product_id")
            ]
        )

    candidate_ids: list[str] = []
    for rank in range(top_k_per_search):
        for search_ids in per_search_ids:
            if rank < len(search_ids):
                unique_extend(candidate_ids, [search_ids[rank]])

    if max_ids <= 0:
        max_ids = len(candidate_ids) + len(original_ids)

    result: list[str] = []
    reserved_originals = [pid for pid in original_ids if pid not in result]
    candidate_budget = max(0, max_ids - len(reserved_originals))
    unique_extend(result, candidate_ids[:candidate_budget])
    unique_extend(result, original_ids)
    if len(result) > max_ids:
        keep = []
        unique_extend(keep, original_ids)
        for pid in result:
            if len(keep) >= max_ids:
                break
            unique_extend(keep, [pid])
        result = keep
    return result


def maybe_augment_view_call(
    call: dict,
    obs: dict | None,
    user_prompt: str,
    config: dict,
    view_cache: ViewInfoCache | None,
    stats: Counter,
) -> tuple[dict, dict | None]:
    if call.get("name") != "view_product_information":
        return call, obs
    if config.get("view_shortlist_top_k_per_search", 0) <= 0:
        return call, obs

    original_ids = parse_product_ids((call.get("parameters") or {}).get("product_ids"))
    state = parse_state_from_prompt(user_prompt)
    shortlist = shortlist_view_ids(
        state,
        original_ids,
        config.get("view_shortlist_top_k_per_search", 0),
        config.get("view_shortlist_max_ids", 0),
    )
    if shortlist == original_ids:
        stats["view_shortlist_unchanged"] += 1
        return call, obs

    augmented_call = copy.deepcopy(call)
    augmented_call.setdefault("parameters", {})["product_ids"] = ",".join(shortlist)
    augmented_obs = copy.deepcopy(obs) if obs is not None else {"tool_call_id": call.get("tool_call_id")}
    if view_cache is None:
        stats["view_shortlist_missing_cache"] += 1
        return augmented_call, augmented_obs

    view_cache.seed((obs or {}).get("results"))
    augmented_obs["results"] = view_cache.get_many(shortlist)
    found = {str(item.get("product_id")) for item in augmented_obs["results"] if isinstance(item, dict)}
    missing = [pid for pid in shortlist if pid not in found]
    if missing:
        stats["view_shortlist_missing_products"] += 1
        augmented_call.setdefault("parameters", {})["product_ids"] = ",".join(
            [pid for pid in shortlist if pid in found]
        )
    stats["view_shortlist_augmented"] += 1
    stats[f"view_shortlist_size::{len(parse_product_ids(augmented_call['parameters']['product_ids']))}"] += 1
    return augmented_call, augmented_obs


def get_user_prompt(message: Message, history_messages: list[str], config: dict) -> str:
    user_message = message.to_string(USER_ROLES)
    if user_message:
        history_messages.append(user_message)

    assistant_message = message.to_string(ASSISTANT_ROLES)
    if assistant_message:
        history_messages.append(assistant_message)

    return build_state_folded_user_prompt(
        history_messages,
        max_candidates_per_search=config.get("state_max_candidates_per_search", 10),
        max_searches=config.get("state_max_searches"),
        max_budget_candidates=config.get("state_max_budget_candidates"),
        max_viewed_products=config.get("state_max_viewed_products"),
        never_expand=config.get("state_never_expand", False),
        min_char_saving_for_state=config.get("state_min_char_saving", 0.0),
    )


def single_action_think(original_think: str, call: dict, call_index: int, call_count: int) -> str:
    if call_count == 1 and original_think:
        return original_think
    name = call.get("name")
    params = call.get("parameters", {}) or {}
    if name == "find_product":
        q = str(params.get("q") or "")
        page = params.get("page")
        return f"Search this product requirement and retain candidate ids, shops, and prices; query={q!r}, page={page}."
    if name == "view_product_information":
        return "Verify the selected candidate product details against the user's requested attributes before recommending."
    if name == "python_execute":
        return "Compute voucher eligibility, payable total, and whether the selected products fit the user's budget from the current state."
    if name == "recommend_product":
        return "Recommend the verified product ids in the user's requested order."
    if name == "terminate":
        return "The recommendation step is complete, so terminate the interaction successfully."
    return original_think or f"Execute the next required tool action ({call_index + 1}/{call_count})."


def action_groups(calls: list[dict], split_mode: str) -> list[tuple[list[int], list[dict]]]:
    if split_mode == "single_action":
        return [([index], [call]) for index, call in enumerate(calls)]
    if split_mode != "hybrid":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    groups = []
    index = 0
    while index < len(calls):
        call = calls[index]
        if call.get("name") != "find_product":
            groups.append(([index], [call]))
            index += 1
            continue
        indices = []
        grouped_calls = []
        while index < len(calls) and calls[index].get("name") == "find_product":
            indices.append(index)
            grouped_calls.append(calls[index])
            index += 1
        groups.append((indices, grouped_calls))
    return groups


def group_think(original_think: str, calls: list[dict], call_indices: list[int], original_call_count: int) -> str:
    if len(calls) == 1:
        return single_action_think(original_think, calls[0], call_indices[0], original_call_count)
    names = {call.get("name") for call in calls}
    if names == {"find_product"}:
        if original_think:
            return original_think
        planned = []
        for call in calls:
            params = call.get("parameters", {}) or {}
            planned.append(f"(q={str(params.get('q') or '')!r}, page={params.get('page')})")
        return "Search the requested product requirements in parallel and retain candidate ids, shops, and prices; planned searches: " + ", ".join(planned) + "."
    return original_think or f"Execute {len(calls)} related tool actions in this assistant turn."


def group_label(sequence: tuple[str, ...]) -> str:
    if sequence and all(name == "find_product" for name in sequence):
        return "find_product_batch"
    return "+".join(sequence) if sequence else "NO_TOOL"


def update_transition_stats(sequences: list[tuple[str, ...]], stats: Counter) -> None:
    labels = [group_label(sequence) for sequence in sequences]
    for label in labels:
        stats[f"turn_label::{label}"] += 1
    for previous, current in zip(labels, labels[1:]):
        stats[f"transition::{previous}->{current}"] += 1


def prefixed_counts(stats: Counter, prefix: str) -> dict[str, int]:
    return {
        key[len(prefix):]: value
        for key, value in sorted(stats.items())
        if key.startswith(prefix)
    }


def obs_by_tool_call_id(step: dict) -> dict:
    obs = step.get("completion", {}).get("message", {}).get("obs", []) or []
    return {
        item.get("tool_call_id"): item
        for item in obs
        if isinstance(item, dict) and item.get("tool_call_id")
    }


def parse_python_result(results) -> dict | None:
    if not isinstance(results, dict):
        return None
    text = results.get("observation")
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def trajectory_quality_issue(trajectory: list[dict]) -> str | None:
    """Return a reason to drop a teacher trajectory, or None if it is clean."""
    saw_recommend = False
    latest_budget = None
    for step in trajectory:
        msg = step.get("completion", {}).get("message", {}) or {}
        observations = obs_by_tool_call_id(step)
        for call in msg.get("tool_call", []) or []:
            name = call.get("name")
            if name == "python_execute":
                obs = observations.get(call.get("tool_call_id"), {})
                latest_budget = parse_python_result(obs.get("results"))
                required = {
                    "product_ids",
                    "shop_ids",
                    "total_before_voucher",
                    "voucher_used",
                    "payable_total",
                    "budget",
                    "within_budget",
                }
                if latest_budget is None:
                    return "python_execute_unparseable_or_failed"
                missing = sorted(required - set(latest_budget))
                if missing:
                    return "python_execute_missing_" + ",".join(missing)
            elif name == "recommend_product":
                saw_recommend = True
                if not isinstance(latest_budget, dict):
                    return "recommend_without_budget_calculation"
                if latest_budget.get("within_budget") is not True:
                    return "recommend_after_not_within_budget"
    if not saw_recommend:
        return "missing_recommend_product"
    return None


def split_teacher_trajectory(
    trajectory: list[dict],
    config: dict,
    system_prompt: str,
    view_cache: ViewInfoCache | None = None,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    if not trajectory:
        return [], stats

    query = trajectory[0].get("extra_info", {}).get("query", "")
    history_messages: list[str] = []
    message = Message(user=query)
    split_steps = []
    trajectory_sequences = []
    split_mode = config.get("split_mode", "hybrid")

    for original_step_index, step in enumerate(trajectory):
        completion = step.get("completion", {})
        msg = completion.get("message", {}) or {}
        calls = copy.deepcopy(msg.get("tool_call", []) or [])
        if not calls:
            stats["empty_tool_steps"] += 1
            continue
        original_think = msg.get("think") or completion.get("reasoning_content") or ""
        observations = obs_by_tool_call_id(step)
        stats[f"original_call_count::{len(calls)}"] += 1
        groups = action_groups(calls, split_mode)
        stats[f"original_group_count::{len(groups)}"] += 1

        for call_indices, group_calls in groups:
            user_prompt = get_user_prompt(message, history_messages, config)
            message.clear()

            normalized_group_calls = []
            group_obs = []
            for call in group_calls:
                call_id = call.get("tool_call_id")
                obs = copy.deepcopy(observations.get(call_id))
                if obs is None:
                    stats["missing_observation"] += 1
                    obs = {"tool_call_id": call_id, "results": None}

                call, obs = maybe_augment_view_call(call, obs, user_prompt, config, view_cache, stats)
                normalized_group_calls.append(call)
                group_obs.append(obs)

            sequence = tuple(call.get("name") for call in normalized_group_calls)
            trajectory_sequences.append(sequence)
            think = group_think(original_think, normalized_group_calls, call_indices, len(calls))
            content = output_content(think, normalized_group_calls)
            group_message = {
                "think": think,
                "tool_call": copy.deepcopy(normalized_group_calls),
                "obs": group_obs,
            }

            extra_info = dict(step.get("extra_info") or {})
            extra_info.update(
                {
                    "original_step": extra_info.get("step", original_step_index + 1),
                    "original_call_index": call_indices[0],
                    "original_call_indices": call_indices,
                    "original_call_count": len(calls),
                    "single_action_sft": split_mode == "single_action",
                    "hybrid_action_sft": split_mode == "hybrid",
                    "split_mode": split_mode,
                    "group_call_count": len(normalized_group_calls),
                    "group_tool_names": list(sequence),
                    "step": len(split_steps) + 1,
                }
            )
            split_steps.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "completion": {
                        "reasoning_content": think,
                        "content": content,
                        "message": group_message,
                    },
                    "extra_info": extra_info,
                }
            )

            message = Message.from_dict(group_message)
            stats[f"tool_sequence::{sequence}"] += 1

    update_transition_stats(trajectory_sequences, stats)
    return split_steps, stats


def rewrite_state_blocks(text: str, stats: Counter) -> str:
    def replace(match: re.Match) -> str:
        raw_state = match.group(1).strip()
        state = json.loads(raw_state)
        normalized = normalize_state_schema(state)
        stats["state_count"] += 1
        if isinstance(normalized, dict) and "pending" in normalized:
            stats["pending_after_normalize"] += 1
        return (
            "<state>"
            + json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</state>"
        )

    return STATE_RE.sub(replace, text)


def step_to_parquet_row(step: dict, idx: int, split: str, system_prompt: str, tokenizer, max_length: int, stats: Counter):
    messages = copy.deepcopy(step["prompt"])
    messages[0] = {"role": "system", "content": system_prompt}
    for item in messages:
        if item.get("role") == "user":
            item["content"] = rewrite_state_blocks(str(item.get("content") or ""), stats)
    messages.append({"role": "assistant", "content": step["completion"]["content"]})
    length = count_chat_tokens(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if length > max_length:
        stats["dropped_overlong"] += 1
        return None
    extra_info = dict(step.get("extra_info") or {})
    extra_info.update({"split": split, "index": idx})
    return {
        "messages": messages,
        "enable_thinking": False,
        "extra_info": extra_info,
        "token_length": length,
    }


def tool_sequence_from_content(content: str) -> tuple[str, ...]:
    match = re.search(r"<tool_call>(.*?)</tool_call>", content or "", flags=re.DOTALL)
    if not match:
        return ()
    try:
        calls = json.loads(match.group(1))
    except Exception:
        return ("PARSE_ERROR",)
    if isinstance(calls, dict):
        calls = [calls]
    return tuple(call.get("name") for call in calls if isinstance(call, dict))


def parse_tool_weights(raw: str) -> dict[str, int]:
    if not raw:
        return {}
    weights: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --train-tool-weights item: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        try:
            weight = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer weight for {name!r}: {value!r}") from exc
        if not name or weight < 1:
            raise ValueError(f"Tool weights must use nonempty names and positive integers: {item!r}")
        weights[name] = weight
    return weights


def row_tool_name(row: dict) -> str:
    sequence = tool_sequence_from_content(row["messages"][-1]["content"])
    return sequence[0] if len(sequence) == 1 else ""


def maybe_duplicate_train_row(row: dict, train_tool_weights: dict[str, int], stats: Counter) -> list[dict]:
    tool = row_tool_name(row)
    weight = train_tool_weights.get(tool, 1)
    if weight <= 1:
        return [row]
    rows = [row]
    for duplicate_index in range(1, weight):
        duplicate = copy.deepcopy(row)
        duplicate["extra_info"] = dict(duplicate.get("extra_info") or {})
        duplicate["extra_info"]["stage_balance_duplicate_index"] = duplicate_index
        duplicate["extra_info"]["stage_balance_source_tool"] = tool
        rows.append(duplicate)
        stats[f"stage_balance_duplicate::{tool}"] += 1
    return rows


def build_dataset(args) -> dict:
    source = read_jsonl(ROOT / args.input)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_name, trust_remote_code=True)
    system_prompt = build_system_prompt(ROOT / args.prompt_file)
    train_tool_weights = parse_tool_weights(args.train_tool_weights)
    config = {
        "split_mode": args.split_mode,
        "history_compression": "state_folded",
        "state_max_candidates_per_search": args.state_max_candidates_per_search,
        "state_max_searches": args.state_max_searches,
        "state_max_budget_candidates": args.state_max_budget_candidates,
        "state_max_viewed_products": args.state_max_viewed_products,
        "state_never_expand": False,
        "state_min_char_saving": 0.0,
        "view_shortlist_top_k_per_search": args.view_shortlist_top_k_per_search,
        "view_shortlist_max_ids": args.view_shortlist_max_ids,
        "train_tool_weights": train_tool_weights,
    }
    view_cache = (
        ViewInfoCache(ROOT / args.view_info_cache, args.search_endpoint)
        if args.view_shortlist_top_k_per_search > 0
        else None
    )

    split_rows = []
    split_stats = Counter()
    for trajectory in source:
        issue = trajectory_quality_issue(trajectory)
        if issue:
            split_stats[f"dropped_trajectory::{issue}"] += 1
            continue
        split_trajectory, stats = split_teacher_trajectory(trajectory, config, system_prompt, view_cache)
        split_rows.append(split_trajectory)
        split_stats.update(stats)

    if view_cache is not None:
        view_cache.save()

    output_jsonl = ROOT / args.output_jsonl
    write_jsonl(output_jsonl, split_rows)

    train_traj, val_traj = split_trajectories(split_rows, args.val_size, args.seed)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": args.input,
        "output_jsonl": args.output_jsonl,
        "output_dir": args.output_dir,
        "split_mode": args.split_mode,
        "trajectories": len(split_rows),
        "view_shortlist_top_k_per_search": args.view_shortlist_top_k_per_search,
        "view_shortlist_max_ids": args.view_shortlist_max_ids,
        "train_tool_weights": train_tool_weights,
        "split_stats": dict(split_stats),
        "transition_counts": prefixed_counts(split_stats, "transition::"),
        "turn_label_counts": prefixed_counts(split_stats, "turn_label::"),
        "splits": {},
    }

    for split, trajectories in (("train", train_traj), ("test", val_traj)):
        rows = []
        stats = Counter()
        tool_counts = Counter()
        for trajectory_idx, trajectory in enumerate(trajectories):
            trajectory_sequences = []
            for step_idx, step in enumerate(trajectory):
                item = step_to_parquet_row(
                    step,
                    len(rows),
                    split,
                    system_prompt,
                    tokenizer,
                    args.max_length,
                    stats,
                )
                if item is None:
                    continue
                item["extra_info"]["trajectory_index"] = trajectory_idx
                item["extra_info"]["trajectory_step_index"] = step_idx
                sequence = tool_sequence_from_content(item["messages"][-1]["content"])
                trajectory_sequences.append(sequence)
                expanded_items = (
                    maybe_duplicate_train_row(item, train_tool_weights, stats)
                    if split == "train"
                    else [item]
                )
                for expanded_item in expanded_items:
                    expanded_item["extra_info"]["index"] = len(rows)
                    tool_counts[tool_sequence_from_content(expanded_item["messages"][-1]["content"])] += 1
                    rows.append(expanded_item)
            update_transition_stats(trajectory_sequences, stats)
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet")
        token_lengths = [row["token_length"] for row in rows]
        report["splits"][split] = {
            "trajectories": len(trajectories),
            "rows": len(rows),
            "stats": dict(stats),
            "tool_sequences": {str(k): v for k, v in tool_counts.items()},
            "transition_counts": prefixed_counts(stats, "transition::"),
            "turn_label_counts": prefixed_counts(stats, "turn_label::"),
            "max_tokens": max(token_lengths, default=0),
            "mean_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        }

    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid-action or single-action state-folded voucher SFT data.")
    parser.add_argument("--input", default="data/teacher_voucher_train_clean691_state_folded.jsonl")
    parser.add_argument("--output-jsonl", default="data/teacher_voucher_train_clean691_state_folded_hybrid_action.jsonl")
    parser.add_argument("--output-dir", default="dataset/shoppingbench_sft_state_folded_hybrid_action_schemav3")
    parser.add_argument("--report", default="reports/sft_hybrid_action_schemav3_data_20260701.json")
    parser.add_argument(
        "--split-mode",
        choices=["hybrid", "single_action"],
        default="hybrid",
        help="hybrid batches consecutive find_product calls from the same original assistant turn; single_action splits every tool call.",
    )
    parser.add_argument("--prompt-file", default="src/agent/prompt/rollout.md")
    parser.add_argument("--model-name", default="model/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=20480)
    parser.add_argument("--state-max-candidates-per-search", type=int, default=10)
    parser.add_argument("--state-max-searches", type=int, default=12)
    parser.add_argument("--state-max-budget-candidates", type=int, default=120)
    parser.add_argument("--state-max-viewed-products", type=int, default=40)
    parser.add_argument("--view-shortlist-top-k-per-search", type=int, default=0)
    parser.add_argument("--view-shortlist-max-ids", type=int, default=10)
    parser.add_argument("--view-info-cache", default="data/cache/view_product_information_cache.json")
    parser.add_argument("--search-endpoint", default="http://127.0.0.1:5631")
    parser.add_argument(
        "--train-tool-weights",
        default="",
        help=(
            "Comma-separated train-only integer oversampling weights, e.g. "
            "view_product_information=3,python_execute=2,recommend_product=3,terminate=4."
        ),
    )
    return parser.parse_args()


def main() -> None:
    report = build_dataset(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
