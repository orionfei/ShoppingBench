#!/usr/bin/env python3
import argparse
import copy
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT_DIR / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

OUTPUT_ROLES = ["think", "tool_call", "response"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build state-folded SFT prompts for ShoppingBench voucher trajectories."
    )
    parser.add_argument(
        "--input-rollout",
        default="data/voucher_obs_compression_10_compact.jsonl",
        help="Input rollout JSONL. Each line is one trajectory list.",
    )
    parser.add_argument(
        "--full-rollout",
        default="data/voucher_obs_compression_10_full.jsonl",
        help="Optional raw/full rollout JSONL for comparison.",
    )
    parser.add_argument(
        "--synthesize-file",
        default="data/synthesize_voucher_train.jsonl",
        help="Voucher synthesize JSONL containing query, reward, and voucher.",
    )
    parser.add_argument(
        "--output-rollout",
        default="data/voucher_state_folded_10.jsonl",
        help="Output rollout JSONL with folded history prompts.",
    )
    parser.add_argument(
        "--output-sft",
        default="data/voucher_state_folded_10_sft.json",
        help="Output alpaca-style SFT JSON list.",
    )
    parser.add_argument(
        "--report-json",
        default="data/voucher_state_folded_10_report.json",
        help="Output report JSON.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional local tokenizer path/name. If loading fails, approximate tokens are used.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_synthesize(path: Path) -> dict:
    by_query = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            by_query[item["query"]] = {
                "reward": item.get("reward", []),
                "voucher": item.get("voucher", {}),
            }
    return by_query


def load_tokenizer(name: str):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        path = Path(name)
        local_only = path.exists()
        return AutoTokenizer.from_pretrained(
            str(path) if local_only else name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
    except Exception:
        return None


def token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return math.ceil(len((text or "").encode("utf-8")) / 4)
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def prompt_user_content(step: dict) -> str:
    prompt = step.get("prompt") or []
    if len(prompt) < 2:
        return ""
    return prompt[1].get("content", "")


def system_content(row: list[dict]) -> str:
    if not row:
        return ""
    prompt = row[0].get("prompt") or []
    if not prompt:
        return ""
    return prompt[0].get("content", "")


def completion_output(step: dict) -> str:
    return message_to_string(step["completion"]["message"], OUTPUT_ROLES)


def sanitize_role_content(content: str) -> str:
    for role in ["user", "think", "tool_call", "obs", "response"]:
        content = content.replace(f"<{role}>", f"[{role}]")
        content = content.replace(f"</{role}>", f"[/{role}]")
    return content


def message_to_string(message: dict, roles: list[str]) -> str:
    parts = []
    for role in roles:
        content = message.get(role)
        if not content:
            continue
        if role == "tool_call" and isinstance(content, list):
            content = [
                {
                    "name": call.get("name"),
                    "parameters": call.get("parameters", {}),
                }
                for call in content
            ]
        if isinstance(content, (dict, list)):
            text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(content, str):
            text = sanitize_role_content(content)
        else:
            raise TypeError(f"Invalid message role content: {role}={type(content)}")
        parts.append(f"<{role}>{text}</{role}>")
    return "\n".join(parts)


def local_format_reward(completion: str, roles: list[str] | None = None) -> float:
    if roles is None:
        roles = OUTPUT_ROLES

    pos = {}
    for role in roles:
        start = [m.start() for m in re.finditer(f"<{role}>", completion)]
        end = [m.start() for m in re.finditer(f"</{role}>", completion)]
        if start or end:
            pos[role] = (start, end)

    if "think" in roles and "think" not in pos:
        return 0
    if "tool_call" not in pos and "response" not in pos:
        return 0

    for role, (start, end) in pos.items():
        if len(start) != 1 or len(end) != 1 or start[0] >= end[0]:
            return 0

    if "tool_call" in pos:
        try:
            tool_call_str = completion[pos["tool_call"][0][0] : pos["tool_call"][1][0]]
            tool_call_str = tool_call_str.replace("<tool_call>", "").replace("</tool_call>", "")
            tool_call = json.loads(tool_call_str)
            if not isinstance(tool_call, list):
                return 0
            for call in tool_call:
                if not isinstance(call.get("name"), str):
                    return 0
                if not isinstance(call.get("parameters"), dict):
                    return 0
        except Exception:
            return 0

    for i, role_i in enumerate(roles):
        for j, role_j in enumerate(roles):
            if i == j or role_i not in pos or role_j not in pos:
                continue
            if pos[role_i][0][0] < pos[role_j][0][0] < pos[role_i][1][0]:
                return 0
            if pos[role_i][0][0] < pos[role_j][1][0] < pos[role_i][1][0]:
                return 0
    return 1


def final_recommended_ids(row: list[dict]) -> list[str]:
    ids = []
    for step in row:
        for call in step.get("completion", {}).get("message", {}).get("tool_call", []) or []:
            if call.get("name") == "recommend_product":
                raw = call.get("parameters", {}).get("product_ids", "")
                ids = [part.strip() for part in raw.split(",") if part.strip()]
    return ids


def voucher_state(voucher: dict) -> dict:
    discount = {"type": voucher.get("discount_type")}
    if voucher.get("discount_type") == "fixed":
        discount["value"] = voucher.get("face_value")
    elif voucher.get("discount_type") == "percentage":
        discount["rate"] = voucher.get("discount")
        discount["cap"] = voucher.get("cap")
    return {
        "scope": voucher.get("voucher_type"),
        "threshold": voucher.get("threshold"),
        "discount": discount,
        "budget": voucher.get("budget"),
    }


def product_requirements(reward: list[dict]) -> list[dict]:
    slots = []
    for idx, item in enumerate(reward, 1):
        req = {"slot_id": idx, "status": "open"}
        fields = []
        if item.get("title"):
            fields.append("title")
        if item.get("attributes"):
            for attr in item["attributes"]:
                fields.extend(attr.keys())
        if item.get("sku_options"):
            for sku in item["sku_options"]:
                fields.extend(sku.keys())
        req["required_fields"] = sorted(set(fields))
        slots.append(req)
    return slots


def slim_product(product: dict) -> dict:
    return {
        key: product[key]
        for key in ("product_id", "shop_id", "title", "price", "service")
        if key in product
    }


def parse_python_observation(results):
    if not isinstance(results, dict):
        return None
    text = results.get("observation")
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        return {"observation": text.strip(), "success": results.get("success")}


def payable_total(total: float, shop_ids: set[str], voucher: dict) -> tuple[bool, float]:
    if total is None:
        return False, total
    eligible_scope = voucher.get("voucher_type") == "platform" or (
        voucher.get("voucher_type") == "shop" and len(shop_ids) == 1
    )
    meets_threshold = total >= (voucher.get("threshold") or 0)
    if not eligible_scope or not meets_threshold:
        return False, total
    if voucher.get("discount_type") == "fixed":
        return True, total - (voucher.get("face_value") or 0)
    if voucher.get("discount_type") == "percentage":
        rate = voucher.get("discount") or 0
        cap = voucher.get("cap") or 0
        return True, max(total * (1 - rate), total - cap)
    return False, total


def build_state(
    query_meta: dict,
    previous_steps: list[dict],
    final_ids: list[str],
) -> dict:
    reward = query_meta.get("reward", [])
    voucher = query_meta.get("voucher", {})
    selected_by_id = {}
    observed_counts = defaultdict(int)
    previous_tools = []
    verified_details = {}
    calculations = []

    for step_idx, step in enumerate(previous_steps, 1):
        message = step.get("completion", {}).get("message", {}) or {}
        calls = message.get("tool_call", []) or []
        obs = message.get("obs", []) or []
        previous_tools.append({"step": step_idx, "tools": [call.get("name") for call in calls]})
        obs_by_id = {item.get("tool_call_id"): item for item in obs}
        for call in calls:
            name = call.get("name")
            call_obs = obs_by_id.get(call.get("tool_call_id"), {})
            results = call_obs.get("results")
            if name == "find_product" and isinstance(results, list):
                observed_counts["find_product_candidates"] += len(results)
                for product in results:
                    pid = str(product.get("product_id", ""))
                    if pid in final_ids and pid not in selected_by_id:
                        selected_by_id[pid] = slim_product(product)
            elif name == "view_product_information" and isinstance(results, list):
                for product in results:
                    pid = str(product.get("product_id", ""))
                    verified_details[pid] = {
                        "product_id": pid,
                        "sku_options": product.get("sku_options", {}),
                        "attributes": product.get("attributes", {}),
                    }
            elif name == "python_execute":
                parsed = parse_python_observation(results)
                if parsed is not None:
                    calculations.append(parsed)

    ordered_selected = [
        selected_by_id[pid] for pid in final_ids if pid in selected_by_id
    ]
    selected_ids = [item["product_id"] for item in ordered_selected]
    selected_shops = {str(item.get("shop_id")) for item in ordered_selected if item.get("shop_id")}
    selected_total = round(sum(float(item.get("price") or 0) for item in ordered_selected), 2)
    voucher_used, discounted_total = payable_total(selected_total, selected_shops, voucher)
    discounted_total = round(discounted_total, 2) if discounted_total is not None else None

    slots = product_requirements(reward)
    for idx, product in enumerate(ordered_selected):
        if idx < len(slots):
            slots[idx]["status"] = "candidate_selected"
            slots[idx]["product"] = product

    pending = []
    if len(ordered_selected) < len(reward):
        if voucher.get("voucher_type") == "shop" and selected_shops:
            pending.append("find_remaining_products_inside_shop_anchor")
        else:
            pending.append("find_remaining_products")
    elif selected_ids and any(pid not in verified_details for pid in selected_ids):
        pending.append("verify_product_information")
        pending.append("check_voucher_budget")
    elif not calculations:
        pending.append("check_voucher_budget")
    else:
        last_calc = calculations[-1]
        if isinstance(last_calc, dict) and last_calc.get("within_budget") is True:
            pending.append("recommend_products_and_terminate")
        else:
            pending.append("revise_selection_or_fail")

    state = {
        "task_type": "voucher_budget",
        "voucher": voucher_state(voucher),
        "requested_product_count": len(reward),
        "slots": slots,
        "selected_product_ids": selected_ids,
        "selected_candidates": ordered_selected,
        "shop_anchor": next(iter(selected_shops), None)
        if voucher.get("voucher_type") == "shop" and selected_shops
        else None,
        "selected_total_before_voucher": selected_total,
        "voucher_applicable_if_now": voucher_used,
        "payable_total_if_now": discounted_total,
        "within_budget_if_now": discounted_total <= voucher.get("budget", -1)
        if discounted_total is not None and voucher.get("budget") is not None
        else None,
        "verified_details": [
            verified_details[pid] for pid in selected_ids if pid in verified_details
        ],
        "latest_budget_calculation": calculations[-1] if calculations else None,
        "observed_counts": dict(observed_counts),
        "previous_tools": previous_tools,
        "pending": pending,
    }
    return state


def folded_prompt(query: str, state: dict | None) -> str:
    parts = [f"<user>{query}</user>"]
    if state:
        parts.append(
            "<state>"
            + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</state>"
        )
    return "# Dialogue Records History\n" + "\n\n".join(parts)


def fold_row(row: list[dict], synthesize_by_query: dict) -> list[dict]:
    query = row[0].get("extra_info", {}).get("query", "")
    meta = synthesize_by_query.get(query, {"reward": [], "voucher": {}})
    final_ids = final_recommended_ids(row)
    folded = []
    for step_idx, step in enumerate(row):
        new_step = copy.deepcopy(step)
        state = None
        if step_idx > 0:
            state = build_state(meta, row[:step_idx], final_ids)
        new_step["prompt"] = [
            {"role": "system", "content": system_content(row)},
            {"role": "user", "content": folded_prompt(query, state)},
        ]
        new_step.setdefault("extra_info", {})["history_compression"] = "state_folded"
        folded.append(new_step)
    return folded


def state_text_from_step(step: dict) -> str:
    content = prompt_user_content(step)
    match = re.search(r"<state>(.+)</state>", content, re.DOTALL)
    return match.group(1) if match else ""


def state_from_step(step: dict) -> dict:
    text = state_text_from_step(step)
    if not text:
        return {}
    return json.loads(text)


def dependency_checks(row: list[dict]) -> list[dict]:
    records = []
    for idx, step in enumerate(row, 1):
        state = state_from_step(step)
        selected_ids = set(state.get("selected_product_ids", []))
        shop_anchor = state.get("shop_anchor")
        checks = []
        for call in step.get("completion", {}).get("message", {}).get("tool_call", []) or []:
            name = call.get("name")
            params = call.get("parameters", {}) or {}
            if name == "find_product" and params.get("shop_id"):
                checks.append(
                    {
                        "tool": name,
                        "field": "shop_id",
                        "ok": str(params["shop_id"]) == str(shop_anchor),
                    }
                )
            elif name in {"view_product_information", "recommend_product"}:
                raw_ids = params.get("product_ids", "")
                ids = {part.strip() for part in raw_ids.split(",") if part.strip()}
                checks.append(
                    {
                        "tool": name,
                        "field": "product_ids",
                        "ok": bool(ids) and ids.issubset(selected_ids),
                    }
                )
            elif name == "python_execute":
                checks.append(
                    {
                        "tool": name,
                        "field": "budget_inputs",
                        "ok": bool(state.get("selected_candidates"))
                        and bool(state.get("voucher")),
                    }
                )
        records.append(
            {
                "step": idx,
                "has_state": bool(state),
                "checks": checks,
                "ok": all(item["ok"] for item in checks),
            }
        )
    return records


def build_sft(rows: list[list[dict]]) -> list[dict]:
    samples = []
    for row in rows:
        for step in row:
            prompt = step.get("prompt") or []
            samples.append(
                {
                    "instruction": prompt[0]["content"] if prompt else "",
                    "input": prompt[1]["content"] if len(prompt) > 1 else "",
                    "output": completion_output(step),
                    "extra_info": step.get("extra_info", {}),
                }
            )
    return samples


def eval_rollout_without_search(row: list[dict], query_meta: dict) -> dict:
    message_outputs = [
        {"completion": {"message": step["completion"]["message"]}} for step in row
    ]
    recommended = final_recommended_ids(row)
    reward_ids = [str(item.get("product_id")) for item in query_meta.get("reward", [])]
    voucher = query_meta.get("voucher", {})
    products = []
    for pid in recommended:
        for step in row:
            msg = step.get("completion", {}).get("message", {})
            for obs in msg.get("obs", []) or []:
                results = obs.get("results")
                if not isinstance(results, list):
                    continue
                for item in results:
                    if isinstance(item, dict) and str(item.get("product_id")) == pid and "price" in item:
                        products.append(item)
                        break
                if products and str(products[-1].get("product_id")) == pid:
                    break
            if products and str(products[-1].get("product_id")) == pid:
                break

    total = round(sum(float(item.get("price") or 0) for item in products), 2)
    shops = {str(item.get("shop_id")) for item in products if item.get("shop_id")}
    _, payable = payable_total(total, shops, voucher)
    payable = round(payable, 2) if payable is not None else None
    return {
        "recommended_ids": recommended,
        "matches_reward_ids": recommended == reward_ids,
        "product_count_ok": len(recommended) == len(reward_ids),
        "same_shop_if_needed": voucher.get("voucher_type") != "shop" or len(shops) == 1,
        "total_before_voucher": total,
        "payable_total": payable,
        "within_budget": payable <= voucher.get("budget", -1)
        if payable is not None and voucher.get("budget") is not None
        else False,
        "format_ok": all(
            local_format_reward(message_to_string(step["completion"]["message"], OUTPUT_ROLES))
            >= 1
            for step in row
        ),
        "message_outputs": len(message_outputs),
    }


def summarize(
    full_rows: list[list[dict]],
    compact_rows: list[list[dict]],
    folded_rows: list[list[dict]],
    synthesize_by_query: dict,
    tokenizer,
) -> dict:
    totals = defaultdict(int)
    row_reports = []
    for row_idx, folded_row in enumerate(folded_rows, 1):
        compact_row = compact_rows[row_idx - 1]
        full_row = full_rows[row_idx - 1] if full_rows else compact_row
        query = folded_row[0].get("extra_info", {}).get("query", "")
        meta = synthesize_by_query.get(query, {"reward": [], "voucher": {}})

        full_prompt = "\n\n".join(prompt_user_content(step) for step in full_row)
        compact_prompt = "\n\n".join(prompt_user_content(step) for step in compact_row)
        folded_prompt_text = "\n\n".join(prompt_user_content(step) for step in folded_row)
        full_chars = len(full_prompt)
        compact_chars = len(compact_prompt)
        folded_chars = len(folded_prompt_text)
        full_tokens = token_count(tokenizer, full_prompt)
        compact_tokens = token_count(tokenizer, compact_prompt)
        folded_tokens = token_count(tokenizer, folded_prompt_text)

        for key, value in {
            "full_prompt_chars": full_chars,
            "compact_prompt_chars": compact_chars,
            "folded_prompt_chars": folded_chars,
            "full_prompt_tokens": full_tokens,
            "compact_prompt_tokens": compact_tokens,
            "folded_prompt_tokens": folded_tokens,
        }.items():
            totals[key] += value

        deps = dependency_checks(folded_row)
        local_eval = eval_rollout_without_search(folded_row, meta)
        row_reports.append(
            {
                "row": row_idx,
                "query": query,
                "steps": len(folded_row),
                "full_prompt_chars": full_chars,
                "compact_prompt_chars": compact_chars,
                "folded_prompt_chars": folded_chars,
                "compact_to_folded_char_savings": compact_chars - folded_chars,
                "compact_to_folded_char_savings_pct": round(
                    (compact_chars - folded_chars) / compact_chars * 100, 2
                )
                if compact_chars
                else 0,
                "full_prompt_tokens": full_tokens,
                "compact_prompt_tokens": compact_tokens,
                "folded_prompt_tokens": folded_tokens,
                "dependency_checks_ok": all(item["ok"] for item in deps),
                "dependency_checks": deps,
                "local_eval": local_eval,
            }
        )

    summary = dict(totals)
    summary["num_queries"] = len(folded_rows)
    summary["num_steps"] = sum(len(row) for row in folded_rows)
    summary["compact_to_folded_char_savings"] = (
        summary["compact_prompt_chars"] - summary["folded_prompt_chars"]
    )
    summary["compact_to_folded_char_savings_pct"] = round(
        summary["compact_to_folded_char_savings"]
        / summary["compact_prompt_chars"]
        * 100,
        2,
    )
    summary["full_to_folded_char_savings"] = (
        summary["full_prompt_chars"] - summary["folded_prompt_chars"]
    )
    summary["full_to_folded_char_savings_pct"] = round(
        summary["full_to_folded_char_savings"] / summary["full_prompt_chars"] * 100,
        2,
    )
    summary["compact_to_folded_token_savings"] = (
        summary["compact_prompt_tokens"] - summary["folded_prompt_tokens"]
    )
    summary["compact_to_folded_token_savings_pct"] = round(
        summary["compact_to_folded_token_savings"]
        / summary["compact_prompt_tokens"]
        * 100,
        2,
    )
    summary["full_to_folded_token_savings"] = (
        summary["full_prompt_tokens"] - summary["folded_prompt_tokens"]
    )
    summary["full_to_folded_token_savings_pct"] = round(
        summary["full_to_folded_token_savings"]
        / summary["full_prompt_tokens"]
        * 100,
        2,
    )
    summary["dependency_all_ok"] = all(
        row["dependency_checks_ok"] for row in row_reports
    )
    summary["local_eval_all_ok"] = all(
        row["local_eval"]["matches_reward_ids"]
        and row["local_eval"]["within_budget"]
        and row["local_eval"]["format_ok"]
        for row in row_reports
    )
    return {"summary": summary, "rows": row_reports}


def main() -> None:
    args = parse_args()
    input_rollout = ROOT_DIR / args.input_rollout
    full_rollout = ROOT_DIR / args.full_rollout
    synthesize_file = ROOT_DIR / args.synthesize_file
    output_rollout = ROOT_DIR / args.output_rollout
    output_sft = ROOT_DIR / args.output_sft
    report_json = ROOT_DIR / args.report_json

    compact_rows = read_jsonl(input_rollout)
    full_rows = read_jsonl(full_rollout) if full_rollout.exists() else []
    synthesize_by_query = load_synthesize(synthesize_file)
    folded_rows = [fold_row(row, synthesize_by_query) for row in compact_rows]
    sft_samples = build_sft(folded_rows)
    tokenizer = load_tokenizer(args.tokenizer)
    report = summarize(full_rows, compact_rows, folded_rows, synthesize_by_query, tokenizer)
    report["summary"]["input_rollout"] = str(input_rollout)
    report["summary"]["output_rollout"] = str(output_rollout)
    report["summary"]["output_sft"] = str(output_sft)
    report["summary"]["report_json"] = str(report_json)
    report["summary"]["tokenizer"] = args.tokenizer or "approx_utf8_bytes_div_4"

    write_jsonl(output_rollout, folded_rows)
    with output_sft.open("w", encoding="utf-8") as fout:
        json.dump(sft_samples, fout, ensure_ascii=False, indent=2)
    with report_json.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
