#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from toolkit import toolmap  # noqa: E402
from util.harness_fsm import (  # noqa: E402
    STATE_CANDIDATE_SELECT,
    build_compact_harness_user_prompt,
    build_harness_snapshot,
    is_duplicate_find_product_in_turn,
    is_repeated_failed_search,
    is_repeated_search,
    search_trace_markdown,
)
from util.llm import ask_llm  # noqa: E402
from util.message import ASSISTANT_ROLES, USER_ROLES, Message, extract_json_value, generate_tool_call_id  # noqa: E402
from util.system_prompt import build_system_prompt  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            if row is None:
                continue
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_broad8(rows: list[dict]) -> list[dict]:
    targets = [
        ("platform", "fixed", 1),
        ("platform", "percentage", 1),
        ("shop", "fixed", 2),
        ("shop", "percentage", 2),
        ("platform", "fixed", 3),
        ("platform", "percentage", 3),
        ("shop", "fixed", 4),
        ("shop", "percentage", 4),
    ]
    selected = []
    used = set()
    for voucher_type, discount_type, product_count in targets:
        for idx, row in enumerate(rows):
            voucher = row.get("voucher") or {}
            if idx in used:
                continue
            if (
                voucher.get("voucher_type") == voucher_type
                and voucher.get("discount_type") == discount_type
                and len(row.get("reward") or []) == product_count
            ):
                selected.append(copy.deepcopy(row))
                used.add(idx)
                break
    if len(selected) < 8:
        def score(idx_row):
            idx, row = idx_row
            voucher = row.get("voucher") or {}
            return (
                len(row.get("reward") or []),
                voucher.get("voucher_type", ""),
                voucher.get("discount_type", ""),
                idx,
            )

        for idx, row in sorted(enumerate(rows), key=score):
            if idx not in used:
                selected.append(copy.deepcopy(row))
                used.add(idx)
                if len(selected) >= 8:
                    break
    return selected[:8]


def state_local_snapshot(history_messages: list[str], max_failed_searches: int):
    snapshot = build_harness_snapshot(history_messages, allow_legacy_python_budget=False)
    if snapshot.state_name == STATE_CANDIDATE_SELECT:
        snapshot.include_tools = {"view_product_information", "budget_check"}
    if isinstance(snapshot.state, dict) and max_failed_searches:
        if "failed_searches" in snapshot.state:
            snapshot.state["failed_searches"] = (snapshot.state.get("failed_searches") or [])[-max_failed_searches:]
        if "failed_retry_searches" in snapshot.state:
            snapshot.state["failed_retry_searches"] = (
                snapshot.state.get("failed_retry_searches") or []
            )[-max_failed_searches:]
    return snapshot


def parse_product_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def products_for_budget_check(product_ids: list[str], snapshot):
    state = snapshot.state if snapshot is not None and isinstance(snapshot.state, dict) else {}
    products_by_id = {}
    for product in state.get("candidate_pool") or []:
        if isinstance(product, dict) and product.get("product_id") is not None:
            products_by_id[str(product["product_id"])] = product
    missing = [product_id for product_id in product_ids if product_id not in products_by_id]
    if missing:
        return [], {"error": "budget_check_product_ids_not_in_candidate_pool", "missing_product_ids": missing}
    if len(product_ids) != len(set(product_ids)):
        return [], {"error": "budget_check_duplicate_product_ids"}
    products = [products_by_id[product_id] for product_id in product_ids]
    for product in products:
        if float_or_none(product.get("price")) is None or product.get("shop_id") is None:
            return [], {"error": "budget_check_missing_price_or_shop_id"}
    return products, None


def compute_voucher_discount(products: list[dict], voucher: dict):
    if not isinstance(voucher, dict):
        voucher = {"type": "none"}
    voucher_type = str(voucher.get("type", "none")).lower()
    if voucher_type in {"none", "no_voucher", "null"}:
        return 0.0, False, None

    total = sum(float(product["price"]) for product in products)
    scope_shop_id = voucher.get("scope_shop_id") or voucher.get("shop_id")
    if "shop" in voucher_type or scope_shop_id:
        if scope_shop_id is None:
            selected_shops = {str(product.get("shop_id")) for product in products}
            if len(selected_shops) != 1:
                return 0.0, False, {"error": "shop_voucher_requires_single_shop_or_scope_shop_id"}
            scope_shop_id = next(iter(selected_shops))
        eligible_total = sum(
            float(product["price"])
            for product in products
            if str(product.get("shop_id")) == str(scope_shop_id)
        )
    else:
        eligible_total = total

    threshold = float_or_none(voucher.get("threshold"))
    if threshold is not None and eligible_total + 1e-9 < threshold:
        return 0.0, False, None

    discount = float_or_none(voucher.get("discount"))
    if discount is None:
        discount = float_or_none(voucher.get("amount"))
    if discount is None:
        discount = float_or_none(voucher.get("value"))
    rate = float_or_none(voucher.get("rate"))
    cap = float_or_none(voucher.get("cap"))
    if rate is not None:
        if rate > 1:
            rate = rate / 100.0
        discount = eligible_total * rate
        if cap is not None:
            discount = min(discount, cap)
    if discount is None:
        return 0.0, False, {"error": "unsupported_voucher_schema"}
    discount = max(0.0, min(float(discount), eligible_total))
    return discount, discount > 0, None


def budget_check_response(call: dict, snapshot) -> dict:
    parameters = call.get("parameters") or {}
    product_ids = parse_product_ids(parameters.get("product_ids"))
    products, error = products_for_budget_check(product_ids, snapshot)
    if error is None and not product_ids:
        error = {"error": "budget_check_requires_product_ids"}
    if error is None:
        discount, voucher_used, error = compute_voucher_discount(products, parameters.get("voucher") or {"type": "none"})
    if error is None and float_or_none(parameters.get("budget")) is None:
        error = {"error": "budget_check_requires_numeric_budget"}
    if error is not None:
        return {**error, "_tool_success": False, "_parse_ok": True}
    budget = float_or_none(parameters.get("budget"))
    total = round(sum(float(product["price"]) for product in products), 2)
    payable_total = round(max(0.0, total - discount), 2)
    return {
        "product_ids": product_ids,
        "shop_ids": [str(product.get("shop_id")) for product in products],
        "total_before_voucher": total,
        "voucher_used": bool(voucher_used),
        "voucher_applied": bool(voucher_used),
        "payable_total": payable_total,
        "budget": budget,
        "within_budget": payable_total <= budget,
        "agent_voucher": parameters.get("voucher") or {"type": "none"},
        "_tool_success": True,
        "_parse_ok": True,
    }


def selected_product_ids(snapshot) -> set[str]:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    result = set()
    for product in state.get("selected_products") or []:
        if isinstance(product, dict) and product.get("product_id") is not None:
            result.add(str(product["product_id"]))
    return result


def budget_product_ids(snapshot) -> set[str]:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    budget = state.get("budget_calculation") or {}
    ids = budget.get("product_ids")
    if not isinstance(ids, list):
        return set()
    return {str(item) for item in ids if item is not None}


def viewed_product_ids(snapshot) -> set[str]:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    return {
        str(product.get("product_id"))
        for product in state.get("viewed_products") or []
        if isinstance(product, dict) and product.get("product_id") is not None
    }


def budget_result(snapshot) -> dict:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    budget = state.get("budget_calculation") or {}
    return budget if isinstance(budget, dict) else {}


def verified_shop_ids(snapshot, product_ids: set[str]) -> set[str]:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    shops = set()
    for product in state.get("selected_products") or []:
        if not isinstance(product, dict):
            continue
        product_id = product.get("product_id")
        if product_id is not None and str(product_id) in product_ids and product.get("shop_id") is not None:
            shops.add(str(product.get("shop_id")))
    budget = budget_result(snapshot)
    budget_ids = [str(item) for item in budget.get("product_ids") or [] if item is not None]
    budget_shops = [str(item) for item in budget.get("shop_ids") or [] if item is not None]
    for product_id, shop_id in zip(budget_ids, budget_shops):
        if product_id in product_ids:
            shops.add(shop_id)
    return shops


def budget_agent_voucher_requires_single_shop(snapshot) -> bool:
    voucher = budget_result(snapshot).get("agent_voucher") or {}
    if not isinstance(voucher, dict):
        return False
    voucher_type = str(voucher.get("type", "")).lower()
    return "shop" in voucher_type or bool(voucher.get("scope_shop_id") or voucher.get("shop_id"))


def validation_error(call: dict, all_calls: list[dict], snapshot) -> dict | None:
    allowed_tools = snapshot.include_tools
    names = {item.get("name") for item in all_calls}
    params = call.get("parameters") or {}
    if call.get("name") == "find_product":
        if not params.get("q") or params.get("page") in (None, ""):
            return {
                "error": "find_product_requires_q_and_page",
                "tool": "find_product",
                "required_fix": 'Use parameters {"q": string, "page": integer}.',
            }
        sort = params.get("sort")
        if sort not in (None, "", "priceasc", "pricedesc", "order", "default"):
            return {
                "error": "find_product_invalid_sort",
                "tool": "find_product",
                "required_fix": 'Use sort only as one of "priceasc", "pricedesc", "order", or "default".',
            }
        service = params.get("service")
        if service:
            allowed_services = {"official", "freeShipping", "COD", "flashsale", "default"}
            invalid_services = [
                item.strip()
                for item in str(service).split(",")
                if item.strip() and item.strip() not in allowed_services
            ]
            if invalid_services:
                return {
                    "error": "find_product_invalid_service",
                    "tool": "find_product",
                    "invalid_service_values": invalid_services,
                    "required_fix": 'Use service values exactly from "COD", "freeShipping", "flashsale", "official", joined by comma if needed.',
                }
    elif call.get("name") == "view_product_information":
        if not params.get("product_ids"):
            return {
                "error": "view_product_information_requires_product_ids",
                "tool": "view_product_information",
                "required_fix": 'Use parameters {"product_ids": "id1,id2"}.',
            }
    elif call.get("name") == "budget_check":
        if not params.get("product_ids") or params.get("budget") in (None, ""):
            return {
                "error": "budget_check_requires_product_ids_and_budget",
                "tool": "budget_check",
                "required_fix": 'Use parameters {"product_ids": [string, ...], "voucher": object, "budget": number}.',
            }
    elif call.get("name") == "recommend_product":
        if not params.get("product_ids"):
            return {
                "error": "recommend_product_requires_product_ids",
                "tool": "recommend_product",
                "required_fix": 'Use parameters {"product_ids": "id1,id2"}.',
            }
    elif call.get("name") == "terminate":
        if params.get("status") != "success":
            return {
                "error": "terminate_status_must_be_success",
                "tool": "terminate",
                "required_fix": 'Use parameters {"status": "success"}.',
            }
    if snapshot.state_name == STATE_CANDIDATE_SELECT and "budget_check" in names:
        if "view_product_information" not in names:
            return {
                "error": "budget_check_requires_view_product_information_in_select",
                "tool": "selection",
                "required_fix": "Call view_product_information and budget_check for the same selected product ids in the same turn.",
            }
        view_ids, budget_ids = [], []
        for item in all_calls:
            params = item.get("parameters") or {}
            if item.get("name") == "view_product_information":
                view_ids.extend(parse_product_ids(params.get("product_ids")))
            elif item.get("name") == "budget_check":
                budget_ids.extend(parse_product_ids(params.get("product_ids")))
        if view_ids and budget_ids and set(view_ids) != set(budget_ids):
            return {
                "error": "view_and_budget_product_ids_must_match",
                "tool": "selection",
                "view_product_ids": view_ids,
                "budget_product_ids": budget_ids,
                "required_fix": "Use the exact same selected product ids in view_product_information and budget_check.",
            }
    if {"find_product", "recommend_product", "terminate"}.issubset(allowed_tools):
        if "find_product" in names and bool({"recommend_product", "terminate"} & names):
            return {
                "error": "mixed_decision_actions_not_allowed",
                "tool": "decision",
                "required_fix": "In DECISION, either call recommend_product plus terminate, or call only find_product.",
            }
        if "terminate" in names and "recommend_product" not in names and call.get("name") == "terminate":
            return {
                "error": "terminate_requires_recommend_product_in_decision",
                "tool": call.get("name"),
                "required_fix": "Call recommend_product and terminate in the same tool_call array.",
            }
        if "recommend_product" in names and "terminate" not in names and call.get("name") == "recommend_product":
            return {
                "error": "recommend_product_requires_terminate_in_decision",
                "tool": call.get("name"),
                "required_fix": "Call recommend_product and terminate in the same tool_call array.",
            }
        if call.get("name") == "recommend_product":
            recommended_list = parse_product_ids((call.get("parameters") or {}).get("product_ids"))
            recommended = set(recommended_list)
            expected = budget_product_ids(snapshot) or selected_product_ids(snapshot)
            if not recommended:
                return {
                    "error": "recommend_product_requires_product_ids",
                    "tool": "recommend_product",
                    "required_fix": 'Use parameters {"product_ids": "id1,id2"}.',
                }
            if len(recommended_list) != len(recommended):
                return {
                    "error": "recommend_product_duplicate_product_ids",
                    "tool": "recommend_product",
                    "recommended_product_ids": recommended_list,
                    "required_fix": "Recommend each verified product id exactly once.",
                }
            if expected and len(recommended) != len(expected):
                return {
                    "error": "recommend_product_product_count_mismatch",
                    "tool": "recommend_product",
                    "recommended_product_ids": sorted(recommended),
                    "expected_product_ids": sorted(expected),
                    "required_fix": "Recommend exactly the same number of product ids as the verified budget_check selection.",
                }
            if expected and recommended != expected:
                return {
                    "error": "recommend_product_ids_must_match_verified_selection",
                    "tool": "recommend_product",
                    "recommended_product_ids": sorted(recommended),
                    "expected_product_ids": sorted(expected),
                    "required_fix": "Recommend exactly the product ids from the verified selection.",
                }
            viewed = viewed_product_ids(snapshot)
            missing_view = sorted(product_id for product_id in recommended if product_id not in viewed)
            if missing_view:
                return {
                    "error": "recommend_requires_viewed_product_evidence",
                    "tool": "recommend_product",
                    "recommended_product_ids": sorted(recommended),
                    "missing_viewed_product_ids": missing_view,
                    "required_fix": "Call view_product_information for every recommended product id before recommending.",
                }
            if budget_agent_voucher_requires_single_shop(snapshot):
                shops = verified_shop_ids(snapshot, recommended)
                if len(shops) != 1:
                    return {
                        "error": "shop_voucher_recommend_requires_single_shop",
                        "tool": "recommend_product",
                        "recommended_product_ids": sorted(recommended),
                        "shop_ids": sorted(shops),
                        "required_fix": "For a shop voucher, recommend only products verified from one shop or search for a same-shop bundle.",
                    }
    if call.get("name") not in allowed_tools:
        return {
            "error": "tool_not_allowed_in_current_state",
            "tool": call.get("name"),
            "allowed_tools": sorted(allowed_tools),
            "required_fix": "Choose a tool from allowed_tools only.",
        }
    if call.get("name") == "find_product" and is_repeated_failed_search(call.get("parameters", {}), snapshot):
        return {
            "error": "repeated_failed_search_not_allowed",
            "tool": "find_product",
            "required_fix": "Change the search parameters before retrying.",
        }
    if call.get("name") == "find_product" and is_duplicate_find_product_in_turn(call, all_calls):
        return {
            "error": "duplicate_find_product_in_same_turn_not_allowed",
            "tool": "find_product",
            "required_fix": "Do not repeat the same find_product parameters within one tool_call array.",
        }
    if call.get("name") == "find_product" and is_repeated_search(call.get("parameters", {}), snapshot):
        return {
            "error": "repeated_search_not_allowed",
            "tool": "find_product",
            "required_fix": "Use a new q, page, shop_id, price, sort, or service value.",
        }
    return None


def execute_calls(message: Message, snapshot) -> list[dict]:
    obs = []
    calls = message.tool_call or []
    for call in calls:
        error = validation_error(call, calls, snapshot)
        if error:
            obs.append({"tool_call_id": call["tool_call_id"], "results": error})
            continue
        name = call["name"]
        if name == "budget_check":
            result = budget_check_response(call, snapshot)
        elif name == "terminate":
            status = (call.get("parameters") or {}).get("status")
            result = {"status": status, "_tool_success": status == "success"}
        elif name in toolmap:
            result = toolmap[name].execute(**(call.get("parameters") or {}))
        else:
            result = {"error": "unknown_tool", "tool": name}
        obs.append({"tool_call_id": call["tool_call_id"], "results": result})
    return obs


def format_error(content: str) -> str | None:
    if content.count("<think>") != 1 or content.count("</think>") != 1:
        return "exactly_one_think_block_required"
    if content.count("<tool_call>") != 1 or content.count("</tool_call>") != 1:
        return "exactly_one_tool_call_block_required"
    if content.find("</think>") > content.find("<tool_call>"):
        return "think_block_must_precede_tool_call_block"
    if not re.match(r"^\s*<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*$", content, flags=re.DOTALL):
        return "state_local_output_must_be_think_then_tool_call_only"
    think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if not think_match or not think_match.group(1).strip():
        return "think_block_must_be_non_empty"
    return None


def structured_reasoning_format_error(reasoning: str, content: str) -> str | None:
    if not reasoning or not reasoning.strip():
        return "reasoning_content_required"
    if not content:
        return "empty_model_content"
    if "<think>" in content or "</think>" in content:
        return "content_must_not_include_think_tags"
    if content.count("<tool_call>") != 1 or content.count("</tool_call>") != 1:
        return "exactly_one_tool_call_block_required"
    if not re.match(r"^\s*<tool_call>.*?</tool_call>\s*$", content, flags=re.DOTALL):
        return "structured_reasoning_output_must_be_tool_call_only"
    return None


def no_think_format_error(content: str) -> str | None:
    if not content:
        return "empty_model_content"
    if "<think>" in content or "</think>" in content:
        return "content_must_not_include_think_tags"
    if content.count("<tool_call>") != 1 or content.count("</tool_call>") != 1:
        return "exactly_one_tool_call_block_required"
    if not re.match(r"^\s*<tool_call>.*?</tool_call>\s*$", content, flags=re.DOTALL):
        return "structured_reasoning_output_must_be_tool_call_only"
    return None


def tool_call_json_error(content: str) -> str | None:
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
    if not match:
        return "exactly_one_tool_call_block_required"
    try:
        parsed = json.loads(match.group(1).strip())
    except Exception:
        return "tool_call_json_must_be_valid_array"
    if not isinstance(parsed, list):
        if isinstance(parsed, dict) and set(parsed.keys()) == {"name", "arguments"}:
            return "tool_call_must_use_parameters_array_not_arguments_object"
        if isinstance(parsed, dict):
            return "tool_call_must_be_json_array"
        return "tool_call_must_be_json_array"
    if not parsed:
        return "tool_call_array_must_be_non_empty"
    for item in parsed:
        if isinstance(item, dict) and "arguments" in item:
            return "tool_call_must_use_parameters_not_arguments"
        if not isinstance(item, dict) or set(item.keys()) != {"name", "parameters"}:
            return "tool_call_items_require_exactly_name_and_parameters"
        if not isinstance(item.get("name"), str) or not isinstance(item.get("parameters"), dict):
            return "tool_call_name_string_parameters_object_required"
    return None


def loose_tool_calls_from_content(content: str) -> list[dict]:
    if not isinstance(content, str) or "<tool_call>" not in content:
        return []
    tail = content.split("<tool_call>", 1)[1]
    parsed = extract_json_value(tail)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    calls = []
    used_ids = set()
    for item in parsed:
        if not isinstance(item, dict) or set(item.keys()) != {"name", "parameters"}:
            return []
        name = item.get("name")
        parameters = item.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            return []
        tool_call_id = generate_tool_call_id(name, parameters)
        suffix = 1
        unique_id = tool_call_id
        while unique_id in used_ids:
            suffix += 1
            unique_id = f"{tool_call_id}_{suffix}"
        used_ids.add(unique_id)
        calls.append({"name": name, "parameters": parameters, "tool_call_id": unique_id})
    return calls


def format_required_fix(error: str) -> str:
    fixes = {
        "empty_model_content": "Output a non-empty <think>...</think><tool_call>...</tool_call> message.",
        "reasoning_content_required": "Use the provider thinking channel for reasoning, then output only <tool_call>...</tool_call> in content.",
        "content_must_not_include_think_tags": "Do not put <think> tags in content when structured reasoning mode is enabled.",
        "structured_reasoning_output_must_be_tool_call_only": "Remove all analysis, markdown, and natural language. Output only one <tool_call>...</tool_call> block.",
        "exactly_one_think_block_required": "Output exactly one opening <think> and one closing </think> before <tool_call>.",
        "exactly_one_tool_call_block_required": "Output exactly one opening <tool_call> and one closing </tool_call>. Do not use XML tags like <find_product>, function-call syntax, markdown, or natural language outside the block.",
        "think_block_must_precede_tool_call_block": "Place <think>...</think> before <tool_call>...</tool_call>.",
        "state_local_output_must_be_think_then_tool_call_only": "Output only <think>...</think><tool_call>...</tool_call>, with no text outside the tags.",
        "think_block_must_be_non_empty": "Put concise reasoning inside <think>...</think>.",
        "tool_call_json_must_be_valid_array": 'Inside <tool_call> must be raw JSON array only, for example [{"name":"find_product","parameters":{"q":"...","page":1}}]. Do not use XML tags like <find_product> or <function=...>.',
        "tool_call_must_be_json_array": 'The <tool_call> content must be a JSON array [...] not a single object {...}. Wrap one call as [{"name":"allowed_tool","parameters":{}}].',
        "tool_call_must_use_parameters_array_not_arguments_object": 'Do not use Qwen/OpenAI native function-call shape. Use a JSON array and the key "parameters": [{"name":"allowed_tool","parameters":{}}].',
        "tool_call_must_use_parameters_not_arguments": 'Use the ShoppingBench key "parameters", not "arguments". Each item must be {"name":"allowed_tool","parameters":{}}.',
        "tool_call_array_must_be_non_empty": "Include at least one tool call object in the JSON array.",
        "tool_call_items_require_exactly_name_and_parameters": 'Each tool call object must contain exactly "name" and "parameters"; do not use "arguments" or "tool_call_id".',
        "tool_call_name_string_parameters_object_required": '"name" must be a string and "parameters" must be an object.',
        "no_valid_tool_call": "Use the exact tool_call schema from the prompt with allowed tool names; do not output OpenAI function-call syntax or XML tool tags.",
    }
    return fixes.get(error, "Fix the output format and retry.")


def format_required_format() -> str:
    return '<think>brief reasoning</think><tool_call>[{"name":"allowed_tool","parameters":{}}]</tool_call>'


def structured_reasoning_required_format() -> str:
    return '<tool_call>[{"name":"allowed_tool","parameters":{}}]</tool_call>'


def tool_call_only_visible_schema() -> str:
    return (
        "\nVisible output schema:\n"
        '<tool_call>[{"name":"allowed_tool","parameters":{}}]</tool_call>\n'
        "The content between <tool_call> tags must be one JSON array.\n"
        "Do not output markdown, analysis text, XML tool tags such as <find_product>, "
        'function-call syntax, "arguments", or a single JSON object.\n'
    )


def tool_call_only_system_prompt(system_prompt: str, *, provider_reasoning: bool) -> str:
    reasoning_sentence = (
        "Use the provider thinking channel for reasoning. In visible content, output exactly one "
        if provider_reasoning
        else "Do any private reasoning silently. In visible content, output exactly one "
    )
    replacements = {
        "Output exactly one `<think>...</think>` block followed by exactly one `<tool_call>...</tool_call>` block and nothing else.": (
            reasoning_sentence + "`<tool_call>...</tool_call>` block, including the closing `</tool_call>` tag, and nothing else."
        ),
        "Use `<think>` to briefly reason about the current state, product fit, voucher interpretation, and the next tool action. Keep it concise but concrete.\n": "",
        "If `last_errors` contains a format error, fix the output format on the next turn before doing anything else.": (
            "If `last_errors` contains a format error, fix the visible content format on the next turn before doing anything else."
        ),
    }
    for old, new in replacements.items():
        system_prompt = system_prompt.replace(old, new)
    return system_prompt.rstrip() + tool_call_only_visible_schema()


def structured_reasoning_system_prompt(system_prompt: str) -> str:
    return tool_call_only_system_prompt(system_prompt, provider_reasoning=True)


def is_successful_terminate(message: Message, obs: list[dict], snapshot) -> bool:
    if "terminate" not in {call.get("name") for call in message.tool_call or []}:
        return False
    if "recommend_product" not in {call.get("name") for call in message.tool_call or []}:
        return False
    if not {"find_product", "recommend_product", "terminate"}.issubset(snapshot.include_tools):
        return False
    for item in obs:
        result = item.get("results")
        if isinstance(result, dict) and result.get("error"):
            return False
    return True


def run_one(query: str, config: dict, system_prompt: str, trace_dir: Path) -> list[dict]:
    history_messages = [Message(user=query).to_string(USER_ROLES)]
    row = []
    for step in range(1, config["max_steps"] + 1):
        snapshot = state_local_snapshot(history_messages, config["max_failed_searches"])
        user_prompt = build_compact_harness_user_prompt(
            snapshot,
            history_messages,
            max_candidates=config["max_candidates"],
            max_failed_searches=config["max_failed_searches"],
            max_viewed_products=config["max_viewed_products"],
        )
        reasoning, content = ask_llm(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            config["model_config"],
            base_url=config["base_url"],
            api_key=config["api_key"],
        )
        message = Message.from_string(reasoning or "", content or "")
        loose_extracted = False
        if config.get("loose_tool_call_extraction") and not message.tool_call:
            loose_calls = loose_tool_calls_from_content(content or "")
            if loose_calls:
                message.tool_call = loose_calls
                message.format_error = ""
                loose_extracted = True
        if config.get("accept_structured_reasoning"):
            if not loose_extracted:
                message.format_error = (
                    structured_reasoning_format_error(reasoning or "", content or "")
                    or tool_call_json_error(content or "")
                    or message.format_error
                    or ""
                )
        elif config.get("no_think_output"):
            if not loose_extracted:
                message.format_error = (
                    no_think_format_error(content or "")
                    or tool_call_json_error(content or "")
                    or message.format_error
                    or ""
                )
        elif not content:
            message.format_error = "empty_model_content"
        else:
            message.format_error = format_error(content) or tool_call_json_error(content) or message.format_error or ""
        if message.format_error:
            required_format = (
                structured_reasoning_required_format()
                if config.get("accept_structured_reasoning") or config.get("no_think_output")
                else format_required_format()
            )
            message.obs = [
                {
                    "tool_call_id": "format_error",
                    "results": {
                        "tool": "format",
                        "error": message.format_error,
                        "required_fix": format_required_fix(message.format_error),
                        "required_format": required_format,
                    },
                }
            ]
        elif message.tool_call:
            message.obs = execute_calls(message, snapshot)
        else:
            message.obs = [
                {
                    "tool_call_id": "format_error",
                    "results": {
                        "tool": "format",
                        "error": "no_valid_tool_call",
                        "required_fix": format_required_fix("no_valid_tool_call"),
                        "required_format": (
                            structured_reasoning_required_format()
                            if config.get("accept_structured_reasoning") or config.get("no_think_output")
                            else format_required_format()
                        ),
                    },
                }
            ]

        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / (str(abs(hash(query))) + ".md")
        trace_file.write_text(search_trace_markdown(query, snapshot.search_trace), encoding="utf-8")
        row.append(
            {
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "completion": {
                    "reasoning_content": reasoning or "",
                    "content": content or "",
                    "message": copy.deepcopy(message.to_dict()),
                },
                "extra_info": {
                    "step": step,
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "history_compression": "state_local",
                    "accept_structured_reasoning": bool(config.get("accept_structured_reasoning")),
                    "no_think_output": bool(config.get("no_think_output")),
                    "loose_tool_call_extraction": loose_extracted,
                    "harness_state": snapshot.state_name,
                    "allowed_tools": sorted(snapshot.include_tools),
                    "harness_search_trace_file": str(trace_file),
                },
            }
        )
        history_messages.append(message.to_string(ASSISTANT_ROLES))
        if is_successful_terminate(message, message.obs, snapshot):
            break
    return row


def audit(rows: list[list[dict]]) -> dict:
    states = Counter()
    tools = Counter()
    errors = Counter()
    terminated = 0
    steps = []
    for row in rows:
        steps.append(len(row))
        last = row[-1]["completion"]["message"] if row else {}
        if any(call.get("name") == "terminate" for call in last.get("tool_call", []) or []):
            terminated += 1
        for step in row:
            states[step["extra_info"].get("harness_state")] += 1
            for call in step["completion"]["message"].get("tool_call", []) or []:
                tools[call.get("name")] += 1
            for obs in step["completion"]["message"].get("obs", []) or []:
                result = obs.get("results")
                if isinstance(result, dict) and result.get("error"):
                    errors[result.get("error")] += 1
    return {
        "rows": len(rows),
        "terminated": terminated,
        "steps": steps,
        "state_counts": dict(states),
        "tool_counts": dict(tools),
        "error_counts": dict(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/synthesize_voucher_train.jsonl")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--rollout-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--model", default="gpt-5.5-medium")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-failed-searches", type=int, default=5)
    parser.add_argument("--max-viewed-products", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1, help="Number of query rollouts to run concurrently.")
    parser.add_argument(
        "--accept-structured-reasoning",
        action="store_true",
        help="Test-only mode: use API reasoning_content as think and require only <tool_call> in content.",
    )
    parser.add_argument(
        "--no-think-output",
        action="store_true",
        help="Require only <tool_call> in visible content; use with provider thinking disabled.",
    )
    parser.add_argument(
        "--thinking-type",
        choices=["default", "enabled", "disabled"],
        default="default",
        help="Optional provider thinking.type value for API probes/rollouts.",
    )
    parser.add_argument(
        "--loose-tool-call-extraction",
        action="store_true",
        help="Test-only mode: extract a valid JSON tool call from malformed visible content.",
    )
    args = parser.parse_args()

    source_rows = read_jsonl(ROOT / args.source)
    sample_path = ROOT / args.sample_file
    if sample_path.exists():
        sample_rows = read_jsonl(sample_path)
    else:
        sample_rows = select_broad8(source_rows)
        write_jsonl(sample_path, sample_rows)

    system_prompt = build_system_prompt("src/agent/prompt/rollout.state_local.md")
    if args.accept_structured_reasoning and args.no_think_output:
        raise ValueError("--accept-structured-reasoning and --no-think-output are mutually exclusive")
    if args.accept_structured_reasoning:
        system_prompt = structured_reasoning_system_prompt(system_prompt)
    elif args.no_think_output:
        system_prompt = tool_call_only_system_prompt(system_prompt, provider_reasoning=False)
    model_config = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
    }
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.thinking_type != "default":
        extra_body["thinking"] = {"type": args.thinking_type}
    if extra_body:
        model_config["extra_body"] = extra_body
    config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_steps": args.max_steps,
        "max_candidates": args.max_candidates,
        "max_failed_searches": args.max_failed_searches,
        "max_viewed_products": args.max_viewed_products,
        "accept_structured_reasoning": args.accept_structured_reasoning,
        "no_think_output": args.no_think_output,
        "loose_tool_call_extraction": args.loose_tool_call_extraction,
        "model_config": model_config,
    }
    rollout_path = ROOT / args.rollout_file
    trace_dir = rollout_path.with_suffix("")
    rows = [None] * len(sample_rows)
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, item in enumerate(sample_rows, 1):
            print(f"[{idx}/{len(sample_rows)}] rolling out", flush=True)
            rows[idx - 1] = run_one(item["query"], config, system_prompt, trace_dir)
            write_jsonl(rollout_path, rows)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, item["query"], config, system_prompt, trace_dir): idx
                for idx, item in enumerate(sample_rows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                print(f"[{idx + 1}/{len(sample_rows)}] completed", flush=True)
                rows[idx] = future.result()
                write_jsonl(rollout_path, rows)
    rows = [row for row in rows if row is not None]

    report = {
        "sample_file": args.sample_file,
        "rollout_file": args.rollout_file,
        "model": args.model,
        "accept_structured_reasoning": args.accept_structured_reasoning,
        "no_think_output": args.no_think_output,
        "thinking_type": args.thinking_type,
        "loose_tool_call_extraction": args.loose_tool_call_extraction,
        "workers": workers,
        "audit": audit(rows),
        "sample_distribution": {
            "product_count": dict(Counter(str(len(row.get("reward") or [])) for row in sample_rows)),
            "voucher_type": dict(Counter((row.get("voucher") or {}).get("voucher_type") for row in sample_rows)),
            "discount_type": dict(Counter((row.get("voucher") or {}).get("discount_type") for row in sample_rows)),
        },
    }
    report_path = ROOT / args.report_file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
