import re
from dataclasses import dataclass
from pathlib import Path

try:
    import ujson as json
except ModuleNotFoundError:
    import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STATE_CANDIDATE_SEARCH = "CANDIDATE_SEARCH"
STATE_CANDIDATE_SELECT = "CANDIDATE_SELECT"
STATE_DECISION = "DECISION"

DEFAULT_PROMPT_FILES = {
    STATE_CANDIDATE_SEARCH: "src/agent/prompt/prompt_SEARCH.md",
    STATE_CANDIDATE_SELECT: "src/agent/prompt/prompt_SELECT.md",
    STATE_DECISION: "src/agent/prompt/prompt_DECISION.md",
}

STATE_TOOLS = {
    STATE_CANDIDATE_SEARCH: {"find_product"},
    STATE_CANDIDATE_SELECT: {"view_product_information", "python_execute"},
    STATE_DECISION: {"find_product", "recommend_product", "terminate"},
}

SEARCH_PARAM_KEYS = ("q", "page", "shop_id", "price", "sort", "service")
ROLE_RE_TEMPLATE = r"<{role}>(.*?)</{role}>"
USER_RE = re.compile(r"<user>(.*?)</user>", re.DOTALL)


@dataclass
class HarnessSnapshot:
    state_name: str
    state: dict
    prompt_file: str
    include_tools: set[str]
    search_trace: list[dict]


def _json_role_value(text: str, role: str):
    match = re.search(ROLE_RE_TEMPLATE.format(role=role), text, re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _first_user_message(history_messages: list[str]) -> str:
    for item in history_messages:
        match = USER_RE.search(item)
        if match:
            return match.group(1).strip()
    return ""


def _sparse_search_parameters(parameters: dict) -> dict:
    if not isinstance(parameters, dict):
        return {}
    result = {}
    for key in SEARCH_PARAM_KEYS:
        value = parameters.get(key)
        if value in (None, "", "default"):
            continue
        result[key] = value
    return result


def _search_signature(parameters: dict) -> str:
    sparse = _sparse_search_parameters(parameters)
    normalized = {
        key: str(value).strip().lower()
        for key, value in sparse.items()
        if value not in (None, "")
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_searches(searches: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in searches:
        signature = _search_signature(item)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        result.append(_sparse_search_parameters(item))
    return result


def is_repeated_failed_search(parameters: dict, snapshot: HarnessSnapshot) -> bool:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    failed_searches = []
    if snapshot.state_name == STATE_CANDIDATE_SEARCH:
        failed_searches = state.get("failed_searches") or []
    elif snapshot.state_name == STATE_DECISION:
        failed_searches = state.get("failed_retry_searches") or []
    failed_signatures = {_search_signature(item) for item in failed_searches if isinstance(item, dict)}
    signature = _search_signature(parameters)
    return bool(signature and signature in failed_signatures)


def _slim_candidate(product: dict) -> dict:
    return {
        key: product[key]
        for key in ("product_id", "shop_id", "title", "price", "service", "sold_count")
        if isinstance(product, dict) and key in product
    }


def _slim_viewed_product(product: dict) -> dict:
    if not isinstance(product, dict):
        return {}
    result = {"product_id": product.get("product_id")}
    for key in (
        "title",
        "description",
        "short_description",
        "product_description",
        "sku_options",
        "attributes",
        "service",
    ):
        if key in product:
            result[key] = product[key]
    return result


def _parse_product_ids(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_python_result(results):
    if not isinstance(results, dict):
        return None
    text = results.get("observation")
    if text is None:
        text = results.get("stdout")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed["_tool_success"] = results.get("success")
            parsed["_parse_ok"] = True
            return parsed
        return {
            "observation": text,
            "_tool_success": results.get("success"),
            "_parse_ok": False,
        }
    except Exception:
        return {
            "observation": text,
            "_tool_success": results.get("success"),
            "_parse_ok": False,
        }


def _parse_budget_result(record: dict, *, allow_legacy_python_budget: bool = True):
    if not isinstance(record, dict):
        return None
    if record.get("name") == "python_execute":
        if not allow_legacy_python_budget:
            return None
        return _parse_python_result(record.get("results"))
    if record.get("name") != "budget_check":
        return None
    results = record.get("results")
    if not isinstance(results, dict):
        return None
    parsed = dict(results)
    parsed.setdefault("_tool_success", not bool(parsed.get("error")))
    parsed.setdefault("_parse_ok", "error" not in parsed)
    return parsed


def _nonempty_find_results(results) -> bool:
    return isinstance(results, list) and len(results) > 0


def _empty_find_results(results) -> bool:
    return isinstance(results, list) and len(results) == 0


def _build_turns(history_messages: list[str]) -> list[dict]:
    turns = []
    for item in history_messages:
        calls = _json_role_value(item, "tool_call")
        obs = _json_role_value(item, "obs")
        if not calls:
            continue
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            continue
        obs = obs if isinstance(obs, list) else []
        obs_by_id = {
            observation.get("tool_call_id"): observation
            for observation in obs
            if isinstance(observation, dict)
        }
        records = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            observation = obs_by_id.get(call.get("tool_call_id"), {})
            records.append(
                {
                    "name": call.get("name"),
                    "parameters": call.get("parameters", {}) or {},
                    "results": observation.get("results"),
                }
            )
        if records:
            turns.append({"records": records})
    return turns


def _view_requested_ids(record: dict) -> list[str]:
    return _parse_product_ids(record.get("parameters", {}).get("product_ids"))


def _budget_product_ids(parsed_budget: dict | None) -> list[str]:
    if not isinstance(parsed_budget, dict):
        return []
    product_ids = parsed_budget.get("product_ids")
    if not isinstance(product_ids, list):
        return []
    return [str(product_id) for product_id in product_ids if product_id is not None]


def _product_ids_from_products(products) -> set[str]:
    if not isinstance(products, list):
        return set()
    return {
        str(product.get("product_id"))
        for product in products
        if isinstance(product, dict) and product.get("product_id") is not None
    }


def _valid_budget_calculation(parsed_budget: dict | None) -> bool:
    if not isinstance(parsed_budget, dict):
        return False
    if parsed_budget.get("_parse_ok") is not True or parsed_budget.get("_tool_success") is False:
        return False
    product_ids = _budget_product_ids(parsed_budget)
    shop_ids = parsed_budget.get("shop_ids")
    if not product_ids or not isinstance(shop_ids, list) or len(shop_ids) != len(product_ids):
        return False
    for key in ("total_before_voucher", "payable_total", "budget"):
        try:
            float(parsed_budget[key])
        except Exception:
            return False
    if not isinstance(parsed_budget.get("within_budget"), bool):
        return False
    if not isinstance(parsed_budget.get("voucher_used"), bool):
        return False
    return True


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _budget_matches_candidate_evidence(parsed_budget: dict | None, products_by_id: dict[str, dict]) -> bool:
    if not _valid_budget_calculation(parsed_budget):
        return False
    product_ids = _budget_product_ids(parsed_budget)
    if len(product_ids) != len(set(product_ids)):
        return False
    shop_ids = [str(shop_id) for shop_id in parsed_budget.get("shop_ids", [])]
    products = []
    for product_id, shop_id in zip(product_ids, shop_ids, strict=False):
        product = products_by_id.get(str(product_id))
        if product is None or product.get("price") is None or product.get("shop_id") is None:
            return False
        if str(product["shop_id"]) != shop_id:
            return False
        price = _to_float(product.get("price"))
        if price is None:
            return False
        products.append(product)
    expected_total = round(sum(float(product["price"]) for product in products), 2)
    reported_total = _to_float(parsed_budget.get("total_before_voucher"))
    payable_total = _to_float(parsed_budget.get("payable_total"))
    budget = _to_float(parsed_budget.get("budget"))
    if reported_total is None or abs(reported_total - expected_total) > 0.01:
        return False
    if payable_total is None or budget is None:
        return False
    if parsed_budget.get("within_budget") != (payable_total <= budget):
        return False
    return True


def _ids_consistent(view_ids: list[str], budget_ids: list[str]) -> bool:
    return (
        bool(budget_ids)
        and len(view_ids) == len(set(view_ids))
        and len(budget_ids) == len(set(budget_ids))
        and set(view_ids) == set(budget_ids)
    )


def _combined_view_requested_ids(records: list[dict]) -> list[str]:
    selected = []
    for record in records:
        for product_id in _view_requested_ids(record):
            if product_id not in selected:
                selected.append(product_id)
    return selected


def _combined_view_results(records: list[dict]) -> list[dict]:
    products = []
    seen_ids = set()
    for record in records:
        results = record.get("results")
        if not isinstance(results, list):
            continue
        for product in results:
            if not isinstance(product, dict) or product.get("product_id") is None:
                continue
            product_id = str(product["product_id"])
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            products.append(product)
    return products


def _valid_view_record(record: dict) -> bool:
    requested_ids = _view_requested_ids(record)
    returned_ids = _product_ids_from_products(record.get("results"))
    return (
        record.get("name") == "view_product_information"
        and bool(requested_ids)
        and isinstance(record.get("results"), list)
        and all(product_id in returned_ids for product_id in requested_ids)
    )


def _valid_budget_record(record: dict, *, allow_legacy_python_budget: bool = True) -> bool:
    parsed = _parse_budget_result(record, allow_legacy_python_budget=allow_legacy_python_budget)
    return _valid_budget_calculation(parsed)


def _selected_ids_from_turn(turn: dict) -> list[str]:
    selected = []
    for record in turn.get("records", []):
        if record.get("name") != "view_product_information":
            continue
        for product_id in _view_requested_ids(record):
            if product_id not in selected:
                selected.append(product_id)
    return selected


def _collect_products_by_id(turns: list[dict]) -> dict[str, dict]:
    products = {}
    for turn in turns:
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            results = record.get("results")
            if not isinstance(results, list):
                continue
            for product in results:
                if not isinstance(product, dict) or product.get("product_id") is None:
                    continue
                product_id = str(product["product_id"])
                products.setdefault(product_id, _slim_candidate(product))
    return products


def _candidate_pool_from_records(records: list[dict], seed_products: list[dict] | None = None) -> list[dict]:
    candidates = []
    seen_ids = set()
    for product in seed_products or []:
        if not isinstance(product, dict) or product.get("product_id") is None:
            continue
        product_id = str(product["product_id"])
        if product_id in seen_ids:
            continue
        seen_ids.add(product_id)
        candidates.append(_slim_candidate(product))
    for record in records:
        if record.get("name") != "find_product":
            continue
        results = record.get("results")
        if not isinstance(results, list):
            continue
        for product in results:
            if not isinstance(product, dict) or product.get("product_id") is None:
                continue
            product_id = str(product["product_id"])
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            candidates.append(_slim_candidate(product))
    return candidates


def _candidate_pool_from_turns(
    turns: list[dict],
    start_index: int = 0,
    seed_products: list[dict] | None = None,
) -> list[dict]:
    records = [
        record
        for turn in turns[start_index:]
        for record in turn.get("records", [])
    ]
    return _candidate_pool_from_records(records, seed_products=seed_products)


def _candidate_ids_from_turns(turns: list[dict]) -> set[str]:
    return {
        str(candidate.get("product_id"))
        for candidate in _candidate_pool_from_turns(turns)
        if candidate.get("product_id") is not None
    }


def _active_candidate_ids_for_check(turns: list[dict], checks: list[dict], current_idx: int) -> set[str]:
    if not checks:
        return _candidate_ids_from_turns(turns[: current_idx + 1])
    previous_check = checks[-1]
    active_ids = set(previous_check.get("view_requested_product_ids") or [])
    retry_turns = turns[previous_check["end_turn_index"] + 1 : current_idx + 1]
    active_ids.update(_candidate_ids_from_turns(retry_turns))
    return {str(product_id) for product_id in active_ids if product_id is not None}


def _valid_checks_from_turns(turns: list[dict], *, allow_legacy_python_budget: bool = True) -> list[dict]:
    checks = []
    latest_nonempty_find_idx = -1
    accumulated_view_records = []

    for idx, turn in enumerate(turns):
        if any(
            record.get("name") == "find_product"
            and _nonempty_find_results(record.get("results"))
            for record in turn.get("records", [])
        ):
            latest_nonempty_find_idx = idx
            accumulated_view_records = []

        turn_view_records = [
            {"turn_index": idx, "record": record}
            for record in turn.get("records", [])
            if _valid_view_record(record)
        ]
        if turn_view_records:
            accumulated_view_records.extend(turn_view_records)

        for record in turn.get("records", []):
            if not _valid_budget_record(record, allow_legacy_python_budget=allow_legacy_python_budget) or not accumulated_view_records:
                continue
            if max(item["turn_index"] for item in accumulated_view_records) <= latest_nonempty_find_idx:
                continue
            parsed_budget = _parse_budget_result(record, allow_legacy_python_budget=allow_legacy_python_budget)
            candidate_ids = _active_candidate_ids_for_check(turns, checks, idx)
            products_by_id = {
                product_id: product
                for product_id, product in _collect_products_by_id(turns[: idx + 1]).items()
                if product_id in candidate_ids
            }
            if not _budget_matches_candidate_evidence(parsed_budget, products_by_id):
                continue
            budget_ids = _budget_product_ids(parsed_budget)
            candidate_view_sets = []
            if turn_view_records:
                candidate_view_sets.append([item["record"] for item in turn_view_records])
            candidate_view_sets.append([item["record"] for item in accumulated_view_records])
            matched_view_records = None
            matched_view_ids = []
            for view_records in candidate_view_sets:
                view_ids = _combined_view_requested_ids(view_records)
                if _ids_consistent(view_ids, budget_ids) and set(view_ids).issubset(candidate_ids):
                    matched_view_records = view_records
                    matched_view_ids = view_ids
                    break
            if matched_view_records is None:
                continue
            checks.append(
                {
                    "view_turn_index": max(
                        item["turn_index"]
                        for item in accumulated_view_records
                        if item["record"] in matched_view_records
                    ),
                    "python_turn_index": idx,
                    "end_turn_index": idx,
                    "view_records": matched_view_records,
                    "python_record": record,
                    "view_requested_product_ids": matched_view_ids,
                    "budget_product_ids": budget_ids,
                }
            )

    return checks


def _find_latest_valid_check(turns: list[dict], *, allow_legacy_python_budget: bool = True) -> dict | None:
    checks = _valid_checks_from_turns(turns, allow_legacy_python_budget=allow_legacy_python_budget)
    return checks[-1] if checks else None


def _failed_retry_records_from_turns(turns: list[dict], checks: list[dict]) -> list[dict]:
    failed = []
    check_end_indices = [check["end_turn_index"] for check in checks]
    for idx, turn in enumerate(turns):
        if not any(check_idx < idx for check_idx in check_end_indices):
            continue
        for record in turn.get("records", []):
            if record.get("name") == "find_product" and _empty_find_results(record.get("results")):
                failed.append(record.get("parameters", {}))
    return failed


def _empty_find_records_from_turns(turns: list[dict]) -> list[dict]:
    return [
        record.get("parameters", {})
        for turn in turns
        for record in turn.get("records", [])
        if record.get("name") == "find_product" and _empty_find_results(record.get("results"))
    ]


def _decision_state_from_check(
    turns: list[dict],
    check: dict,
    failed_retry_records: list[dict],
    *,
    allow_legacy_python_budget: bool = True,
) -> dict:
    selected_ids = check["view_requested_product_ids"]
    products_by_id = _collect_products_by_id(turns[: check["end_turn_index"] + 1])
    selected_products = [
        products_by_id[product_id]
        for product_id in selected_ids
        if product_id in products_by_id
    ]

    viewed_by_id = {}
    for product in _combined_view_results(check["view_records"]):
        viewed = _slim_viewed_product(product)
        product_id = viewed.get("product_id")
        if product_id is not None:
            viewed_by_id[str(product_id)] = viewed

    budget_calculation = _parse_budget_result(
        check["python_record"],
        allow_legacy_python_budget=allow_legacy_python_budget,
    ) or {}
    selected_shop_ids = sorted(
        {
            str(product.get("shop_id"))
            for product in selected_products
            if product.get("shop_id") is not None
        }
    )

    return {
        "selected_products": selected_products,
        "selected_shop_ids": selected_shop_ids,
        "view_requested_product_ids": check["view_requested_product_ids"],
        "viewed_products": list(viewed_by_id.values()),
        "budget_product_ids": check["budget_product_ids"],
        "budget_calculation": budget_calculation or {},
        "selection_consistency": True,
        "failed_retry_searches": _unique_searches(failed_retry_records),
    }


def _candidate_pool_from_turn(turn: dict) -> list[dict]:
    return _candidate_pool_from_records(turn.get("records", []))


def _search_trace_from_turns(turns: list[dict]) -> list[dict]:
    trace = []
    current_phase = STATE_CANDIDATE_SEARCH
    for turn_no, turn in enumerate(turns, 1):
        names = {record.get("name") for record in turn.get("records", [])}
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            results = record.get("results")
            result_count = len(results) if isinstance(results, list) else None
            trace.append(
                {
                    "turn": turn_no,
                    "phase": current_phase,
                    "parameters": _sparse_search_parameters(record.get("parameters", {})),
                    "result_count": result_count,
                    "empty": _empty_find_results(results),
                }
            )
        if "view_product_information" in names and bool({"python_execute", "budget_check"} & names):
            current_phase = STATE_DECISION
        elif any(
            record.get("name") == "find_product" and _nonempty_find_results(record.get("results"))
            for record in turn.get("records", [])
        ):
            current_phase = STATE_CANDIDATE_SELECT
    return trace


def build_harness_snapshot(
    history_messages: list[str],
    prompt_files: dict[str, str] | None = None,
    *,
    allow_legacy_python_budget: bool = True,
) -> HarnessSnapshot:
    prompt_files = {**DEFAULT_PROMPT_FILES, **(prompt_files or {})}
    turns = _build_turns(history_messages)
    search_trace = _search_trace_from_turns(turns)

    if not turns:
        state_name = STATE_CANDIDATE_SEARCH
        state = {}
        return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)

    valid_checks = _valid_checks_from_turns(turns, allow_legacy_python_budget=allow_legacy_python_budget)
    latest_check = valid_checks[-1] if valid_checks else None
    latest_check_end_idx = latest_check["end_turn_index"] if latest_check else None
    latest_nonempty_find_idx = None
    first_nonempty_find_seen = False
    cold_failed_searches = []
    all_empty_searches = _empty_find_records_from_turns(turns)

    for idx, turn in enumerate(turns):
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            if _nonempty_find_results(record.get("results")):
                latest_nonempty_find_idx = idx
                first_nonempty_find_seen = True
            elif _empty_find_results(record.get("results")) and not first_nonempty_find_seen:
                cold_failed_searches.append(record.get("parameters", {}))

    if latest_check is None:
        if latest_nonempty_find_idx is None:
            state_name = STATE_CANDIDATE_SEARCH
            state = {}
            failed_searches = _unique_searches(all_empty_searches)
            if failed_searches:
                state["failed_searches"] = failed_searches
        else:
            state_name = STATE_CANDIDATE_SELECT
            state = {"candidate_pool": _candidate_pool_from_turns(turns)}
        return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)

    retry_find_indices = [
        idx
        for idx in range(latest_check_end_idx + 1, len(turns))
        if any(record.get("name") == "find_product" for record in turns[idx].get("records", []))
    ]
    latest_retry_nonempty_idx = None
    failed_retry_records = all_empty_searches
    for idx in retry_find_indices:
        turn = turns[idx]
        turn_has_nonempty = False
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            if _nonempty_find_results(record.get("results")):
                turn_has_nonempty = True
        if turn_has_nonempty:
            latest_retry_nonempty_idx = idx

    if latest_retry_nonempty_idx is not None:
        state_name = STATE_CANDIDATE_SELECT
        previous_decision_state = _decision_state_from_check(
            turns,
            latest_check,
            failed_retry_records,
            allow_legacy_python_budget=allow_legacy_python_budget,
        )
        state = {
            "candidate_pool": _candidate_pool_from_turns(
                turns,
                start_index=latest_check_end_idx + 1,
                seed_products=previous_decision_state["selected_products"],
            ),
            "previous_decision": {
                "selected_products": previous_decision_state["selected_products"],
                "viewed_products": previous_decision_state["viewed_products"],
                "budget_calculation": previous_decision_state["budget_calculation"],
            },
        }
    else:
        state_name = STATE_DECISION
        state = _decision_state_from_check(
            turns,
            latest_check,
            failed_retry_records,
            allow_legacy_python_budget=allow_legacy_python_budget,
        )
    return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)


def build_harness_user_prompt(snapshot: HarnessSnapshot, history_messages: list[str]) -> str:
    query = _first_user_message(history_messages)
    state_text = json.dumps(snapshot.state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "# Dialogue Records History\n" + f"<user>{query}</user>\n\n<state>{state_text}</state>"


def build_harness_user_prompt_with_instructions(
    snapshot: HarnessSnapshot,
    history_messages: list[str],
) -> str:
    from util.system_prompt import build_system_prompt

    instructions = build_system_prompt(
        snapshot.prompt_file,
        include_tools=snapshot.include_tools,
    )
    return (
        "# Harness State Instructions\n"
        + instructions
        + "\n\n"
        + build_harness_user_prompt(snapshot, history_messages)
    )


def _brief_state_instructions(snapshot: HarnessSnapshot) -> str:
    tools = ", ".join(sorted(snapshot.include_tools))
    if snapshot.state_name == STATE_CANDIDATE_SEARCH:
        rules = [
            "Use only find_product.",
            'find_product parameters: {"q": string, "page": integer}; optional keys include shop_id, price, sort, and service.',
            "Choose queries from the user request and cover all requested product needs.",
            "Do not repeat exact parameters listed in failed_searches.",
            "Multiple independent calls may be placed in one JSON array.",
        ]
    elif snapshot.state_name == STATE_CANDIDATE_SELECT:
        rules = [
            "Use only view_product_information and python_execute.",
            'view_product_information parameters: {"product_ids": "id1,id2"}.',
            'python_execute parameters: {"code": string}. The code must print one JSON object.',
            "Choose product ids only from candidate_pool.",
            "Use python_execute to calculate voucher eligibility, payable_total, budget, and within_budget from the user's query, using selected ids, shop ids, and prices from candidate_pool.",
            "The printed JSON must include product_ids, shop_ids, total_before_voucher, payable_total, budget, within_budget, and voucher_used.",
            "The budget JSON must match candidate_pool prices and shop ids.",
        ]
    else:
        rules = [
            "Use only find_product, recommend_product, and terminate.",
            'find_product parameters: {"q": string, "page": integer}; optional keys include shop_id, price, sort, and service.',
            'recommend_product parameters: {"product_ids": "id1,id2"}.',
            'terminate parameters: {"status": "success"}.',
            "If the selected products satisfy the user request, the voucher/budget calculation is correct for the user's query, and within_budget is true, output recommend_product and terminate in the same JSON array.",
            "If voucher eligibility, payable_total, budget, or within_budget is invalid or inconsistent with the user's query, output only find_product calls.",
            "Otherwise output only find_product calls for replacement candidates.",
            "Do not repeat exact parameters listed in failed_retry_searches.",
        ]
    return "\n".join(
        [
            f"Current state: {snapshot.state_name}",
            f"Allowed tools: {tools}",
            "Use only the latest <state> block in this message for state decisions.",
            "Output exactly one <think> block followed by exactly one <tool_call> block.",
            "The <tool_call> block must contain one JSON array of calls, each with name and parameters.",
            "Rules:",
            *[f"- {rule}" for rule in rules],
        ]
    )


def build_harness_user_prompt_with_brief_instructions(
    snapshot: HarnessSnapshot,
    history_messages: list[str],
) -> str:
    return (
        "# Harness State Instructions\n"
        + _brief_state_instructions(snapshot)
        + "\n\n"
        + build_harness_user_prompt(snapshot, history_messages)
    )


def _clip_text(value, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _compact_candidate_row(product: dict, title_chars: int = 80) -> list:
    return [
        str(product.get("product_id", "")),
        str(product.get("shop_id", "")),
        _clip_text(product.get("title", ""), title_chars),
        product.get("price"),
        product.get("service", []),
        product.get("sold_count"),
    ]


def _compact_selected_row(product: dict, title_chars: int = 80) -> list:
    return [
        str(product.get("product_id", "")),
        str(product.get("shop_id", "")),
        _clip_text(product.get("title", ""), title_chars),
        product.get("price"),
    ]


def _compact_viewed_row(product: dict, detail_chars: int = 220) -> list:
    detail_parts = []
    for key in ("description", "short_description", "product_description", "sku_options", "attributes", "service"):
        if key in product and product[key] not in (None, "", [], {}):
            detail_parts.append(f"{key}={product[key]}")
    return [
        str(product.get("product_id", "")),
        _clip_text(product.get("title", ""), 80),
        _clip_text("; ".join(detail_parts), detail_chars),
    ]


def _compact_budget_result(value: dict | None) -> dict:
    if not isinstance(value, dict):
        return {}
    keep = (
        "product_ids",
        "shop_ids",
        "total_before_voucher",
        "voucher_used",
        "payable_total",
        "budget",
        "within_budget",
        "agent_voucher",
    )
    return {key: value[key] for key in keep if key in value}


def _compact_parameters(parameters: dict) -> dict:
    if not isinstance(parameters, dict):
        return {}
    result = {}
    for key, value in parameters.items():
        if key == "code":
            continue
        if isinstance(value, str):
            result[key] = _clip_text(value, 120)
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, list):
            result[key] = value[:10]
        elif isinstance(value, dict):
            result[key] = {
                str(sub_key): (_clip_text(sub_value, 80) if isinstance(sub_value, str) else sub_value)
                for sub_key, sub_value in list(value.items())[:10]
            }
    return result


def _latest_structured_errors(history_messages: list[str], max_errors: int = 3) -> list[dict]:
    for item in reversed(history_messages):
        calls = _json_role_value(item, "tool_call")
        obs = _json_role_value(item, "obs")
        if not calls or not obs:
            continue
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list) or not isinstance(obs, list):
            continue
        calls_by_id = {
            call.get("tool_call_id"): call
            for call in calls
            if isinstance(call, dict)
        }
        errors = []
        for observation in obs:
            if not isinstance(observation, dict):
                continue
            result = observation.get("results")
            if not isinstance(result, dict):
                continue
            error = result.get("error")
            if not error and result.get("_tool_success") is not False:
                continue
            call = calls_by_id.get(observation.get("tool_call_id"), {})
            errors.append(
                {
                    "tool": call.get("name"),
                    "error": _clip_text(error or "tool_success_false", 120),
                    "parameters": _compact_parameters(call.get("parameters", {})),
                }
            )
        return errors[-max_errors:] if errors else []
    return []


def _compact_state_payload(
    snapshot: HarnessSnapshot,
    *,
    max_candidates: int | None = None,
    max_failed_searches: int | None = 5,
    max_viewed_products: int | None = None,
) -> dict:
    state = snapshot.state if isinstance(snapshot.state, dict) else {}
    payload = {
        "state": snapshot.state_name,
        "allowed_tools": sorted(snapshot.include_tools),
    }
    if snapshot.state_name == STATE_CANDIDATE_SEARCH:
        failed = state.get("failed_searches") or []
        payload["failed_searches"] = failed[-max_failed_searches:] if max_failed_searches else failed
    elif snapshot.state_name == STATE_CANDIDATE_SELECT:
        candidates = state.get("candidate_pool") or []
        if max_candidates:
            candidates = candidates[:max_candidates]
        payload["candidate_pool"] = [_compact_candidate_row(item) for item in candidates if isinstance(item, dict)]
        previous = state.get("previous_decision")
        if isinstance(previous, dict):
            viewed = previous.get("viewed_products") or []
            if max_viewed_products:
                viewed = viewed[:max_viewed_products]
            payload["previous_decision"] = {
                "selected_products": [
                    _compact_selected_row(item)
                    for item in previous.get("selected_products", [])
                    if isinstance(item, dict)
                ],
                "viewed_products": [_compact_viewed_row(item) for item in viewed if isinstance(item, dict)],
                "budget_result": _compact_budget_result(previous.get("budget_calculation")),
            }
    else:
        viewed = state.get("viewed_products") or []
        if max_viewed_products:
            viewed = viewed[:max_viewed_products]
        failed = state.get("failed_retry_searches") or []
        payload.update(
            {
                "selected_products": [
                    _compact_selected_row(item)
                    for item in state.get("selected_products", [])
                    if isinstance(item, dict)
                ],
                "viewed_products": [_compact_viewed_row(item) for item in viewed if isinstance(item, dict)],
                "budget_result": _compact_budget_result(state.get("budget_calculation")),
                "failed_retry_searches": failed[-max_failed_searches:] if max_failed_searches else failed,
            }
        )
    return payload


def build_compact_harness_user_prompt(
    snapshot: HarnessSnapshot,
    history_messages: list[str],
    *,
    max_candidates: int | None = None,
    max_failed_searches: int | None = 5,
    max_viewed_products: int | None = None,
) -> str:
    payload = _compact_state_payload(
        snapshot,
        max_candidates=max_candidates,
        max_failed_searches=max_failed_searches,
        max_viewed_products=max_viewed_products,
    )
    payload["query"] = _first_user_message(history_messages)
    latest_errors = _latest_structured_errors(history_messages)
    if latest_errors:
        payload["last_errors"] = latest_errors
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<state>{text}</state>"


def search_trace_markdown(query: str, search_trace: list[dict]) -> str:
    lines = ["## Query", "", query.strip(), "", "## Search Trace", ""]
    if not search_trace:
        lines.append("_No search attempts recorded._")
        return "\n".join(lines).strip() + "\n"
    for item in search_trace:
        params = json.dumps(item.get("parameters", {}), ensure_ascii=False, sort_keys=True)
        lines.append(
            f"- turn={item.get('turn')} phase={item.get('phase')} empty={item.get('empty')} "
            f"result_count={item.get('result_count')} params=`{params}`"
        )
    return "\n".join(lines).strip() + "\n"
