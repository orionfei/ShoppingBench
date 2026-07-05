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


def _ids_consistent(view_ids: list[str], budget_ids: list[str]) -> bool:
    return bool(budget_ids) and set(view_ids) == set(budget_ids)


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


def _valid_python_record(record: dict) -> bool:
    parsed = _parse_python_result(record.get("results"))
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


def _valid_checks_from_turns(turns: list[dict]) -> list[dict]:
    checks = []
    latest_nonempty_find_idx = -1
    latest_view_records = []

    for idx, turn in enumerate(turns):
        if any(
            record.get("name") == "find_product"
            and _nonempty_find_results(record.get("results"))
            for record in turn.get("records", [])
        ):
            latest_nonempty_find_idx = idx
            latest_view_records = []

        turn_view_records = [
            {"turn_index": idx, "record": record}
            for record in turn.get("records", [])
            if _valid_view_record(record)
        ]
        if turn_view_records:
            latest_view_records = turn_view_records

        for record in turn.get("records", []):
            if not _valid_python_record(record) or not latest_view_records:
                continue
            if max(item["turn_index"] for item in latest_view_records) <= latest_nonempty_find_idx:
                continue
            parsed_budget = _parse_python_result(record.get("results"))
            view_records = [item["record"] for item in latest_view_records]
            view_ids = _combined_view_requested_ids(view_records)
            budget_ids = _budget_product_ids(parsed_budget)
            if not _ids_consistent(view_ids, budget_ids):
                continue
            checks.append(
                {
                    "view_turn_index": max(item["turn_index"] for item in latest_view_records),
                    "python_turn_index": idx,
                    "end_turn_index": idx,
                    "view_records": view_records,
                    "python_record": record,
                    "view_requested_product_ids": view_ids,
                    "budget_product_ids": budget_ids,
                }
            )

    return checks


def _find_latest_valid_check(turns: list[dict]) -> dict | None:
    checks = _valid_checks_from_turns(turns)
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


def _decision_state_from_check(
    turns: list[dict],
    check: dict,
    failed_retry_records: list[dict],
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

    budget_calculation = _parse_python_result(check["python_record"].get("results")) or {}
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
        if "view_product_information" in names and "python_execute" in names:
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
) -> HarnessSnapshot:
    prompt_files = {**DEFAULT_PROMPT_FILES, **(prompt_files or {})}
    turns = _build_turns(history_messages)
    search_trace = _search_trace_from_turns(turns)

    if not turns:
        state_name = STATE_CANDIDATE_SEARCH
        state = {}
        return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)

    valid_checks = _valid_checks_from_turns(turns)
    latest_check = valid_checks[-1] if valid_checks else None
    latest_check_end_idx = latest_check["end_turn_index"] if latest_check else None
    latest_nonempty_find_idx = None
    first_nonempty_find_seen = False
    cold_failed_searches = []

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
            failed_searches = _unique_searches(cold_failed_searches)
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
    failed_retry_records = [
        *cold_failed_searches,
        *_failed_retry_records_from_turns(turns, valid_checks),
    ]
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
        previous_decision_state = _decision_state_from_check(turns, latest_check, failed_retry_records)
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
        state = _decision_state_from_check(turns, latest_check, failed_retry_records)
    return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)


def build_harness_user_prompt(snapshot: HarnessSnapshot, history_messages: list[str]) -> str:
    query = _first_user_message(history_messages)
    state_text = json.dumps(snapshot.state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "# Dialogue Records History\n" + f"<user>{query}</user>\n\n<state>{state_text}</state>"


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
