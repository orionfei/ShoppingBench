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
        for key in ("product_id", "shop_id", "title", "price", "service")
        if isinstance(product, dict) and key in product
    }


def _slim_viewed_product(product: dict) -> dict:
    if not isinstance(product, dict):
        return {}
    result = {"product_id": product.get("product_id")}
    for key in ("sku_options", "attributes", "service"):
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
        return parsed if isinstance(parsed, dict) else {"observation": text}
    except Exception:
        return {"observation": text, "success": results.get("success")}


def _nonempty_find_results(results) -> bool:
    return isinstance(results, list) and len(results) > 0


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


def _selected_ids_from_turn(turn: dict) -> list[str]:
    selected = []
    for record in turn.get("records", []):
        if record.get("name") != "view_product_information":
            continue
        for product_id in _parse_product_ids(record.get("parameters", {}).get("product_ids")):
            if product_id not in selected:
                selected.append(product_id)
    if selected:
        return selected
    for record in turn.get("records", []):
        if record.get("name") != "python_execute":
            continue
        parsed = _parse_python_result(record.get("results"))
        product_ids = parsed.get("product_ids") if isinstance(parsed, dict) else None
        if isinstance(product_ids, list):
            for product_id in product_ids:
                product_id = str(product_id)
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


def _decision_state_from_check_turn(
    turns: list[dict],
    check_turn_index: int,
    failed_retry_records: list[dict],
) -> dict:
    check_turn = turns[check_turn_index]
    selected_ids = _selected_ids_from_turn(check_turn)
    products_by_id = _collect_products_by_id(turns[: check_turn_index + 1])
    selected_products = [
        products_by_id[product_id]
        for product_id in selected_ids
        if product_id in products_by_id
    ]

    viewed_by_id = {}
    budget_calculation = None
    for record in check_turn.get("records", []):
        if record.get("name") == "view_product_information" and isinstance(record.get("results"), list):
            for product in record["results"]:
                viewed = _slim_viewed_product(product)
                product_id = viewed.get("product_id")
                if product_id is not None:
                    viewed_by_id[str(product_id)] = viewed
        elif record.get("name") == "python_execute":
            parsed = _parse_python_result(record.get("results"))
            if parsed is not None:
                budget_calculation = parsed

    return {
        "selected_products": selected_products,
        "viewed_products": list(viewed_by_id.values()),
        "budget_calculation": budget_calculation or {},
        "failed_retry_searches": _unique_searches(failed_retry_records),
    }


def _candidate_pool_from_turn(turn: dict) -> list[dict]:
    candidates = []
    seen_ids = set()
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
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            candidates.append(_slim_candidate(product))
    return candidates


def _search_trace_from_turns(turns: list[dict]) -> list[dict]:
    trace = []
    current_phase = STATE_CANDIDATE_SEARCH
    for turn_no, turn in enumerate(turns, 1):
        names = {record.get("name") for record in turn.get("records", [])}
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            results = record.get("results")
            result_count = len(results) if isinstance(results, list) else 0
            trace.append(
                {
                    "turn": turn_no,
                    "phase": current_phase,
                    "parameters": _sparse_search_parameters(record.get("parameters", {})),
                    "result_count": result_count,
                    "empty": result_count == 0,
                }
            )
        if "view_product_information" in names or "python_execute" in names:
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

    latest_check_idx = None
    latest_nonempty_find_idx = None
    first_nonempty_find_seen = False
    cold_failed_searches = []

    for idx, turn in enumerate(turns):
        names = {record.get("name") for record in turn.get("records", [])}
        if "view_product_information" in names or "python_execute" in names:
            latest_check_idx = idx
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            if _nonempty_find_results(record.get("results")):
                latest_nonempty_find_idx = idx
                first_nonempty_find_seen = True
            elif not first_nonempty_find_seen:
                cold_failed_searches.append(record.get("parameters", {}))

    if latest_check_idx is None:
        if latest_nonempty_find_idx is None:
            state_name = STATE_CANDIDATE_SEARCH
            state = {}
            failed_searches = _unique_searches(cold_failed_searches)
            if failed_searches:
                state["failed_searches"] = failed_searches
        else:
            state_name = STATE_CANDIDATE_SELECT
            state = {"candidate_pool": _candidate_pool_from_turn(turns[latest_nonempty_find_idx])}
        return HarnessSnapshot(state_name, state, prompt_files[state_name], STATE_TOOLS[state_name], search_trace)

    retry_find_indices = [
        idx
        for idx in range(latest_check_idx + 1, len(turns))
        if any(record.get("name") == "find_product" for record in turns[idx].get("records", []))
    ]
    latest_retry_nonempty_idx = None
    failed_retry_records = []
    for idx in retry_find_indices:
        turn = turns[idx]
        turn_has_nonempty = False
        for record in turn.get("records", []):
            if record.get("name") != "find_product":
                continue
            if _nonempty_find_results(record.get("results")):
                turn_has_nonempty = True
            else:
                failed_retry_records.append(record.get("parameters", {}))
        if turn_has_nonempty:
            latest_retry_nonempty_idx = idx
            failed_retry_records = []

    if latest_retry_nonempty_idx is not None:
        state_name = STATE_CANDIDATE_SELECT
        state = {"candidate_pool": _candidate_pool_from_turn(turns[latest_retry_nonempty_idx])}
    else:
        state_name = STATE_DECISION
        state = _decision_state_from_check_turn(turns, latest_check_idx, failed_retry_records)
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
