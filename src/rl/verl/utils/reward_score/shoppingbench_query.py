import ast
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PRODUCT_CACHE = ROOT / "dataset" / "shoppingbench_query" / "product_cache.json"

LEGAL_TOOLS = {
    "find_product",
    "view_product_information",
    "recommend_product",
    "python_execute",
    "terminate",
}

PROGRESS_WEIGHTS = {
    "search_gold_recall": 0.18,
    "select_gold_f1": 0.18,
    "verify_gold_f1": 0.12,
    "shop_constraint_correct": 0.08,
    "budget_attempt_quality": 0.10,
    "budget_recomputed_correct": 0.12,
    "budget_numeric_alignment": 0.12,
    "within_budget_correct": 0.10,
    "recommend_gold_f1": 0.35,
    "recommend_count_match": 0.12,
    "set_exact": 0.25,
    "terminate_quality": 0.10,
}


@dataclass
class ToolEvent:
    name: str
    parameters: dict[str, Any]
    observation_text: str | None = None
    observation: Any = None


def _strip_assistant_markers(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def _has_valid_format(text: str) -> bool:
    text = _strip_assistant_markers(text or "")
    if "<think>" not in text or "</think>" not in text:
        return False
    if "<tool_call>" not in text and "<response>" not in text:
        return False
    if text.count("<think>") != text.count("</think>") or text.count("<think>") != 1:
        return False
    has_tool = "<tool_call>" in text
    has_response = "<response>" in text
    if has_tool and text.count("<tool_call>") != text.count("</tool_call>"):
        return False
    if has_response and text.count("<response>") != text.count("</response>"):
        return False
    return has_tool or has_response


def _extract_json_value(text: str):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict | list):
            return value
    return None


def _object_to_plain(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _object_to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_object_to_plain(item) for item in value]
    return value


def _content_to_text(content) -> str:
    content = _object_to_plain(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return json.dumps(content, ensure_ascii=False)


def _parse_parameters(value) -> dict[str, Any]:
    value = _object_to_plain(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_call(raw) -> dict[str, Any] | None:
    raw = _object_to_plain(raw)
    if not isinstance(raw, dict):
        return None
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        params = function.get("arguments")
    else:
        name = raw.get("name")
        params = raw.get("parameters") or raw.get("arguments")
    params = _parse_parameters(params or {})
    if not isinstance(name, str):
        return None
    return {"name": name, "parameters": params}


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", flags=re.DOTALL)
    calls = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = _extract_json_value(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            call = _normalize_call(item)
            if call is not None:
                calls.append(call)
    return calls


def _product_ids(value) -> list[str]:
    value = _object_to_plain(value)
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return re.findall(r"\d{5,}", value)


def _extract_obs_texts(text: str) -> list[str]:
    matches = re.findall(r"<obs>\s*(.*?)\s*</obs>", text or "", flags=re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def _extract_state_snapshots(text: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<state>\s*(.*?)\s*</state>", text or "", flags=re.DOTALL)
    states = []
    for raw in matches:
        try:
            state = json.loads(raw)
        except Exception:
            state = _extract_json_value(raw)
        if isinstance(state, dict):
            states.append(state)
    return states


def _parse_observation(text: str):
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("<obs>"):
        values = _extract_obs_texts(text)
        text = values[-1] if values else text
    try:
        return json.loads(text)
    except Exception:
        value = _extract_json_value(text)
    return value if value is not None else text


def _messages_from_solution_text(solution_str: str) -> list[dict[str, Any]]:
    text = _strip_assistant_markers(solution_str or "")
    if not text:
        return []
    markers = list(re.finditer(r"(?m)^(assistant|user|tool)\n", text))
    if not markers:
        return [{"role": "assistant", "content": text}]

    messages = []
    prefix = text[: markers[0].start()].strip()
    if prefix:
        messages.append({"role": "assistant", "content": prefix})
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        content = text[start:end].strip()
        if content:
            messages.append({"role": marker.group(1), "content": content})
    return messages


def _messages_from_extra(extra_info, solution_str: str) -> list[dict[str, Any]]:
    extra_info = extra_info or {}
    raw = extra_info.get("messages")
    raw = _object_to_plain(raw)
    if isinstance(raw, dict) and "messages" in raw:
        raw = raw["messages"]
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    return _messages_from_solution_text(solution_str or "")


def _events_from_messages(messages: list[dict[str, Any]]) -> tuple[list[str], list[ToolEvent]]:
    assistant_texts: list[str] = []
    events: list[ToolEvent] = []
    pending: list[ToolEvent] = []

    for message in messages:
        role = message.get("role")
        content = _content_to_text(message.get("content", ""))
        direct_calls = message.get("tool_call")
        direct_obs = message.get("obs")

        if role == "assistant" or direct_calls is not None:
            assistant_texts.append(content)
            calls = []
            if message.get("tool_calls") is not None:
                calls.extend(
                    call for call in (_normalize_call(item) for item in message.get("tool_calls") or []) if call
                )
            if direct_calls is not None:
                calls.extend(call for call in (_normalize_call(item) for item in direct_calls or []) if call)
            if not calls:
                calls.extend(_parse_tool_calls(content))
            for call in calls:
                event = ToolEvent(name=call["name"], parameters=call["parameters"])
                events.append(event)
                pending.append(event)
            if isinstance(direct_obs, list) and calls:
                obs_by_id = {
                    item.get("tool_call_id"): item
                    for item in direct_obs
                    if isinstance(item, dict) and item.get("tool_call_id")
                }
                for event, call in zip(pending[-len(calls) :], calls, strict=False):
                    raw_obs = obs_by_id.get(call.get("tool_call_id")) if isinstance(call, dict) else None
                    if raw_obs is None and direct_obs:
                        raw_obs = direct_obs.pop(0)
                    if isinstance(raw_obs, dict) and "results" in raw_obs:
                        event.observation = raw_obs["results"]
                        event.observation_text = json.dumps(raw_obs["results"], ensure_ascii=False)
            continue

        if role == "tool":
            obs_texts = [content]
        elif role == "user":
            obs_texts = _extract_obs_texts(content)
        else:
            obs_texts = []

        for obs_text in obs_texts:
            if not pending:
                continue
            event = pending.pop(0)
            event.observation_text = obs_text
            event.observation = _parse_observation(obs_text)

    return assistant_texts, events


def _states_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states = []
    for message in messages:
        content = _content_to_text(message.get("content", ""))
        states.extend(_extract_state_snapshots(content))
    return states


@lru_cache(maxsize=1)
def _product_cache() -> dict[str, dict]:
    raw_path = os.getenv("SHOPPINGBENCH_PRODUCT_CACHE")
    cache_path = Path(raw_path) if raw_path else DEFAULT_PRODUCT_CACHE
    with cache_path.open(encoding="utf-8") as fin:
        return {str(key): value for key, value in json.load(fin).items()}


def _merge_product(products_by_id: dict[str, dict], product: dict) -> None:
    product_id = product.get("product_id")
    if product_id is None:
        return
    product_id = str(product_id)
    merged = dict(products_by_id.get(product_id) or {})
    for key, value in product.items():
        if value is not None:
            merged[key] = value
    products_by_id[product_id] = merged


def _product_evidence(events: list[ToolEvent]) -> tuple[set[str], set[str], dict[str, dict]]:
    observed_candidate_ids: set[str] = set()
    viewed_ids: set[str] = set()
    products_by_id = dict(_product_cache())
    for event in events:
        obs = event.observation
        if event.name == "find_product" and isinstance(obs, list):
            for product in obs:
                if not isinstance(product, dict) or product.get("product_id") is None:
                    continue
                product_id = str(product["product_id"])
                observed_candidate_ids.add(product_id)
                _merge_product(products_by_id, product)
        elif event.name == "view_product_information" and isinstance(obs, list):
            for product in obs:
                if not isinstance(product, dict) or product.get("product_id") is None:
                    continue
                product_id = str(product["product_id"])
                viewed_ids.add(product_id)
                _merge_product(products_by_id, product)
    return observed_candidate_ids, viewed_ids, products_by_id


def _state_product_evidence(states: list[dict[str, Any]]) -> tuple[set[str], set[str], dict[str, dict]]:
    observed_candidate_ids: set[str] = set()
    viewed_ids: set[str] = set()
    products_by_id: dict[str, dict] = {}
    for state in states:
        for search in state.get("searches") or []:
            if not isinstance(search, dict):
                continue
            for product in search.get("candidates") or []:
                if not isinstance(product, dict) or product.get("product_id") is None:
                    continue
                product_id = str(product["product_id"])
                observed_candidate_ids.add(product_id)
                _merge_product(products_by_id, product)
        for product in state.get("budget_candidates") or []:
            if not isinstance(product, dict) or product.get("product_id") is None:
                continue
            product_id = str(product["product_id"])
            observed_candidate_ids.add(product_id)
            _merge_product(products_by_id, product)
        for product in state.get("viewed_products") or []:
            if not isinstance(product, dict) or product.get("product_id") is None:
                continue
            product_id = str(product["product_id"])
            viewed_ids.add(product_id)
            _merge_product(products_by_id, product)
    return observed_candidate_ids, viewed_ids, products_by_id


def _merge_evidence_products(target: dict[str, dict], source: dict[str, dict]) -> None:
    for product in source.values():
        _merge_product(target, product)


def _normalize_voucher(voucher: dict) -> dict[str, Any]:
    discount = voucher.get("discount")
    discount_dict = discount if isinstance(discount, dict) else {}
    return {
        "voucher_type": voucher.get("voucher_type") or voucher.get("scope"),
        "threshold": _to_float(voucher.get("threshold")),
        "discount_type": voucher.get("discount_type") or discount_dict.get("type"),
        "face_value": _to_float(voucher.get("face_value") or discount_dict.get("value")),
        "discount": _to_float(discount_dict.get("rate") if discount_dict else discount),
        "cap": _to_float(voucher.get("cap") or discount_dict.get("cap")),
        "budget": _to_float(voucher.get("budget")),
    }


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _payable_total(total: float, shop_ids: set[str], voucher: dict) -> tuple[bool, float]:
    threshold = voucher.get("threshold")
    eligible = voucher.get("voucher_type") == "platform" or (
        voucher.get("voucher_type") == "shop" and len(shop_ids) == 1
    )
    if threshold is None or not eligible or total < threshold:
        return False, total
    if voucher.get("discount_type") == "fixed":
        return True, total - float(voucher.get("face_value") or 0)
    if voucher.get("discount_type") == "percentage":
        rate = float(voucher.get("discount") or 0)
        cap = float(voucher.get("cap") or 0)
        return True, max(total * (1 - rate), total - cap)
    return False, total


def _recompute_budget(product_ids: list[str], products_by_id: dict[str, dict], voucher: dict) -> dict[str, Any]:
    if not product_ids:
        return {"supported": False, "within_budget": False}
    products = []
    missing = []
    for product_id in product_ids:
        product = products_by_id.get(str(product_id))
        if product is None or product.get("price") is None or product.get("shop_id") is None:
            missing.append(str(product_id))
        else:
            products.append(product)
    if missing or len(products) != len(product_ids) or voucher.get("budget") is None:
        return {"supported": False, "within_budget": False, "missing_product_ids": missing}
    total = round(sum(float(product.get("price") or 0) for product in products), 2)
    shop_ids = {str(product.get("shop_id")) for product in products if product.get("shop_id") is not None}
    voucher_used, payable = _payable_total(total, shop_ids, voucher)
    payable = round(payable, 2)
    return {
        "supported": True,
        "total_before_voucher": total,
        "shop_ids": sorted(shop_ids),
        "voucher_used": voucher_used,
        "payable_total": payable,
        "within_budget": payable <= float(voucher["budget"]),
    }


def _parse_budget_calc_from_python(event: ToolEvent) -> dict[str, Any] | None:
    obs = event.observation
    if not isinstance(obs, dict):
        return None
    success = obs.get("success")
    stdout = obs.get("stdout") or obs.get("observation") or ""
    parsed = _extract_json_value(stdout)
    if isinstance(parsed, dict):
        parsed["_success"] = success is not False
        return parsed
    return {"_success": success is not False, "stdout": stdout}


def _ids_from_python_code(code: str) -> list[str]:
    patterns = [
        r"\bproduct_ids\s*[:=]\s*(\[[^\]]+\])",
        r"['\"]product_ids['\"]\s*:\s*(\[[^\]]+\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, code or "")
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(1))
        except Exception:
            continue
        ids = _product_ids(value)
        if ids:
            return ids
    return []


def _looks_like_budget_code(code: str) -> bool:
    text = (code or "").lower()
    if "product_ids" not in text:
        return False
    budget_terms = ("budget", "voucher", "payable", "total", "price", "discount", "threshold", "within")
    return any(term in text for term in budget_terms)


def _looks_like_budget_calc(calc: dict[str, Any] | None) -> bool:
    if not isinstance(calc, dict):
        return False
    budget_keys = {
        "product_ids",
        "prices",
        "shops",
        "total",
        "total_before_voucher",
        "voucher",
        "budget",
        "discount",
        "threshold",
        "payable",
        "payable_total",
        "apply_ok",
        "within_budget",
    }
    if any(key in calc for key in budget_keys):
        return True
    stdout = str(calc.get("stdout") or "").lower()
    return "product_ids" in stdout and any(
        term in stdout for term in ("budget", "voucher", "payable", "total", "price", "discount", "threshold", "within")
    )


def _first_float(mapping: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in mapping:
            continue
        value = _to_float(mapping.get(name))
        if value is not None:
            return value
    return None


def _to_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "within", "ok"}:
            return True
        if normalized in {"false", "no", "n", "0", "over", "exceed", "exceeds"}:
            return False
    return None


def _numeric_match_score(claimed: float | None, expected: float | None) -> float:
    if claimed is None or expected is None:
        return 0.0
    error = abs(float(claimed) - float(expected))
    denominator = max(abs(float(expected)), 1.0)
    relative_error = error / denominator
    if error <= 0.05 or relative_error <= 0.01:
        return 1.0
    if relative_error <= 0.05:
        return 0.5
    return 0.0


def _budget_numeric_alignment(budget_calcs: list[dict[str, Any]], recomputed: dict[str, Any]) -> float:
    if not budget_calcs or not recomputed.get("supported"):
        return 0.0
    calc = next((item for item in reversed(budget_calcs) if isinstance(item, dict)), None)
    if calc is None:
        return 0.0

    scores = []
    claimed_total = _first_float(calc, ("total_before_voucher", "total", "subtotal", "original_total"))
    if claimed_total is not None:
        scores.append(_numeric_match_score(claimed_total, recomputed.get("total_before_voucher")))

    claimed_payable = _first_float(calc, ("payable_total", "payable", "final_total", "price_after_voucher"))
    if claimed_payable is not None:
        scores.append(_numeric_match_score(claimed_payable, recomputed.get("payable_total")))

    claimed_within_budget = None
    for key in ("within_budget", "is_within_budget", "budget_ok", "within"):
        if key in calc:
            claimed_within_budget = _to_bool(calc.get(key))
            break
    if claimed_within_budget is not None and recomputed.get("within_budget") is not None:
        scores.append(1.0 if claimed_within_budget == bool(recomputed.get("within_budget")) else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def _recommend_count_penalty(recommended_count: int, expected_count: int) -> float:
    if recommended_count <= 0 or expected_count <= 0:
        return 0.0
    distance = abs(recommended_count - expected_count)
    return min(0.24, 0.08 * distance)


def _trajectory_state(events: list[ToolEvent], states: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    event_recommendations: list[list[str]] = []
    state_recommendations: list[list[str]] = []
    terminations: list[str] = []
    budget_calcs: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    budget_attempted = False

    for event in events:
        if event.name == "recommend_product":
            ids = _product_ids(event.parameters.get("product_ids"))
            if ids:
                event_recommendations.append(ids)
        elif event.name == "terminate":
            status = event.parameters.get("status")
            if status is not None:
                terminations.append(str(status))
        elif event.name == "python_execute":
            calc = _parse_budget_calc_from_python(event)
            code = str(event.parameters.get("code") or "")
            calc_is_budget = _looks_like_budget_calc(calc) or _looks_like_budget_code(code)
            if calc_is_budget:
                budget_attempted = True
            if calc and calc_is_budget:
                budget_calcs.append(calc)
                ids = _product_ids(calc.get("product_ids"))
                if not ids:
                    ids = _ids_from_python_code(code)
                if ids:
                    selected_ids = ids
            else:
                ids = _ids_from_python_code(code)
                if ids and calc_is_budget:
                    selected_ids = ids

    for state in states or []:
        state_selected = _product_ids(state.get("selected_product_ids"))
        if state_selected:
            selected_ids = state_selected
        for recommendation in state.get("recommendations") or []:
            if isinstance(recommendation, dict):
                ids = _product_ids(recommendation.get("product_ids"))
            else:
                ids = _product_ids(recommendation)
            if ids:
                state_recommendations.append(ids)
        for termination in state.get("terminations") or []:
            status = termination.get("status") if isinstance(termination, dict) else termination
            if status is not None:
                terminations.append(str(status))
        calc = state.get("latest_budget_calculation")
        if isinstance(calc, dict) and _looks_like_budget_calc(calc):
            budget_attempted = True
            budget_calcs.append(calc)
            ids = _product_ids(calc.get("product_ids"))
            if ids:
                selected_ids = ids
        elif state.get("budget_calculation_trusted"):
            budget_attempted = True

    if event_recommendations:
        recommended_ids = event_recommendations[-1]
    elif state_recommendations:
        recommended_ids = state_recommendations[-1]
    else:
        recommended_ids = []
    if not selected_ids:
        selected_ids = recommended_ids
    return {
        "selected_ids": selected_ids,
        "recommended_ids": recommended_ids,
        "terminate_success": any(status == "success" for status in terminations),
        "budget_calculations": budget_calcs,
        "budget_attempted": budget_attempted,
    }


def _ratio_overlap(left: set[str] | list[str], right: set[str] | list[str]) -> float:
    right_set = set(right)
    if not right_set:
        return 0.0
    return len(set(left) & right_set) / len(right_set)


def _set_precision(left: set[str] | list[str], right: set[str] | list[str]) -> float:
    left_set = set(left)
    if not left_set:
        return 0.0
    return len(left_set & set(right)) / len(left_set)


def _set_f1(left: set[str] | list[str], right: set[str] | list[str]) -> float:
    precision = _set_precision(left, right)
    recall = _ratio_overlap(left, right)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _same_shop_correct(product_ids: list[str], products_by_id: dict[str, dict], voucher: dict) -> float:
    if voucher.get("voucher_type") == "platform":
        return 1.0
    if not product_ids:
        return 0.0
    shop_ids = set()
    for product_id in product_ids:
        product = products_by_id.get(str(product_id))
        if product is None or product.get("shop_id") is None:
            return 0.0
        shop_ids.add(str(product["shop_id"]))
    return 1.0 if len(shop_ids) == 1 else 0.0


def _tool_validity(events: list[ToolEvent], assistant_texts: list[str]) -> float:
    if not events:
        return 1.0 if any("<response>" in text for text in assistant_texts) else 0.0
    scores = []
    for event in events:
        params = event.parameters
        obs = event.observation
        ok = event.name in LEGAL_TOOLS and isinstance(params, dict)
        if event.name == "find_product":
            ok = ok and isinstance(params.get("q"), str) and params.get("page") is not None and isinstance(obs, list)
        elif event.name == "view_product_information":
            ok = ok and bool(_product_ids(params.get("product_ids"))) and isinstance(obs, list)
        elif event.name == "recommend_product":
            ok = ok and bool(_product_ids(params.get("product_ids"))) and event.observation_text is not None
        elif event.name == "terminate":
            ok = ok and params.get("status") in {"success", "failure"} and event.observation_text is not None
        elif event.name == "python_execute":
            ok = ok and isinstance(params.get("code"), str) and isinstance(obs, dict) and obs.get("success") is not False
        scores.append(1.0 if ok else 0.0)
    return sum(scores) / len(scores)


def _format_score(assistant_texts: list[str]) -> float:
    if not assistant_texts:
        return 0.0
    return sum(1.0 if _has_valid_format(text) else 0.0 for text in assistant_texts) / len(assistant_texts)


def _load_ground_truth(ground_truth) -> dict:
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return ground_truth


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _protocol_weight(extra_info: dict) -> float:
    start = _env_float("SHOPPINGBENCH_PROTOCOL_WEIGHT_START", 0.2)
    explicit_steps = _env_float("SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS", 0.0)
    total_steps = _to_float((extra_info or {}).get("total_training_steps"))
    fraction = _env_float("SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION", 0.1)
    anneal_steps = explicit_steps if explicit_steps > 0 else (total_steps or 0) * fraction
    if anneal_steps <= 0:
        return 0.0
    step = _to_float((extra_info or {}).get("global_step")) or 0.0
    return max(0.0, start * (1.0 - step / anneal_steps))


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    gt = _load_ground_truth(ground_truth)
    gold_items = gt.get("reward") or []
    gold_ids = [str(item.get("product_id")) for item in gold_items if item.get("product_id") is not None]
    voucher = _normalize_voucher(gt.get("voucher") or {})

    messages = _messages_from_extra(extra_info, solution_str or "")
    assistant_texts, events = _events_from_messages(messages)
    states = _states_from_messages(messages)
    message_count = len(messages)
    assistant_message_count = sum(1 for message in messages if message.get("role") == "assistant")
    user_obs_message_count = sum(
        1 for message in messages if message.get("role") == "user" and "<obs>" in _content_to_text(message.get("content"))
    )
    state_user_message_count = sum(
        1 for message in messages if message.get("role") == "user" and "<state>" in _content_to_text(message.get("content"))
    )
    observed_event_count = sum(1 for event in events if event.observation is not None or event.observation_text is not None)
    observed_candidate_ids, viewed_ids, products_by_id = _product_evidence(events)
    state_candidate_ids, state_viewed_ids, state_products_by_id = _state_product_evidence(states)
    observed_candidate_ids.update(state_candidate_ids)
    viewed_ids.update(state_viewed_ids)
    _merge_evidence_products(products_by_id, state_products_by_id)
    state = _trajectory_state(events, states)

    selected_ids = state["selected_ids"]
    recommended_ids = state["recommended_ids"]
    budget_ids = selected_ids or recommended_ids
    selected_budget = _recompute_budget(budget_ids, products_by_id, voucher)
    recommended_budget = _recompute_budget(recommended_ids, products_by_id, voucher)

    format_valid = _format_score(assistant_texts)
    tool_valid = _tool_validity(events, assistant_texts)
    protocol_reward = 0.5 * format_valid + 0.5 * tool_valid

    search_gold_recall = _ratio_overlap(observed_candidate_ids, gold_ids)
    select_gold_overlap = _ratio_overlap(selected_ids, gold_ids)
    select_gold_precision = _set_precision(selected_ids, gold_ids)
    select_gold_f1 = _set_f1(selected_ids, gold_ids)
    same_shop_correct = _same_shop_correct(budget_ids, products_by_id, voucher)
    verify_selected_gold = _ratio_overlap(set(viewed_ids) & set(selected_ids), gold_ids)
    verify_gold_f1 = _set_f1(viewed_ids, gold_ids)
    budget_gold_f1 = _set_f1(budget_ids, gold_ids)
    shop_constraint_correct = same_shop_correct * budget_gold_f1
    budget_recomputed_correct = (
        (1.0 if state["budget_attempted"] else 0.0)
        * (1.0 if selected_budget.get("supported") else 0.0)
        * budget_gold_f1
    )
    budget_attempt_quality = (1.0 if state["budget_attempted"] else 0.0) * budget_gold_f1
    budget_numeric_alignment = _budget_numeric_alignment(state["budget_calculations"], selected_budget) * budget_gold_f1
    within_budget_correct = (1.0 if selected_budget.get("within_budget") else 0.0) * budget_gold_f1
    recommend_gold_overlap = _ratio_overlap(recommended_ids, gold_ids)
    recommend_gold_precision = _set_precision(recommended_ids, gold_ids)
    recommend_gold_f1 = _set_f1(recommended_ids, gold_ids)
    recommend_count_match = (
        recommend_gold_f1 if recommended_ids and len(recommended_ids) == len(gold_ids) else 0.0
    )
    ordered_exact_success = 1.0 if recommended_ids == gold_ids and gold_ids else 0.0
    set_exact_success = (
        1.0 if recommended_ids and len(recommended_ids) == len(gold_ids) and set(recommended_ids) == set(gold_ids) else 0.0
    )
    terminate_quality = recommend_gold_f1 if state["terminate_success"] and recommended_ids else 0.0

    progress_components = {
        "search_gold_recall": search_gold_recall,
        "select_gold_f1": select_gold_f1,
        "verify_gold_f1": verify_gold_f1,
        "shop_constraint_correct": shop_constraint_correct,
        "budget_attempt_quality": budget_attempt_quality,
        "budget_recomputed_correct": budget_recomputed_correct,
        "budget_numeric_alignment": budget_numeric_alignment,
        "within_budget_correct": within_budget_correct,
        "recommend_gold_f1": recommend_gold_f1,
        "recommend_count_match": recommend_count_match,
        "set_exact": set_exact_success,
        "terminate_quality": terminate_quality,
    }
    progress = sum(PROGRESS_WEIGHTS[key] * progress_components[key] for key in PROGRESS_WEIGHTS)

    exact_success = set_exact_success
    budget_success = 1.0 if set_exact_success and recommended_budget.get("within_budget") else 0.0
    terminate_after_valid_recommend = 1.0 if state["terminate_success"] and budget_success else 0.0
    final_success = 1.0 if budget_success and state["terminate_success"] else 0.0
    outcome = 1.5 * final_success + 0.5 * budget_success + 0.25 * set_exact_success

    steps = max(1, len(assistant_texts))
    step_penalty = _env_float("SHOPPINGBENCH_STEP_PENALTY", 0.005) * steps
    wrong_recommend_penalty = 0.15 if recommended_ids and recommend_gold_f1 == 0.0 else 0.0
    count_penalty = _recommend_count_penalty(len(recommended_ids), len(gold_ids))
    premature_terminate_penalty = 0.10 if state["terminate_success"] and not recommended_ids else 0.0
    invalid_tool_penalty = 0.05 * (1.0 - tool_valid)
    penalties = step_penalty + wrong_recommend_penalty + count_penalty + premature_terminate_penalty + invalid_tool_penalty
    task_reward = progress + outcome - penalties
    protocol_weight = _protocol_weight(extra_info)
    score = task_reward + protocol_weight * protocol_reward

    return {
        "score": score,
        "success": final_success,
        "final_success": final_success,
        "format": format_valid,
        "tool_valid": tool_valid,
        "protocol": protocol_reward,
        "protocol_weight": protocol_weight,
        "progress": progress,
        "outcome": outcome,
        "task": task_reward,
        "exact": exact_success,
        "ordered_exact": ordered_exact_success,
        "set_exact": set_exact_success,
        "budget": budget_success,
        "terminate": 1.0 if state["terminate_success"] else 0.0,
        "same_shop": same_shop_correct,
        "step_penalty": step_penalty,
        "wrong_recommend_penalty": wrong_recommend_penalty,
        "count_penalty": count_penalty,
        "premature_terminate_penalty": premature_terminate_penalty,
        "invalid_tool_penalty": invalid_tool_penalty,
        "penalties": penalties,
        "steps": steps,
        "message_count": message_count,
        "assistant_message_count": assistant_message_count,
        "user_obs_message_count": user_obs_message_count,
        "state_user_message_count": state_user_message_count,
        "event_count": len(events),
        "observed_event_count": observed_event_count,
        "state_count": len(states),
        "recommended_count": len(recommended_ids),
        "expected_count": len(gold_ids),
        "search_gold_recall": search_gold_recall,
        "select_gold_overlap": select_gold_overlap,
        "select_gold_precision": select_gold_precision,
        "select_gold_f1": select_gold_f1,
        "verify_selected_gold": verify_selected_gold,
        "verify_gold_f1": verify_gold_f1,
        "shop_constraint_correct": shop_constraint_correct,
        "budget_attempted": 1.0 if state["budget_attempted"] else 0.0,
        "budget_attempt_quality": budget_attempt_quality,
        "budget_gold_f1": budget_gold_f1,
        "budget_recomputed_correct": budget_recomputed_correct,
        "budget_numeric_alignment": budget_numeric_alignment,
        "within_budget_correct": within_budget_correct,
        "recommend_gold_overlap": recommend_gold_overlap,
        "recommend_gold_precision": recommend_gold_precision,
        "recommend_gold_f1": recommend_gold_f1,
        "recommend_count_match": recommend_count_match,
        "terminate_quality": terminate_quality,
        "terminate_after_valid_recommend": terminate_after_valid_recommend,
        "total_before_voucher": recommended_budget.get("total_before_voucher"),
        "payable_total": recommended_budget.get("payable_total"),
        "selected_total_before_voucher": selected_budget.get("total_before_voucher"),
        "selected_payable_total": selected_budget.get("payable_total"),
        "recommended_ids": ",".join(recommended_ids),
        "expected_ids": ",".join(gold_ids),
    }
