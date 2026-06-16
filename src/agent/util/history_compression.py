import re
import ujson as json


USER_RE = re.compile(r"<user>(.*?)</user>", re.DOTALL)
ROLE_RE_TEMPLATE = r"<{role}>(.*?)</{role}>"


def _role_value(text: str, role: str):
    match = re.search(ROLE_RE_TEMPLATE.format(role=role), text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _json_role_value(text: str, role: str):
    value = _role_value(text, role)
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


def _number_after(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    value = float(raw)
    return int(value) if value.is_integer() else value


def parse_voucher_from_query(query: str) -> dict:
    discount_type = None
    discount = {}
    fixed_value = _number_after(r"fixed discount of `?([0-9]+(?:\.[0-9]+)?)`?", query)
    percentage_value = _number_after(
        r"percentage discount of `?([0-9]+(?:\.[0-9]+)?)%`?", query
    )
    if fixed_value is not None:
        discount_type = "fixed"
        discount = {"type": "fixed", "value": fixed_value}
    elif percentage_value is not None:
        discount_type = "percentage"
        cap = _number_after(r"cap of `?([0-9]+(?:\.[0-9]+)?)`?", query)
        discount = {
            "type": "percentage",
            "rate": percentage_value / 100,
            "cap": cap,
        }

    scope = None
    lowered = query.lower()
    if "same shop" in lowered:
        scope = "shop"
    elif "applies to all products" in lowered:
        scope = "platform"

    return {
        "scope": scope,
        "threshold": _number_after(
            r"(?:exceeds|exceeding|over|above) `?([0-9]+(?:\.[0-9]+)?)`?",
            query,
        ),
        "budget": _number_after(r"budget is only `?([0-9]+(?:\.[0-9]+)?)`?", query),
        "discount": discount,
        "discount_type": discount_type,
    }


def _slim_find_product(product: dict) -> dict:
    return {
        key: product[key]
        for key in ("product_id", "shop_id", "title", "price", "service")
        if key in product
    }


def _slim_view_product(product: dict) -> dict:
    return {
        "product_id": product.get("product_id"),
        "sku_options": product.get("sku_options", {}),
        "attributes": product.get("attributes", {}),
    }


def _parse_product_ids(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_python_observation(results):
    if not isinstance(results, dict):
        return None
    observation = results.get("observation")
    if not isinstance(observation, str):
        return None
    text = observation.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return {"observation": text, "success": results.get("success")}


def _discounted_total(total: float, shop_ids: set[str], voucher: dict):
    if total is None:
        return None, False
    scope = voucher.get("scope")
    threshold = voucher.get("threshold")
    discount = voucher.get("discount") or {}
    eligible_scope = scope == "platform" or (scope == "shop" and len(shop_ids) == 1)
    meets_threshold = threshold is None or total >= threshold
    if not eligible_scope or not meets_threshold:
        return total, False
    if discount.get("type") == "fixed":
        return total - (discount.get("value") or 0), True
    if discount.get("type") == "percentage":
        rate = discount.get("rate") or 0
        cap = discount.get("cap") or 0
        return max(total * (1 - rate), total - cap), True
    return total, False


def build_state_from_history(
    history_messages: list[str],
    max_candidates_per_search: int = 10,
) -> dict:
    query = _first_user_message(history_messages)
    voucher = parse_voucher_from_query(query)
    searches = []
    viewed_products = {}
    selected_product_ids = []
    budget_calculations = []
    recommendations = []
    terminations = []

    for step_no, item in enumerate(history_messages, 1):
        calls = _json_role_value(item, "tool_call") or []
        obs = _json_role_value(item, "obs") or []
        obs_by_id = {
            observation.get("tool_call_id"): observation
            for observation in obs
            if isinstance(observation, dict)
        }
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            params = call.get("parameters", {}) or {}
            observation = obs_by_id.get(call.get("tool_call_id"), {})
            results = observation.get("results")
            if name == "find_product":
                products = results if isinstance(results, list) else []
                search_params = {
                    key: params[key]
                    for key in ("q", "page", "shop_id", "price", "sort", "service")
                    if key in params and params[key] not in (None, "", "default")
                }
                searches.append(
                    {
                        "parameters": search_params,
                        "candidates": [
                            _slim_find_product(product)
                            for product in products[:max_candidates_per_search]
                            if isinstance(product, dict)
                        ],
                    }
                )
            elif name == "view_product_information":
                ids = _parse_product_ids(params.get("product_ids"))
                selected_product_ids.extend(pid for pid in ids if pid not in selected_product_ids)
                if isinstance(results, list):
                    for product in results:
                        if isinstance(product, dict) and product.get("product_id"):
                            viewed_products[str(product["product_id"])] = _slim_view_product(product)
            elif name == "python_execute":
                parsed = _parse_python_observation(results)
                if parsed is not None:
                    budget_calculations.append(parsed)
            elif name == "recommend_product":
                ids = _parse_product_ids(params.get("product_ids"))
                selected_product_ids.extend(pid for pid in ids if pid not in selected_product_ids)
                recommendations.append({"product_ids": ids})
            elif name == "terminate":
                terminations.append({"status": params.get("status")})

    selected_candidates = []
    seen = set()
    for pid in selected_product_ids:
        for search in searches:
            for candidate in search["candidates"]:
                if str(candidate.get("product_id")) == pid and pid not in seen:
                    selected_candidates.append(candidate)
                    seen.add(pid)

    selected_total = round(
        sum(float(item.get("price") or 0) for item in selected_candidates), 2
    )
    selected_shop_ids = {
        str(item.get("shop_id"))
        for item in selected_candidates
        if item.get("shop_id") is not None
    }
    payable_total, voucher_applicable = _discounted_total(
        selected_total, selected_shop_ids, voucher
    )
    if payable_total is not None:
        payable_total = round(payable_total, 2)

    pending = []
    if recommendations:
        pending.append("terminate_if_not_done")
    elif selected_product_ids and budget_calculations:
        pending.append("recommend_products")
    elif selected_product_ids:
        pending.append("verify_missing_details_or_check_budget")
    elif searches:
        pending.append("select_candidates_from_search_results")
    else:
        pending.append("search_products")

    return {
        "task_type": "voucher_budget",
        "voucher": {
            "scope": voucher.get("scope"),
            "threshold": voucher.get("threshold"),
            "budget": voucher.get("budget"),
            "discount": voucher.get("discount"),
        },
        "searches": searches,
        "selected_product_ids": selected_product_ids,
        "selected_total_before_voucher": selected_total if selected_candidates else None,
        "shop_anchor": next(iter(selected_shop_ids), None)
        if voucher.get("scope") == "shop" and selected_shop_ids
        else None,
        "voucher_applicable_if_now": voucher_applicable if selected_candidates else None,
        "payable_total_if_now": payable_total if selected_candidates else None,
        "within_budget_if_now": payable_total <= voucher.get("budget")
        if payable_total is not None and voucher.get("budget") is not None
        else None,
        "viewed_products": list(viewed_products.values()),
        "latest_budget_calculation": budget_calculations[-1]
        if budget_calculations
        else None,
        "recommendations": recommendations,
        "terminations": terminations,
        "pending": pending,
    }


def build_state_folded_user_prompt(
    history_messages: list[str],
    max_candidates_per_search: int = 10,
) -> str:
    query = _first_user_message(history_messages)
    parts = [f"<user>{query}</user>"]
    has_assistant_history = any("<tool_call>" in item or "<response>" in item for item in history_messages)
    if has_assistant_history:
        state = build_state_from_history(history_messages, max_candidates_per_search)
        parts.append(
            "<state>"
            + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</state>"
        )
    return "# Dialogue Records History\n" + "\n\n".join(parts)
