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
    lowered_query = query.lower()
    voucher_pos = -1
    for marker in (
        "but i have a voucher",
        "i have a voucher",
        "voucher with the following rules",
        "voucher rules",
    ):
        voucher_pos = lowered_query.find(marker)
        if voucher_pos >= 0:
            break
    if voucher_pos < 0:
        voucher_pos = lowered_query.rfind("voucher")
    voucher_text = query[voucher_pos:] if voucher_pos >= 0 else query
    discount_type = None
    discount = {}
    fixed_value = _number_after(
        r"fixed discount of `?([0-9]+(?:\.[0-9]+)?)`?",
        voucher_text,
    )
    percentage_value = _number_after(
        r"percentage discount of `?([0-9]+(?:\.[0-9]+)?)%`?",
        voucher_text,
    )
    if fixed_value is not None:
        discount_type = "fixed"
        discount = {"type": "fixed", "value": fixed_value}
    elif percentage_value is not None:
        discount_type = "percentage"
        cap = _number_after(r"cap of `?([0-9]+(?:\.[0-9]+)?)`?", voucher_text)
        discount = {
            "type": "percentage",
            "rate": percentage_value / 100,
            "cap": cap,
        }

    scope = None
    lowered = voucher_text.lower()
    if "same shop" in lowered:
        scope = "shop"
    elif "applies to all products" in lowered:
        scope = "platform"

    voucher = {
        "scope": scope,
        "threshold": _number_after(
            r"(?:exceeds|exceeding|over|above) `?([0-9]+(?:\.[0-9]+)?)`?",
            voucher_text,
        ),
        "budget": _number_after(r"budget is only `?([0-9]+(?:\.[0-9]+)?)`?", query),
        "discount": discount,
        "discount_type": discount_type,
    }
    discount_ok = bool(discount) and (
        discount_type == "fixed"
        or (discount_type == "percentage" and discount.get("cap") is not None)
    )
    voucher["parse_ok"] = all(
        voucher.get(key) is not None for key in ("scope", "threshold", "budget")
    ) and discount_ok
    return voucher


def _slim_find_product(product: dict) -> dict:
    return {
        key: product[key]
        for key in ("product_id", "shop_id", "title", "price", "service")
        if key in product
    }


def _budget_candidate(product: dict) -> dict:
    return {
        key: product[key]
        for key in ("product_id", "shop_id", "price")
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
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed["_success"] = results.get("success") is True
        return parsed
    except Exception:
        return {"observation": text, "success": results.get("success")}


def _budget_status(calculation) -> bool | None:
    if not isinstance(calculation, dict):
        return None
    value = calculation.get("within_budget")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _close_amount(left, right) -> bool:
    try:
        return abs(float(left) - float(right)) <= 0.01
    except Exception:
        return False


def _budget_lookup(budget_candidates: list[dict]) -> dict[str, dict]:
    lookup = {}
    for candidate in budget_candidates:
        if not isinstance(candidate, dict) or not candidate.get("product_id"):
            continue
        product_id = str(candidate["product_id"])
        if product_id not in lookup:
            lookup[product_id] = candidate
    return lookup


def _budget_calculation_trusted(
    calculation: dict,
    voucher: dict,
    budget_candidates: list[dict],
) -> bool:
    if not isinstance(calculation, dict):
        return False
    if calculation.get("_success") is False:
        return False
    if voucher.get("parse_ok") is not True:
        return False

    product_ids = calculation.get("product_ids")
    shop_ids = calculation.get("shop_ids")
    if not isinstance(product_ids, list) or not product_ids:
        return False
    if not isinstance(shop_ids, list) or len(shop_ids) != len(product_ids):
        return False

    for key in ("total_before_voucher", "payable_total", "budget"):
        try:
            float(calculation[key])
        except Exception:
            return False

    if _budget_status(calculation) is None:
        return False
    if not isinstance(calculation.get("voucher_used"), bool):
        return False

    candidate_by_id = _budget_lookup(budget_candidates)
    prices = []
    candidate_shop_ids = []
    for product_id, calc_shop_id in zip(product_ids, shop_ids):
        candidate = candidate_by_id.get(str(product_id))
        if not candidate:
            return False
        if candidate.get("price") is None or candidate.get("shop_id") is None:
            return False
        if str(candidate.get("shop_id")) != str(calc_shop_id):
            return False
        try:
            prices.append(float(candidate["price"]))
        except Exception:
            return False
        candidate_shop_ids.append(str(candidate["shop_id"]))

    recomputed_total = round(sum(prices), 2)
    recomputed_payable, recomputed_voucher_used = _discounted_total(
        recomputed_total,
        set(candidate_shop_ids),
        voucher,
        selected_count=len(product_ids),
        known_shop_count=len(candidate_shop_ids),
    )
    if recomputed_payable is None:
        return False
    recomputed_payable = round(float(recomputed_payable), 2)
    recomputed_within_budget = recomputed_payable <= float(voucher["budget"])

    return (
        _close_amount(calculation.get("total_before_voucher"), recomputed_total)
        and _close_amount(calculation.get("payable_total"), recomputed_payable)
        and _close_amount(calculation.get("budget"), voucher.get("budget"))
        and calculation.get("voucher_used") is recomputed_voucher_used
        and _budget_status(calculation) is recomputed_within_budget
    )


def _discounted_total(
    total: float,
    shop_ids: set[str],
    voucher: dict,
    selected_count: int | None = None,
    known_shop_count: int | None = None,
):
    if total is None:
        return None, False
    scope = voucher.get("scope")
    threshold = voucher.get("threshold")
    discount = voucher.get("discount") or {}
    all_shops_known = (
        selected_count is None
        or known_shop_count is None
        or known_shop_count == selected_count
    )
    eligible_scope = scope == "platform" or (
        scope == "shop" and all_shops_known and len(shop_ids) == 1
    )
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
    requested_view_product_ids = []
    budget_calculations = []
    recommendations = []
    terminations = []
    budget_candidates = []
    budget_candidate_ids = set()

    for step_no, item in enumerate(history_messages, 1):
        calls = _json_role_value(item, "tool_call") or []
        obs = _json_role_value(item, "obs") or []
        obs_by_id = {}
        for observation in obs:
            if not isinstance(observation, dict):
                continue
            tool_call_id = observation.get("tool_call_id")
            if tool_call_id and tool_call_id not in obs_by_id:
                obs_by_id[tool_call_id] = observation
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            params = call.get("parameters", {}) or {}
            observation = obs_by_id.get(call.get("tool_call_id"), {})
            results = observation.get("results")
            if name == "find_product":
                products = results if isinstance(results, list) else []
                for product in products:
                    if not isinstance(product, dict) or not product.get("product_id"):
                        continue
                    product_id = str(product["product_id"])
                    if product_id in budget_candidate_ids:
                        continue
                    budget_candidate_ids.add(product_id)
                    budget_candidates.append(_budget_candidate(product))
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
                requested_view_product_ids.extend(
                    pid for pid in ids if pid not in requested_view_product_ids
                )
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
                recommendations.append({"product_ids": ids})
            elif name == "terminate":
                terminations.append({"status": params.get("status")})

    latest_budget_calculation = budget_calculations[-1] if budget_calculations else None
    budget_calculation_trusted = _budget_calculation_trusted(
        latest_budget_calculation, voucher, budget_candidates
    )

    if recommendations:
        selected_product_ids = recommendations[-1]["product_ids"]
    elif budget_calculation_trusted:
        selected_product_ids = [
            str(pid) for pid in latest_budget_calculation.get("product_ids", [])
        ]
    else:
        selected_product_ids = []

    selected_candidates = []
    seen = set()
    for pid in selected_product_ids:
        for search in searches:
            for candidate in search["candidates"]:
                if str(candidate.get("product_id")) == pid and pid not in seen:
                    selected_candidates.append(candidate)
                    seen.add(pid)

    calc_total = (
        latest_budget_calculation.get("total_before_voucher")
        if budget_calculation_trusted
        else None
    )
    calc_payable = (
        latest_budget_calculation.get("payable_total")
        if budget_calculation_trusted
        else None
    )
    calc_shop_ids = (
        latest_budget_calculation.get("shop_ids")
        if budget_calculation_trusted
        else None
    )

    all_selected_prices_known = (
        bool(selected_product_ids)
        and len(selected_candidates) == len(selected_product_ids)
        and all("price" in item for item in selected_candidates)
    )
    if calc_total is not None:
        selected_total = round(float(calc_total), 2)
    elif all_selected_prices_known:
        selected_total = round(
            sum(float(item.get("price") or 0) for item in selected_candidates), 2
        )
    else:
        selected_total = None

    selected_shop_ids = {
        str(item.get("shop_id"))
        for item in selected_candidates
        if item.get("shop_id") is not None
    }
    known_shop_count = len(
        [item for item in selected_candidates if item.get("shop_id") is not None]
    )
    if isinstance(calc_shop_ids, list):
        selected_shop_ids.update(str(shop_id) for shop_id in calc_shop_ids)
        known_shop_count = len(calc_shop_ids)
    shop_info_complete = bool(selected_product_ids) and known_shop_count == len(
        selected_product_ids
    )

    if calc_payable is not None:
        payable_total = round(float(calc_payable), 2)
        voucher_applicable = latest_budget_calculation.get("voucher_used")
    else:
        payable_total, voucher_applicable = _discounted_total(
            selected_total,
            selected_shop_ids,
            voucher,
            selected_count=len(selected_product_ids) if selected_product_ids else None,
            known_shop_count=known_shop_count,
        )
    if payable_total is not None:
        payable_total = round(payable_total, 2)

    pending = []
    if recommendations:
        pending.append("terminate_if_not_done")
    elif selected_product_ids:
        missing_details = [
            pid for pid in selected_product_ids if pid not in viewed_products
        ]
        if budget_calculations:
            budget_status = (
                _budget_status(budget_calculations[-1])
                if budget_calculation_trusted
                else None
            )
        else:
            budget_status = None

        if budget_calculations and not budget_calculation_trusted:
            pending.append("check_voucher_budget")
        elif budget_status is False:
            pending.append("revise_selection_or_fail")
        elif missing_details:
            pending.append("verify_product_information")
            if not budget_calculations:
                pending.append("check_voucher_budget")
        elif budget_status is True:
            pending.append("recommend_products")
        else:
            pending.append("check_voucher_budget")
    elif requested_view_product_ids:
        pending.append("check_voucher_budget")
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
            "parse_ok": voucher.get("parse_ok"),
        },
        "searches": searches,
        "budget_candidates": budget_candidates,
        "requested_view_product_ids": requested_view_product_ids,
        "selected_product_ids": selected_product_ids,
        "selected_total_before_voucher": selected_total,
        "shop_anchor": next(iter(selected_shop_ids), None)
        if voucher.get("scope") == "shop"
        and shop_info_complete
        and len(selected_shop_ids) == 1
        else None,
        "voucher_applicable_if_now": voucher_applicable
        if selected_product_ids
        else None,
        "payable_total_if_now": payable_total if selected_product_ids else None,
        "within_budget_if_now": payable_total <= voucher.get("budget")
        if selected_product_ids
        and payable_total is not None
        and voucher.get("budget") is not None
        else None,
        "viewed_products": list(viewed_products.values()),
        "latest_budget_calculation": latest_budget_calculation,
        "budget_calculation_trusted": budget_calculation_trusted,
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
