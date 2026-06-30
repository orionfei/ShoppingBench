import re

try:
    import ujson as json
except ModuleNotFoundError:
    import json


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


def _candidate_product_id(product: dict) -> str | None:
    if not isinstance(product, dict) or not product.get("product_id"):
        return None
    return str(product["product_id"])


def _focused_search_candidates(
    products: list[dict],
    max_candidates_per_search: int,
    focus_shop_ids: set[str],
    extra_focus_candidates: int = 3,
) -> list[dict]:
    """Keep top search hits plus a few same-shop candidates needed for voucher decisions."""
    candidates = []
    seen_ids = set()
    for product in products[:max_candidates_per_search]:
        product_id = _candidate_product_id(product)
        if not product_id:
            continue
        candidates.append(_slim_find_product(product))
        seen_ids.add(product_id)

    if not focus_shop_ids:
        return candidates

    extra_count = 0
    for product in products[max_candidates_per_search:]:
        product_id = _candidate_product_id(product)
        if not product_id or product_id in seen_ids:
            continue
        shop_id = product.get("shop_id")
        if shop_id is None or str(shop_id) not in focus_shop_ids:
            continue
        candidates.append(_slim_find_product(product))
        seen_ids.add(product_id)
        extra_count += 1
        if extra_count >= extra_focus_candidates:
            break
    return candidates


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
    if observation is None:
        observation = results.get("stdout")
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


def _has_verified_details_for_all(state: dict, selected_product_ids: list[str]) -> bool:
    if not selected_product_ids:
        return False
    viewed_products = state.get("viewed_products")
    verified_details = state.get("verified_details")
    detail_items = []
    if isinstance(viewed_products, list):
        detail_items.extend(viewed_products)
    if isinstance(verified_details, list):
        detail_items.extend(verified_details)
    detailed_ids = {
        str(item.get("product_id"))
        for item in detail_items
        if isinstance(item, dict) and item.get("product_id") is not None
    }
    return all(pid in detailed_ids for pid in selected_product_ids)


def _search_signature(search: dict) -> str:
    if not isinstance(search, dict):
        return ""
    params = search.get("parameters") or {}
    if not isinstance(params, dict):
        return ""
    normalized = {
        key: str(params.get(key, "")).strip().lower()
        for key in ("q", "page", "shop_id", "price", "sort", "service")
        if params.get(key) not in (None, "", "default")
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _search_repetition_summary(searches: list[dict]) -> dict:
    signatures = []
    repeated = []
    seen = set()
    for search in searches:
        signature = _search_signature(search)
        if not signature:
            continue
        signatures.append(signature)
        if signature in seen and signature not in repeated:
            repeated.append(signature)
        seen.add(signature)
    return {
        "total_searches": len(signatures),
        "unique_searches": len(set(signatures)),
        "repeated_searches": repeated,
        "has_repeated_search": bool(repeated),
    }


def _state_decision_hint(state: dict) -> dict:
    selected_product_ids = [
        str(pid)
        for pid in state.get("selected_product_ids", [])
        if pid is not None
    ]
    recommendations = state.get("recommendations")
    has_recommendations = bool(recommendations)
    if not has_recommendations:
        has_recommendations = bool(state.get("recommended_product_ids"))

    searches = state.get("searches") or []
    observed_counts = state.get("observed_counts") or {}
    has_search_evidence = bool(searches)
    if isinstance(observed_counts, dict):
        has_search_evidence = has_search_evidence or bool(
            observed_counts.get("find_product_candidates")
        )

    latest_budget_calculation = state.get("latest_budget_calculation")
    budget_calculation_trusted = state.get("budget_calculation_trusted")
    if budget_calculation_trusted is not False:
        budget_status = _budget_status(latest_budget_calculation)
    else:
        budget_status = None
    if budget_status is None and isinstance(state.get("within_budget_if_now"), bool):
        budget_status = state.get("within_budget_if_now")

    has_budget_evidence = isinstance(latest_budget_calculation, dict)
    details_complete = _has_verified_details_for_all(state, selected_product_ids)

    if has_recommendations:
        return {
            "summary": "A product recommendation has already been sent.",
            "allowed_next_tools": ["terminate"],
            "notes": ["End the dialogue after confirming whether the task is complete."],
        }
    if selected_product_ids:
        if has_budget_evidence and budget_status is False:
            return {
                "summary": "The current selected products do not satisfy the voucher budget.",
                "allowed_next_tools": [
                    "find_product",
                    "view_product_information",
                    "python_execute",
                    "terminate",
                ],
                "notes": [
                    "Revise the product set using observed candidates or finish with failure if no valid set remains."
                ],
            }
        if not details_complete and not has_budget_evidence:
            return {
                "summary": "Products have been selected but still need detail verification and voucher-budget calculation.",
                "allowed_next_tools": ["view_product_information", "python_execute"],
                "notes": ["Verify selected product ids before recommending."],
            }
        if not details_complete:
            return {
                "summary": "Products have been selected but product details are still missing.",
                "allowed_next_tools": ["view_product_information", "python_execute"],
                "notes": ["Check attributes, SKU options, and service constraints."],
            }
        if not has_budget_evidence or budget_calculation_trusted is False:
            return {
                "summary": "Products have been selected and need a trustworthy voucher-budget calculation.",
                "allowed_next_tools": ["python_execute"],
                "notes": ["Calculate total price, voucher eligibility, payable total, and budget status."],
            }
        if budget_status is True:
            return {
                "summary": "The selected products have verified details and satisfy the voucher budget.",
                "allowed_next_tools": ["recommend_product", "terminate"],
                "notes": ["Recommend the selected product ids in the user's requested order."],
            }
        return {
            "summary": "The selected products need a final voucher-budget decision.",
            "allowed_next_tools": ["python_execute", "recommend_product", "terminate"],
            "notes": ["Use the calculation result to decide whether recommendation is valid."],
        }
    if state.get("requested_view_product_ids"):
        return {
            "summary": "Product details have been requested for candidate products.",
            "allowed_next_tools": ["python_execute"],
            "notes": ["Use product details and prices to compute voucher budget before recommending."],
        }
    if has_search_evidence:
        repetition = state.get("search_repetition")
        if not isinstance(repetition, dict):
            repetition = _search_repetition_summary(searches)
        if repetition.get("has_repeated_search"):
            return {
                "summary": "Search results are already available and a search has been repeated.",
                "allowed_next_tools": [
                    "view_product_information",
                    "python_execute",
                ],
                "conditional_next_tools": [
                    {
                        "tool": "find_product",
                        "when": "Only if a required item has no plausible retained candidate, and the new search must change q, page, shop_id, price, sort, or service.",
                    }
                ],
                "notes": [
                    "Do not repeat an identical find_product call.",
                    "Select visible candidate product ids first, then verify details and compute voucher budget.",
                ],
            }
        return {
            "summary": "Search results are available but no product set has been selected yet.",
            "allowed_next_tools": [
                "view_product_information",
                "python_execute",
            ],
            "conditional_next_tools": [
                {
                    "tool": "find_product",
                    "when": "Only if a required item has no plausible retained candidate, and the new search must change q, page, shop_id, price, sort, or service.",
                }
            ],
            "notes": [
                "Choose candidate product ids that match the user request before searching again.",
                "Do not repeat an identical find_product call.",
                "Verify selected ids and compute voucher budget before recommending.",
            ],
        }
    return {
        "summary": "No product evidence has been collected yet.",
        "allowed_next_tools": ["find_product"],
        "notes": ["Search for products matching each requested item."],
    }


def normalize_state_schema(state: dict) -> dict:
    """Return the canonical state-folded schema used by SFT and online rollout."""
    if not isinstance(state, dict):
        return state
    normalized = dict(state)
    normalized.pop("pending", None)
    normalized.pop("decision_hint", None)
    searches = normalized.get("searches") or []
    if isinstance(searches, list):
        normalized["search_repetition"] = _search_repetition_summary(searches)
    normalized["decision_hint"] = _state_decision_hint(normalized)
    return normalized


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


def _positive_int_or_none(value):
    if value is None:
        return None
    try:
        value = int(value)
    except Exception:
        return None
    return value if value > 0 else None


def _cap_recent(items: list, max_items: int | None):
    max_items = _positive_int_or_none(max_items)
    if max_items is None or len(items) <= max_items:
        return items
    return items[-max_items:]


def _cap_by_product_id_priority(
    items: list[dict],
    max_items: int | None,
    priority_product_ids: list[str],
) -> list[dict]:
    max_items = _positive_int_or_none(max_items)
    if max_items is None or len(items) <= max_items:
        return items

    priority = [str(pid) for pid in priority_product_ids if pid is not None]
    selected = []
    selected_ids = set()

    for pid in priority:
        for item in items:
            if not isinstance(item, dict):
                continue
            product_id = item.get("product_id")
            if product_id is None or str(product_id) != pid or str(product_id) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(str(product_id))
            break
        if len(selected) >= max_items:
            return selected

    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        product_id = item.get("product_id")
        if product_id is not None and str(product_id) in selected_ids:
            continue
        selected.append(item)
        if product_id is not None:
            selected_ids.add(str(product_id))
        if len(selected) >= max_items:
            break

    selected.reverse()
    priority_order = {pid: idx for idx, pid in enumerate(priority)}
    return sorted(
        selected,
        key=lambda item: priority_order.get(str(item.get("product_id")), len(priority_order)),
    )


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
    max_searches: int | None = None,
    max_budget_candidates: int | None = None,
    max_viewed_products: int | None = None,
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
    budget_candidate_by_id = {}
    focus_shop_ids = set()

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
                    candidate = _budget_candidate(product)
                    budget_candidates.append(candidate)
                    budget_candidate_by_id[product_id] = candidate
                search_params = {
                    key: params[key]
                    for key in ("q", "page", "shop_id", "price", "sort", "service")
                    if key in params and params[key] not in (None, "", "default")
                }
                search_focus_shop_ids = set(focus_shop_ids)
                if search_params.get("shop_id") is not None:
                    search_focus_shop_ids.add(str(search_params["shop_id"]))
                searches.append(
                    {
                        "parameters": search_params,
                        "candidates": _focused_search_candidates(
                            products,
                            max_candidates_per_search,
                            search_focus_shop_ids,
                        ),
                    }
                )
            elif name == "view_product_information":
                ids = _parse_product_ids(params.get("product_ids"))
                requested_view_product_ids.extend(
                    pid for pid in ids if pid not in requested_view_product_ids
                )
                for pid in ids:
                    candidate = budget_candidate_by_id.get(str(pid))
                    if candidate and candidate.get("shop_id") is not None:
                        focus_shop_ids.add(str(candidate["shop_id"]))
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

    state = {
        "task_type": "voucher_budget",
        "voucher": {
            "scope": voucher.get("scope"),
            "threshold": voucher.get("threshold"),
            "budget": voucher.get("budget"),
            "discount": voucher.get("discount"),
            "parse_ok": voucher.get("parse_ok"),
        },
        "searches": _cap_recent(searches, max_searches),
        "budget_candidates": _cap_by_product_id_priority(
            budget_candidates,
            max_budget_candidates,
            selected_product_ids,
        ),
        "requested_view_product_ids": _cap_recent(
            requested_view_product_ids,
            max_viewed_products,
        ),
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
        "viewed_products": _cap_by_product_id_priority(
            list(viewed_products.values()),
            max_viewed_products,
            selected_product_ids,
        ),
        "latest_budget_calculation": latest_budget_calculation,
        "budget_calculation_trusted": budget_calculation_trusted,
        "recommendations": recommendations,
        "terminations": terminations,
    }
    return normalize_state_schema(state)


def build_state_folded_user_prompt(
    history_messages: list[str],
    max_candidates_per_search: int = 10,
    max_searches: int | None = None,
    max_budget_candidates: int | None = None,
    max_viewed_products: int | None = None,
    never_expand: bool = False,
    min_char_saving_for_state: float = 0.0,
) -> str:
    query = _first_user_message(history_messages)
    raw_prompt = "# Dialogue Records History\n" + "\n\n".join(history_messages)
    parts = [f"<user>{query}</user>"]
    has_assistant_history = any("<tool_call>" in item or "<response>" in item for item in history_messages)
    if has_assistant_history:
        state = build_state_from_history(
            history_messages,
            max_candidates_per_search,
            max_searches=max_searches,
            max_budget_candidates=max_budget_candidates,
            max_viewed_products=max_viewed_products,
        )
        parts.append(
            "<state>"
            + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</state>"
        )
    folded_prompt = "# Dialogue Records History\n" + "\n\n".join(parts)
    if never_expand:
        try:
            min_char_saving_for_state = float(min_char_saving_for_state)
        except Exception:
            min_char_saving_for_state = 0.0
        max_folded_chars = len(raw_prompt) * max(0.0, 1.0 - min_char_saving_for_state)
        if len(folded_prompt) > max_folded_chars:
            return raw_prompt
    return folded_prompt
