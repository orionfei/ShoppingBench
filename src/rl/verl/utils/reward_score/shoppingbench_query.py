import json
import os
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PRODUCT_CACHE = ROOT / "dataset" / "shoppingbench_query" / "product_cache.json"


def _strip_assistant_markers(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def _has_valid_format(text: str) -> bool:
    if "<think>" not in text or "</think>" not in text:
        return False
    if "<tool_call>" not in text and "<response>" not in text:
        return False
    if text.count("<think>") != text.count("</think>") or text.count("<think>") != 1:
        return False
    if "<tool_call>" in text:
        return text.count("<tool_call>") == text.count("</tool_call>") == 1
    return text.count("<response>") == text.count("</response>") == 1


def _parse_tool_calls(text: str) -> list[dict]:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    calls = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue
        for call in parsed:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            params = call.get("parameters") or call.get("arguments") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            if isinstance(name, str) and isinstance(params, dict):
                calls.append({"name": name, "parameters": params})
    return calls


def _recommended_ids(text: str) -> list[str]:
    product_ids = []
    for call in _parse_tool_calls(text):
        if call["name"] != "recommend_product":
            continue
        raw = call["parameters"].get("product_ids", "")
        if isinstance(raw, list):
            product_ids = [str(item).strip() for item in raw if str(item).strip()]
        elif isinstance(raw, str):
            product_ids = [item.strip() for item in raw.split(",") if item.strip()]
    return product_ids


def _terminated_success(text: str) -> bool:
    for call in _parse_tool_calls(text):
        if call["name"] == "terminate" and call["parameters"].get("status") == "success":
            return True
    return False


@lru_cache(maxsize=1)
def _product_cache() -> dict[str, dict]:
    raw_path = os.getenv("SHOPPINGBENCH_PRODUCT_CACHE")
    cache_path = Path(raw_path) if raw_path else DEFAULT_PRODUCT_CACHE
    with cache_path.open(encoding="utf-8") as fin:
        return json.load(fin)


def _payable_total(total: float, shop_ids: set[str], voucher: dict) -> tuple[bool, float]:
    eligible = voucher.get("voucher_type") == "platform" or (
        voucher.get("voucher_type") == "shop" and len(shop_ids) == 1
    )
    if not eligible or total < float(voucher.get("threshold", 0)):
        return False, total
    if voucher.get("discount_type") == "fixed":
        return True, total - float(voucher.get("face_value") or 0)
    if voucher.get("discount_type") == "percentage":
        rate = float(voucher.get("discount") or 0)
        cap = float(voucher.get("cap") or 0)
        return True, max(total * (1 - rate), total - cap)
    return False, total


def _load_ground_truth(ground_truth) -> dict:
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return ground_truth


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    text = _strip_assistant_markers(solution_str or "")
    gt = _load_ground_truth(ground_truth)
    reward = gt.get("reward") or []
    voucher = gt.get("voucher") or {}
    expected_ids = [str(item.get("product_id")) for item in reward]
    predicted_ids = _recommended_ids(text)

    format_score = 1.0 if _has_valid_format(text) else 0.0
    count_score = 1.0 if len(predicted_ids) == len(expected_ids) and expected_ids else 0.0
    exact_score = 1.0 if predicted_ids == expected_ids and expected_ids else 0.0
    terminate_score = 1.0 if _terminated_success(text) else 0.0

    products = []
    product_cache = _product_cache()
    for product_id in predicted_ids:
        product = product_cache.get(str(product_id))
        if product is not None:
            products.append(product)

    budget_score = 0.0
    same_shop_score = 0.0
    payable = None
    total = None
    if len(products) == len(expected_ids) and len(products) == len(predicted_ids):
        total = sum(float(product.get("price") or 0) for product in products)
        shop_ids = {str(product.get("shop_id")) for product in products}
        same_shop_score = 1.0 if voucher.get("voucher_type") != "shop" or len(shop_ids) == 1 else 0.0
        _, payable = _payable_total(total, shop_ids, voucher)
        budget_score = 1.0 if payable <= float(voucher.get("budget", -1)) else 0.0

    success = 1.0 if exact_score and budget_score else 0.0
    score = (
        0.25 * format_score
        + 0.25 * count_score
        + 2.0 * exact_score
        + 1.0 * budget_score
        + 0.25 * terminate_score
    )

    return {
        "score": score,
        "success": success,
        "format": format_score,
        "count": count_score,
        "exact": exact_score,
        "budget": budget_score,
        "same_shop": same_shop_score,
        "terminate": terminate_score,
        "recommended_count": len(predicted_ids),
        "expected_count": len(expected_ids),
        "total_before_voucher": total,
        "payable_total": payable,
    }
