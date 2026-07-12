"""Batched, outcome-only ShoppingBench Coupon & Budget reward.

This module is loaded by VERL's ``BatchRewardManager``.  Its scalar reward is
strictly binary::

    score = terminal_asr = paper_asr * terminate_success

``paper_asr`` reproduces the Coupon & Budget success condition in
``src/agent/run_evaluate.py``: every requested product must obtain a rule score
of one and the resulting basket must be within budget (with the voucher rules
applied).  No format, progress, tool-use, or length shaping is included.

Title embeddings are deliberately lazy, CPU-only, batched across unique titles,
and cached for the lifetime of the reward worker.  Exact product-id matches take
the evaluator's fast path and therefore do not load the embedding model.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCT_CACHE = ROOT / "dataset" / "shoppingbench_query" / "product_cache.json"
DEFAULT_EMBEDDING_MODEL = ROOT / "model" / "Qwen3-Embedding-0.6B"
TITLE_SIMILARITY_THRESHOLD = 0.5


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_json_object(value: Any, *, field: str) -> dict[str, Any]:
    value = _plain(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object, got {type(value).__name__}")
    return value


@lru_cache(maxsize=4)
def _load_product_cache(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fin:
        raw = json.load(fin)
    return {str(product_id): product for product_id, product in raw.items() if isinstance(product, dict)}


def _products(product_cache: Any = None) -> dict[str, dict[str, Any]]:
    if isinstance(product_cache, dict):
        return {str(key): value for key, value in product_cache.items() if isinstance(value, dict)}
    path = product_cache or os.getenv("SHOPPINGBENCH_PRODUCT_CACHE") or DEFAULT_PRODUCT_CACHE
    return _load_product_cache(str(Path(path).resolve()))


def _merge_product(target: dict[str, dict[str, Any]], product: Any) -> None:
    product = _plain(product)
    if not isinstance(product, dict) or product.get("product_id") is None:
        return
    product_id = str(product["product_id"])
    merged = dict(target.get(product_id) or {})
    merged.update({key: value for key, value in product.items() if value is not None})
    target[product_id] = merged


def _augment_products_from_messages(products: dict[str, dict[str, Any]], extra_info: Any, solution: str) -> None:
    """Merge products observed during rollout without making them reward requirements."""

    try:
        from verl.utils.reward_score import shoppingbench_query

        messages = shoppingbench_query._messages_from_extra(extra_info, solution or "")
        _assistant, events = shoppingbench_query._events_from_messages(messages)
        states = shoppingbench_query._states_from_messages(messages)
        _state_candidate_ids, _state_viewed_ids, state_observed = shoppingbench_query._state_product_evidence(states)
        observed_products = []
        for event in events:
            if event.name not in {"find_product", "view_product_information"} or not isinstance(event.observation, list):
                continue
            observed_products.extend(product for product in event.observation if isinstance(product, dict))
        for product in observed_products + list(state_observed.values()):
            _merge_product(products, product)
    except (ImportError, AttributeError, OSError, ValueError, TypeError):
        # The canonical on-disk cache remains the normal source.  Message
        # evidence is an optional resilience path for changed product indexes.
        return


def _augment_products_from_server(products: dict[str, dict[str, Any]], product_ids: Iterable[str]) -> None:
    missing = list(dict.fromkeys(str(item) for item in product_ids if str(item) not in products))
    if not missing:
        return
    try:
        import requests

        base_url = os.getenv("SEARCH_SERVER_URL", "http://127.0.0.1:5631/").rstrip("/")
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            f"{base_url}/view_product_information",
            params={"product_ids": ",".join(missing)},
            timeout=float(os.getenv("SHOPPINGBENCH_REWARD_LOOKUP_TIMEOUT", "5")),
        )
        response.raise_for_status()
        payload = response.json()
        for product in payload if isinstance(payload, list) else []:
            _merge_product(products, product)
    except Exception:
        # An unavailable lookup is a missing product, hence outcome failure; it
        # must never crash or stall a reward batch.
        return


def _dense_diagnostics(solution: str, ground_truth: Any, extra_info: Any) -> dict[str, Any]:
    try:
        from verl.utils.reward_score import shoppingbench_query

        dense = shoppingbench_query.compute_score(solution, ground_truth, extra_info=extra_info)
    except Exception:
        dense = {}
    if not isinstance(dense, dict):
        dense = {}
    plain_extra = _plain(extra_info) or {}
    if not isinstance(plain_extra, dict):
        plain_extra = {}
    return {
        "format": float(dense.get("format", 0.0) or 0.0),
        "tool_valid": float(dense.get("tool_valid", 0.0) or 0.0),
        "protocol": float(dense.get("protocol", 0.0) or 0.0),
        "workflow_valid": float(dense.get("workflow_valid", 0.0) or 0.0),
        "steps": float(dense.get("steps", 0.0) or 0.0),
        "dense_final_success": float(dense.get("final_success", 0.0) or 0.0),
        "dense_task": float(dense.get("task", 0.0) or 0.0),
        "dense_progress": float(dense.get("progress", 0.0) or 0.0),
        "response_tokens": float(plain_extra.get("response_tokens", 0.0) or 0.0),
        "length_truncated": bool(plain_extra.get("length_truncated", False)),
        "output_chars": len(solution or ""),
    }


class _CpuTitleEmbedder:
    """Lazy Transformers wrapper with a process-local title cache."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, tuple[float, ...]] = {}

    def _get_model(self):
        if self._model is None:
            # Import lazily so exact-id-only reward batches do not import torch or
            # reserve any accelerator memory.
            from transformers import AutoModel, AutoTokenizer

            configured = os.getenv("SHOPPINGBENCH_EMBEDDING_MODEL")
            model_name = configured or (
                str(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL.exists() else "Qwen/Qwen3-Embedding-0.6B"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self._model = AutoModel.from_pretrained(model_name).to("cpu").eval()
        return self._tokenizer, self._model

    @staticmethod
    def _vector(value: Any) -> tuple[float, ...]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return tuple(float(item) for item in value)

    def prepare(self, titles: Iterable[str]) -> None:
        missing = list(dict.fromkeys(str(title) for title in titles if str(title) not in self._cache))
        if not missing:
            return
        import torch
        import torch.nn.functional as functional

        tokenizer, model = self._get_model()
        for start in range(0, len(missing), 64):
            title_batch = missing[start : start + 64]
            inputs = tokenizer(
                title_batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to("cpu")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                attention_mask = inputs["attention_mask"]
                if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
                    pooled = hidden[:, -1]
                else:
                    lengths = attention_mask.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0]), lengths]
                encoded = functional.normalize(pooled.float(), p=2, dim=1).cpu()
            for title, vector in zip(title_batch, encoded, strict=True):
                self._cache[title] = self._vector(vector)

    def similarity(self, left: str, right: str) -> float:
        self.prepare((left, right))
        a = self._cache[str(left)]
        b = self._cache[str(right)]
        numerator = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


_TITLE_EMBEDDER = _CpuTitleEmbedder()


def _normalise_call(raw: Any) -> dict[str, Any] | None:
    raw = _plain(raw)
    if not isinstance(raw, dict):
        return None
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        parameters = function.get("arguments", {})
    else:
        name = raw.get("name")
        parameters = raw.get("parameters", raw.get("arguments", {}))
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            parameters = {}
    if not isinstance(name, str) or not isinstance(parameters, dict):
        return None
    return {"name": name, "parameters": parameters}


def _calls_in_text(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", flags=re.DOTALL):
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            continue
        values = parsed if isinstance(parsed, list) else [parsed]
        for value in values:
            call = _normalise_call(value)
            if call is not None:
                calls.append(call)
    return calls


def _malformed_tool_call(solution_str: str, extra_info: Any) -> bool:
    # VERL's trajectory ``messages`` may contain internal executed-tool event
    # objects which are not model JSON and therefore must not be validated as
    # assistant output.  The decoded solution is the canonical generated text.
    del extra_info
    for text in (solution_str or "",):
        for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                return True
            values = parsed if isinstance(parsed, list) else [parsed]
            if any(_normalise_call(value) is None for value in values):
                return True
    return False


def _server_error_observed(solution_str: str, extra_info: Any) -> bool:
    texts = [solution_str or ""]
    for message in _message_list(extra_info):
        content = message.get("content", "")
        texts.append(content if isinstance(content, str) else json.dumps(_plain(content), ensure_ascii=False))
    text = " ".join(texts).lower()
    return any(marker in text for marker in ("internal server error", "engine error", "server_error", "connection error"))


def _message_list(extra_info: Any) -> list[dict[str, Any]]:
    extra = _plain(extra_info) or {}
    if not isinstance(extra, dict):
        return []
    messages = extra.get("messages")
    if isinstance(messages, dict):
        messages = messages.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _tool_calls(solution_str: str, extra_info: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    messages = _message_list(extra_info)
    for message in messages:
        direct = message.get("tool_calls", message.get("tool_call"))
        if direct is not None:
            values = direct if isinstance(direct, list) else [direct]
            calls.extend(call for call in (_normalise_call(value) for value in values) if call is not None)
        else:
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(_plain(content), ensure_ascii=False)
            calls.extend(_calls_in_text(content))
    # BatchRewardManager does not always attach trajectory messages.  In that
    # case the decoded multi-turn solution is the source of truth.  Avoid
    # parsing it a second time when messages were supplied.
    if not messages:
        calls.extend(_calls_in_text(solution_str or ""))
    return calls


def _product_ids(value: Any) -> list[str]:
    value = _plain(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _trajectory_outcome(solution_str: str, extra_info: Any) -> tuple[list[str], float]:
    recommendations: list[list[str]] = []
    terminate_success = False
    for call in _tool_calls(solution_str, extra_info):
        if call["name"] == "recommend_product":
            ids = _product_ids(call["parameters"].get("product_ids"))
            if ids:
                recommendations.append(ids)
        elif call["name"] == "terminate" and call["parameters"].get("status") == "success":
            terminate_success = True
    # run_evaluate.py overwrites product_ids for each recommendation, so the
    # final recommendation is the evaluator-compatible one.
    return (recommendations[-1] if recommendations else []), float(terminate_success)


def _sku_and_attribute_counts(product: dict[str, Any], reward: dict[str, Any]) -> tuple[int, int]:
    sku_flattens: list[set[tuple[Any, Any]]] = [set()]
    for option in (product.get("sku_options") or {}).values():
        if isinstance(option, dict):
            sku_flattens.append(set(option.items()))

    attr_flatten: set[tuple[Any, Any]] = set()
    for key, values in (product.get("attributes") or {}).items():
        for value in values or []:
            attr_flatten.add((key, value))

    max_total = 0
    max_hit = 0
    for sku_flatten in sku_flattens:
        current_total = 0
        current_hit = 0
        for option in reward.get("sku_options") or []:
            if not isinstance(option, dict):
                continue
            for key, value in option.items():
                current_total += 1
                current_hit += int((key, value) in sku_flatten or (key, value) in attr_flatten)
        for attribute in reward.get("attributes") or []:
            if not isinstance(attribute, dict):
                continue
            for key, values in attribute.items():
                for value in values or []:
                    current_total += 1
                    current_hit += int((key, value) in sku_flatten or (key, value) in attr_flatten)
        # These are intentionally independent maxima, matching rewards/orm.py.
        max_total = max(max_total, current_total)
        max_hit = max(max_hit, current_hit)
    return max_total, max_hit


def _rule_score(product: dict[str, Any], reward: dict[str, Any], embedder: Any) -> float:
    if str(product.get("product_id")) == str(reward.get("product_id")):
        return 1.0

    total = 0
    hits = 0
    for title in reward.get("title") or []:
        total += 1
        if embedder.similarity(str(product.get("title", "")), str(title)) >= TITLE_SIMILARITY_THRESHOLD:
            hits += 1

    price = product.get("price")
    for price_range in reward.get("price") or []:
        if not isinstance(price_range, dict):
            continue
        for mode, bounds in price_range.items():
            total += 1
            try:
                lower_bound, upper_bound = bounds
                numeric_price = float(price)
                if mode == "less than" and numeric_price <= float(upper_bound):
                    hits += 1
                elif mode == "greater than" and numeric_price >= float(lower_bound):
                    hits += 1
                elif mode == "between" and float(lower_bound) <= numeric_price <= float(upper_bound):
                    hits += 1
            except (TypeError, ValueError):
                pass

    services = product.get("service") or []
    for service in reward.get("service") or []:
        total += 1
        hits += int(service in services)

    sku_total, sku_hits = _sku_and_attribute_counts(product, reward)
    total += sku_total
    hits += sku_hits
    return hits / total if total else 0.0


def _needed_titles(recommended_ids: list[str], rewards: list[dict[str, Any]], products: dict[str, dict]) -> list[str]:
    titles: list[str] = []
    for index, reward in enumerate(rewards):
        if index >= len(recommended_ids):
            break
        product = products.get(str(recommended_ids[index]))
        if product is None or str(product.get("product_id")) == str(reward.get("product_id")):
            continue
        for title in reward.get("title") or []:
            titles.extend((str(product.get("title", "")), str(title)))
    return titles


def _score_coupon_budget(
    recommended_ids: list[str],
    ground_truth: dict[str, Any],
    products: dict[str, dict[str, Any]],
    embedder: Any,
) -> tuple[float, float]:
    rewards = ground_truth.get("reward") or []
    voucher = ground_truth.get("voucher") or {}
    if not isinstance(rewards, list) or not rewards:
        return 0.0, 0.0

    rule_sum = 0.0
    total_price = 0.0
    shop_ids: set[Any] = set()
    num_hits = 0
    complete_prices = True
    for index, reward in enumerate(rewards):
        if index >= len(recommended_ids) or not isinstance(reward, dict):
            continue
        product = products.get(str(recommended_ids[index]))
        if product is None:
            continue
        rule_sum += _rule_score(product, reward, embedder)
        try:
            total_price += float(product["price"])
        except (KeyError, TypeError, ValueError):
            # The official index always has price; an incomplete custom cache
            # cannot establish budget success.
            complete_prices = False
            continue
        num_hits += 1
        shop_ids.add(product.get("shop_id"))

    rule = rule_sum / len(rewards)
    budget = 0.0
    if complete_prices and num_hits == len(rewards):
        try:
            budget_limit = float(voucher["budget"])
            if total_price <= budget_limit:
                budget = 1.0
            elif voucher.get("voucher_type") == "platform" or (
                voucher.get("voucher_type") == "shop" and len(shop_ids) == 1
            ):
                if total_price >= float(voucher["threshold"]):
                    if voucher.get("discount_type") == "fixed":
                        payable = total_price - float(voucher["face_value"])
                    elif voucher.get("discount_type") == "percentage":
                        payable = max(
                            total_price * (1.0 - float(voucher["discount"])),
                            total_price - float(voucher["cap"]),
                        )
                    else:
                        payable = math.inf
                    budget = float(payable <= budget_limit)
        except (KeyError, TypeError, ValueError):
            budget = 0.0
    return rule, budget


def compute_score_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    product_cache=None,
    embedder=None,
    **_kwargs,
):
    """Return BatchRewardManager-compatible outcome dictionaries."""

    batch_started = time.perf_counter()

    sizes = {len(data_sources), len(solution_strs), len(ground_truths), len(extra_infos)}
    if len(sizes) != 1:
        raise ValueError("data_sources, solution_strs, ground_truths, and extra_infos must have equal lengths")
    unsupported = sorted({str(source) for source in data_sources if str(source) != "shoppingbench_query"})
    if unsupported:
        raise NotImplementedError(f"Unsupported data_source(s): {unsupported}")

    # Copy because message/server evidence is specific to this batch and the
    # cached canonical dictionary must remain immutable across calls.
    products = dict(_products(product_cache))
    title_embedder = embedder or _TITLE_EMBEDDER
    cache_size_before = len(getattr(title_embedder, "_cache", {}))
    prepared: list[tuple[list[str], float, dict[str, Any], dict[str, Any]]] = []
    batch_titles: list[str] = []
    for solution, ground_truth, extra_info in zip(solution_strs, ground_truths, extra_infos, strict=True):
        gt = _load_json_object(ground_truth, field="ground_truth")
        recommended_ids, terminate_success = _trajectory_outcome(str(solution or ""), extra_info)
        _augment_products_from_messages(products, extra_info, str(solution or ""))
        if product_cache is None:
            _augment_products_from_server(products, recommended_ids)
        diagnostics = _dense_diagnostics(str(solution or ""), ground_truth, extra_info)
        diagnostics["json_decode_failure"] = _malformed_tool_call(str(solution or ""), extra_info)
        diagnostics["server_error"] = _server_error_observed(str(solution or ""), extra_info)
        prepared.append((recommended_ids, terminate_success, gt, diagnostics))
        batch_titles.extend(_needed_titles(recommended_ids, gt.get("reward") or [], products))
    if batch_titles:
        title_embedder.prepare(batch_titles)

    results = []
    for recommended_ids, terminate_success, gt, diagnostics in prepared:
        rule, budget = _score_coupon_budget(recommended_ids, gt, products, title_embedder)
        paper_asr = float(rule >= 1.0 and budget >= 1.0)
        terminal_asr = paper_asr * terminate_success
        result = {
            "score": terminal_asr,
            "paper_asr": paper_asr,
            "terminate_success": terminate_success,
            "terminal_asr": terminal_asr,
            "final_success": terminal_asr,
            "rule": rule,
            "budget": budget,
        }
        result.update(diagnostics)
        results.append(result)
    batch_wall = time.perf_counter() - batch_started
    cache_size_after = len(getattr(title_embedder, "_cache", {}))
    for result in results:
        result.update({
            "reward_batch_wall_seconds": batch_wall,
            "reward_batch_size": len(results),
            "title_embedding_cache_size": cache_size_after,
            "title_embeddings_added": cache_size_after - cache_size_before,
            "title_embedding_model_loaded": bool(getattr(title_embedder, "_model", None) is not None),
        })
    return results


# A conventional alias makes the function convenient in configs while retaining
# the explicit batched name used by VERL examples.
compute_score = compute_score_batched
