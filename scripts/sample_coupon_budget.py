from __future__ import annotations

import argparse
import copy
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import ujson as json
except ImportError:
    import json

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc
            self.n = 0

        def update(self, n=1):
            self.n += n

        def close(self):
            return None


ROOT = Path(__file__).resolve().parents[1]
SERVICE_TEXT = {
    "official": "",
    "freeShipping": "The product offer with free shipping service.",
    "COD": "The product offer with cash on delivery service.",
    "flashsale": "The product offer with LazFlash deals.",
}
VOUCHER_KEYS = {
    "voucher_type",
    "threshold",
    "discount_type",
    "face_value",
    "discount",
    "cap",
    "price_after_voucher",
    "budget",
}
DEFAULT_MODEL_CONFIG = {
    "model": "mimo-v2.5",
    "temperature": 0.35,
    "top_p": 0.8,
    "max_completion_tokens": 512,
    "extra_body": {"thinking": {"type": "disabled"}},
}
PRODUCT_COUNT_WEIGHTS = {1: 0.088, 2: 0.305, 3: 0.317, 4: 0.290}
VOUCHER_TYPE_WEIGHTS = {"platform": 0.40, "shop": 0.60}
DISCOUNT_TYPE_WEIGHTS = {"fixed": 0.40, "percentage": 0.60}
THRESHOLD_BUCKETS = {
    "low": (0.25, 0.20, 0.50),
    "mid": (0.50, 0.50, 0.80),
    "high": (0.25, 0.80, 0.95),
}
BUDGET_SLACK_BUCKETS = {
    "hard": (0.35, 1.00, 1.02),
    "medium": (0.45, 1.02, 1.06),
    "easy": (0.20, 1.06, 1.10),
}
STYLE_BUCKET_WEIGHTS = {
    "conversational": 0.70,
    "brief": 0.15,
    "list-like": 0.15,
}
STYLE_INSTRUCTIONS = {
    "brief": (
        "Write the query as a concise shopping request in one short sentence when possible. "
        "Include every specified product requirement."
    ),
    "conversational": (
        "Write the query in a natural shopping-request style. Include every specified "
        "product requirement, but do not default to First/Second/Third sequencing; use "
        "natural connectors such as and, also, plus, or commas unless the query would be unclear."
    ),
    "list-like": (
        "Write the query in a list-like or semicolon-separated style that clearly separates "
        "the requested products. First/Second/Third wording or numbered fragments are allowed. "
        "Include every specified product requirement."
    ),
}
OPENER_BUCKET_WEIGHTS = {
    "im_looking": 0.45,
    "looking_for": 0.25,
    "show_me": 0.20,
    "find": 0.10,
}
OPENER_INSTRUCTIONS = {
    "im_looking": "Start the query with exactly: I'm looking for",
    "looking_for": "Start the query with exactly: Looking for",
    "show_me": "Start the query with exactly: Show me",
    "find": "Start the query with exactly: Find",
}


def iter_products(documents_file: Path, max_docs: int | None):
    with documents_file.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if max_docs is not None and idx >= max_docs:
                break
            if not line.strip():
                continue
            yield json.loads(line)["product"]


def load_sampling_pool(documents_file: Path, max_docs: int | None):
    products = []
    shop2products = defaultdict(list)
    for product in iter_products(documents_file, max_docs):
        if not product.get("product_id") or product.get("price", 0) <= 0:
            continue
        products.append(product)
        shop2products[product.get("shop_id")].append(product)
    return products, shop2products


def largest_remainder_quota(total: int, weights: dict):
    if total <= 0:
        return {key: 0 for key in weights}
    raw = {key: total * weight for key, weight in weights.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    ranked = sorted(weights, key=lambda key: (raw[key] - quotas[key], raw[key]), reverse=True)
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def expand_quota(quotas: dict):
    values = []
    for key, count in quotas.items():
        values.extend([key] * count)
    return values


def build_sampling_specs(total: int, voucher_type_weights: dict | None = None):
    voucher_type_weights = voucher_type_weights or VOUCHER_TYPE_WEIGHTS
    product_counts = expand_quota(largest_remainder_quota(total, PRODUCT_COUNT_WEIGHTS))
    voucher_types = expand_quota(largest_remainder_quota(total, voucher_type_weights))
    discount_types = expand_quota(largest_remainder_quota(total, DISCOUNT_TYPE_WEIGHTS))
    style_buckets = expand_quota(largest_remainder_quota(total, STYLE_BUCKET_WEIGHTS))
    threshold_buckets = expand_quota(
        largest_remainder_quota(total, {k: v[0] for k, v in THRESHOLD_BUCKETS.items()})
    )
    budget_buckets = expand_quota(
        largest_remainder_quota(total, {k: v[0] for k, v in BUDGET_SLACK_BUCKETS.items()})
    )

    random.shuffle(product_counts)
    random.shuffle(voucher_types)
    random.shuffle(discount_types)
    random.shuffle(style_buckets)
    random.shuffle(threshold_buckets)
    random.shuffle(budget_buckets)

    specs = []
    for idx in range(total):
        n_products = product_counts[idx]
        voucher_type = voucher_types[idx]
        if n_products == 1:
            voucher_type = "platform"
        specs.append(
            {
                "sample_id": f"voucher_train_{idx:06d}",
                "n_products": n_products,
                "voucher_type": voucher_type,
                "discount_type": discount_types[idx],
                "threshold_bucket": threshold_buckets[idx],
                "budget_slack_bucket": budget_buckets[idx],
                "style_bucket": style_buckets[idx],
            }
        )
    return rebalance_voucher_types(specs, total, voucher_type_weights)


def rebalance_voucher_types(specs: list[dict], total: int, voucher_type_weights: dict | None = None):
    target = largest_remainder_quota(total, voucher_type_weights or VOUCHER_TYPE_WEIGHTS)
    counts = Counter(spec["voucher_type"] for spec in specs)
    platform_extra = counts["platform"] - target["platform"]
    if platform_extra <= 0:
        return specs

    for spec in specs:
        if platform_extra <= 0:
            break
        if spec["voucher_type"] == "platform" and spec["n_products"] > 1:
            spec["voucher_type"] = "shop"
            platform_extra -= 1
    return specs


def product_snapshot(product: dict):
    return {
        "product_id": product.get("product_id"),
        "shop_id": product.get("shop_id"),
        "title": product.get("title"),
        "price": product.get("price"),
        "sku_options": product.get("sku_options") or {},
        "attributes": product.get("attributes") or {},
        "service": product.get("service") or [],
    }


def sample_products(
    products: list[dict],
    shop2products: dict,
    voucher_type: str,
    n_products: int,
    used_product_ids: set[str],
):
    if voucher_type == "platform":
        candidates = [p for p in products if p["product_id"] not in used_product_ids]
        if len(candidates) < n_products:
            raise RuntimeError("Not enough unused products for platform voucher sampling.")
        return random.sample(candidates, n_products)

    candidate_shops = [
        [p for p in items if p["product_id"] not in used_product_ids]
        for items in shop2products.values()
    ]
    candidate_shops = [items for items in candidate_shops if len(items) >= n_products]
    if not candidate_shops:
        raise RuntimeError("No shop has enough unused products for shop voucher sampling.")
    return random.sample(random.choice(candidate_shops), n_products)


def sample_threshold(total_price: float, bucket: str):
    _, low, high = THRESHOLD_BUCKETS[bucket]
    lower = max(1, int(math.floor(total_price * low)))
    upper = max(lower, int(math.floor(total_price * high)))
    threshold = random.randint(lower, upper)
    return min(threshold, int(math.floor(total_price)))


def build_voucher(
    products: list[dict],
    voucher_type: str,
    discount_type: str,
    threshold_bucket: str,
    budget_slack_bucket: str,
):
    total_price = sum(float(product["price"]) for product in products)
    if total_price < 100:
        return None

    threshold = sample_threshold(total_price, threshold_bucket)
    face_value = None
    discount = None
    cap = None

    if discount_type == "fixed":
        low = max(1, int(math.floor(threshold * 0.1)))
        high = max(low, int(math.floor(threshold * 0.5)))
        face_value = random.randint(low, high)
        price_after_voucher = total_price - face_value
    else:
        discount_percent = random.randint(10, 50)
        minimum_cap = int(math.floor(threshold * discount_percent / 100.0))
        if minimum_cap >= threshold:
            return None
        cap = random.randint(minimum_cap + 1, threshold)
        discount = discount_percent / 100.0
        price_after_voucher = max(total_price * (1 - discount), total_price - cap)

    _, low_slack, high_slack = BUDGET_SLACK_BUCKETS[budget_slack_bucket]
    budget = math.ceil(price_after_voucher * random.uniform(low_slack, high_slack))
    if budget < price_after_voucher:
        budget = math.ceil(price_after_voucher)

    return {
        "voucher_type": voucher_type,
        "threshold": threshold,
        "discount_type": discount_type,
        "face_value": face_value,
        "discount": discount,
        "cap": cap,
        "price_after_voucher": price_after_voucher,
        "budget": budget,
    }


def describe_voucher(voucher: dict):
    lines = [
        "1. The voucher only applies to the products from the same shop."
        if voucher["voucher_type"] == "shop"
        else "1. The voucher applies to all products.",
        f"2. It is valid only when the total price of the products exceeds `{voucher['threshold']}`.",
    ]
    if voucher["discount_type"] == "fixed":
        lines.append(f"3. It provides a fixed discount of `{voucher['face_value']}`.")
    else:
        percent = voucher["discount"] * 100
        lines.append(
            f"3. It provides a percentage discount of `{percent:.1f}%` "
            f"with a cap of `{voucher['cap']}`."
        )
    return "\n".join(lines)


def candidate_field_groups(product: dict):
    groups = []
    for option in (product.get("sku_options") or {}).values():
        if option:
            requirements = [
                (key, value, f"The `{key}` is `{value}`.")
                for key, value in option.items()
                if key and value is not None
            ]
            if requirements:
                groups.append(("sku_options", option, requirements))

    for key, values in (product.get("attributes") or {}).items():
        if not key or not values:
            continue
        selected = random.sample(values, 1)
        groups.append(
            (
                "attributes",
                {key: selected},
                [(key, selected, f"The `{key}` is `{', '.join(selected)}`.")],
            )
        )

    for service in product.get("service") or []:
        text = SERVICE_TEXT.get(service, "")
        if text:
            groups.append(("service", service, [(service, service, text)]))
    random.shuffle(groups)
    return groups


def sample_target_field_count():
    roll = random.random()
    if roll < 0.10:
        return random.randint(5, 7), "hard"
    if roll < 0.35:
        return 2, "light"
    return random.randint(3, 4), "standard"


def sample_product_fields(product: dict):
    title = product.get("title")
    reward = {"product_id": product["product_id"], "title": [title]}
    requirements = [f"1. The `title` is `{title}`."]
    target_count, field_bucket = sample_target_field_count()
    field_count = 1
    selected_keys = set()

    for field_type, reward_value, requirement_items in candidate_field_groups(product):
        if field_count >= target_count:
            break
        group_keys = {str(key) for key, _, _ in requirement_items}
        if selected_keys.intersection(group_keys):
            continue
        if field_type == "sku_options":
            reward["sku_options"] = [reward_value]
        elif field_type == "attributes":
            reward.setdefault("attributes", []).append(reward_value)
        elif field_type == "service":
            reward.setdefault("service", []).append(reward_value)
        selected_keys.update(group_keys)
        for _, _, text in requirement_items:
            requirements.append(f"{len(requirements) + 1}. {text}")
            field_count += 1

    return reward, requirements, {"target_field_count": target_count, "field_bucket": field_bucket}


def build_prompt(prompt_template: str, requirement_list: list[list[str]], style_bucket: str):
    if len(requirement_list) == 1:
        blocks = ["\n".join(requirement_list[0])]
    else:
        blocks = [
            f"## Product {idx}\n" + "\n".join(requirements)
            for idx, requirements in enumerate(requirement_list, start=1)
        ]
    prompt = (
        prompt_template.replace("<|task|>", "one or more products")
        .replace("<|requirements|>", "\n\n".join(blocks))
    )
    return (
        f"{prompt}\n\n# Query Style\n"
        f"{STYLE_INSTRUCTIONS[style_bucket]} Do not mention product prices."
    )


def assign_opener_buckets(rows: list[dict], seed: int):
    openers = expand_quota(largest_remainder_quota(len(rows), OPENER_BUCKET_WEIGHTS))
    rng = random.Random(seed)
    rng.shuffle(openers)
    return {row["sample_id"]: opener for row, opener in zip(rows, openers)}


def stage3_prompt(row: dict, opener_bucket: str):
    style_bucket = row.get("sampling_buckets", {}).get("style_bucket", "conversational")
    style_instruction = STYLE_INSTRUCTIONS.get(style_bucket, STYLE_INSTRUCTIONS["conversational"])
    opener_instruction = OPENER_INSTRUCTIONS[opener_bucket]
    return (
        f"{row['prompt']}\n\n# Stage III Query Control\n"
        f"{style_instruction}\n"
        f"{opener_instruction}.\n"
        "Do not start with a generic phrase like \"I need a few different items\" unless it is "
        "required by the opener above. Do not mention product prices. Return only the JSON object."
    )


def build_plan_item(
    spec: dict,
    products: list[dict],
    prompt_template: str,
    voucher: dict,
):
    reward = []
    requirement_list = []
    field_buckets = []
    for product in products:
        product_reward, requirements, field_meta = sample_product_fields(product)
        reward.append(product_reward)
        requirement_list.append(requirements)
        field_buckets.append(field_meta)

    prompt = build_prompt(prompt_template, requirement_list, spec["style_bucket"])
    external = (
        f"My budget is only `{voucher['budget']}`, but I have a voucher "
        f"with the following rules:\n{describe_voucher(voucher)}"
    )
    return {
        "sample_id": spec["sample_id"],
        "intent": "Coupon & Budget",
        "sampled_products": [product_snapshot(product) for product in products],
        "sampled_product_ids": [product["product_id"] for product in products],
        "sampled_shop_ids": sorted({str(product.get("shop_id")) for product in products}),
        "total_price_before_voucher": sum(float(product["price"]) for product in products),
        "requirements": requirement_list,
        "prompt": prompt,
        "external_budget_and_voucher": external,
        "reward": reward,
        "voucher": voucher,
        "sampling_buckets": {
            "n_products": spec["n_products"],
            "voucher_type": spec["voucher_type"],
            "discount_type": spec["discount_type"],
            "threshold_bucket": spec["threshold_bucket"],
            "budget_slack_bucket": spec["budget_slack_bucket"],
            "style_bucket": spec["style_bucket"],
            "field_buckets": field_buckets,
            "title_only_product_count": sum(1 for requirements in requirement_list if len(requirements) == 1),
        },
    }


def validate_voucher(voucher: dict):
    if set(voucher) != VOUCHER_KEYS:
        raise ValueError(f"Voucher keys mismatch: {sorted(voucher)}")
    if voucher["voucher_type"] not in {"platform", "shop"}:
        raise ValueError("Invalid voucher_type.")
    if voucher["discount_type"] not in {"fixed", "percentage"}:
        raise ValueError("Invalid discount_type.")
    if voucher["price_after_voucher"] > voucher["budget"]:
        raise ValueError("price_after_voucher exceeds budget.")
    if voucher["discount_type"] == "fixed" and voucher["face_value"] is None:
        raise ValueError("Fixed voucher is missing face_value.")
    if voucher["discount_type"] == "percentage" and (
        voucher["discount"] is None or voucher["cap"] is None
    ):
        raise ValueError("Percentage voucher is missing discount or cap.")


def validate_plan_item(item: dict, used_product_ids: set[str] | None = None):
    required = {
        "sample_id",
        "intent",
        "sampled_products",
        "sampled_product_ids",
        "sampled_shop_ids",
        "total_price_before_voucher",
        "requirements",
        "prompt",
        "external_budget_and_voucher",
        "reward",
        "voucher",
        "sampling_buckets",
    }
    if set(item) != required:
        raise ValueError(f"Plan keys mismatch for {item.get('sample_id')}.")
    if not item["sample_id"] or not item["prompt"]:
        raise ValueError("Plan item missing sample_id or prompt.")
    validate_voucher(item["voucher"])
    if len(item["reward"]) != len(item["sampled_product_ids"]):
        raise ValueError("Reward/product count mismatch.")
    if len(item["requirements"]) != len(item["reward"]):
        raise ValueError("Requirement/product count mismatch.")
    for reward in item["reward"]:
        if not reward.get("product_id") or not reward.get("title"):
            raise ValueError("Each reward must include product_id and title.")
        if "price" in reward:
            raise ValueError("Coupon & Budget rewards must not expose price.")
    if item["voucher"]["voucher_type"] == "shop" and len(item["sampled_shop_ids"]) != 1:
        raise ValueError("Shop voucher sampled products from multiple shops.")
    if used_product_ids is not None:
        overlap = used_product_ids.intersection(item["sampled_product_ids"])
        if overlap:
            raise ValueError(f"Duplicate sampled product IDs: {sorted(overlap)[:3]}")
        used_product_ids.update(item["sampled_product_ids"])


def validate_final_item(item: dict):
    if set(item) != {"query", "reward", "voucher"}:
        raise ValueError("Final item must have exactly query, reward, voucher.")
    if not isinstance(item["query"], str) or not item["query"].strip():
        raise ValueError("Final query is empty.")
    if "My budget is only" not in item["query"] or "voucher with the following rules" not in item["query"]:
        raise ValueError("Final query is missing budget/voucher suffix.")
    if not isinstance(item["reward"], list) or not item["reward"]:
        raise ValueError("Final reward must be a non-empty list.")
    for reward in item["reward"]:
        if not reward.get("product_id") or not reward.get("title"):
            raise ValueError("Each final reward must include product_id and title.")
        if "price" in reward:
            raise ValueError("Coupon & Budget final rewards must not expose price.")
    validate_voucher(item["voucher"])


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_plan(args):
    random.seed(args.seed)
    documents_file = Path(args.documents_file)
    if not documents_file.is_file():
        raise FileNotFoundError(
            f"Documents file not found: {documents_file}. "
            "If needed, run: gunzip -c documents.jsonl.gz > resources/documents.jsonl"
        )
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    products, shop2products = load_sampling_pool(documents_file, args.max_docs)
    if len(products) < 4:
        raise ValueError("Not enough products loaded for voucher sampling.")

    specs = build_sampling_specs(args.total, parse_voucher_type_weights(args.voucher_type_weights))
    total_sampled_products = sum(spec["n_products"] for spec in specs)
    title_only_limit = math.ceil(total_sampled_products * args.max_title_only_product_ratio)
    title_only_count = 0
    used_product_ids = set()
    rows = []
    attempts = 0
    pbar = tqdm(total=args.total, desc="Stage I/II plan")
    for spec in specs:
        accepted = None
        for _ in range(args.max_attempts_per_item):
            attempts += 1
            selected = sample_products(
                products,
                shop2products,
                spec["voucher_type"],
                spec["n_products"],
                used_product_ids,
            )
            voucher = build_voucher(
                selected,
                spec["voucher_type"],
                spec["discount_type"],
                spec["threshold_bucket"],
                spec["budget_slack_bucket"],
            )
            if not voucher:
                continue
            accepted = build_plan_item(spec, selected, prompt_template, voucher)
            validate_plan_item(accepted)
            accepted_title_only = accepted["sampling_buckets"]["title_only_product_count"]
            if title_only_count + accepted_title_only > title_only_limit:
                continue
            break
        if not accepted:
            raise RuntimeError(f"Failed to sample {spec['sample_id']} after retries.")
        used_product_ids.update(accepted["sampled_product_ids"])
        title_only_count += accepted["sampling_buckets"]["title_only_product_count"]
        rows.append(accepted)
        pbar.update(1)
    pbar.close()

    validate_plan(rows)
    write_jsonl(Path(args.plan_output), rows)
    print(
        f"Wrote {len(rows)} plan rows to {args.plan_output} after {attempts} attempts. "
        f"title-only products: {title_only_count}/{total_sampled_products}."
    )


def validate_plan(rows: list[dict]):
    used = set()
    for row in rows:
        validate_plan_item(row, used)


def parse_voucher_type_weights(raw: str):
    weights = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Invalid voucher type weight item: {part!r}")
        key = key.strip()
        if key not in {"platform", "shop"}:
            raise ValueError(f"Invalid voucher type: {key!r}")
        weights[key] = float(value)
    if set(weights) != {"platform", "shop"}:
        raise ValueError("--voucher-type-weights must include platform and shop.")
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("--voucher-type-weights values must be non-negative and sum positive.")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completion_url(base_url: str):
    return base_url.rstrip("/") + "/chat/completions"


def should_bypass_env_proxy(url: str):
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} or "mimo" in hostname


def build_request_body(prompt: str, model_config: dict):
    config = copy.deepcopy(model_config)
    extra_body = config.pop("extra_body", {}) or {}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        **config,
        **extra_body,
    }
    return body


def http_chat_completion(prompt: str, model_config: dict, base_url: str, api_key: str, timeout: int):
    body = json.dumps(build_request_body(prompt, model_config), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        completion_url(base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    if should_bypass_env_proxy(base_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response_context = opener.open(request, timeout=timeout)
    else:
        response_context = urllib.request.urlopen(request, timeout=timeout)

    with response_context as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content") or ""


def parse_llm_query(content: str):
    match = re.search(r"```json\s*(.+?)\s*```", content, re.DOTALL)
    raw = match.group(1).strip() if match else content.strip()
    try:
        return json.loads(raw).get("query")
    except ValueError:
        return None


def generate_query(prompt: str, external: str, model_config: dict, base_url: str, api_key: str, timeout: int, retries: int):
    if not api_key:
        raise ValueError("Missing API key. Set the environment variable selected by --api-key-env.")

    for attempt in range(1, retries + 1):
        try:
            content = http_chat_completion(prompt, model_config, base_url, api_key, timeout)
            query = parse_llm_query(content)
            if query and query.strip():
                query = query.strip()
                if query == external or len(query) < 10:
                    return None
                return f"{query}\n\n{external}"
            return None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
            if attempt >= retries:
                print(f"LLM request failed after {retries} attempts: {exc}", file=sys.stderr)
                return None
            time.sleep(2 * attempt)
    return None


def load_completed_sample_ids(output: Path, metadata_output: Path, plan_rows: list[dict]):
    if metadata_output.is_file():
        completed = set()
        for row in load_jsonl(metadata_output):
            if row.get("sample_id"):
                completed.add(row["sample_id"])
        return completed
    if output.is_file():
        count = sum(1 for line in output.open("r", encoding="utf-8") if line.strip())
        return {row["sample_id"] for row in plan_rows[:count]}
    return set()


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()


def generate_one_from_plan(row: dict, opener_bucket: str, model_config: dict, args):
    query = generate_query(
        stage3_prompt(row, opener_bucket),
        row["external_budget_and_voucher"],
        model_config,
        args.base_url,
        os.environ.get(args.api_key_env),
        args.request_timeout,
        args.llm_retries,
    )
    if not query:
        return row["sample_id"], None, None
    final = {"query": query, "reward": row["reward"], "voucher": row["voucher"]}
    validate_final_item(final)
    metadata = {
        "sample_id": row["sample_id"],
        "style_bucket": row.get("sampling_buckets", {}).get("style_bucket"),
        "opener_bucket": opener_bucket,
    }
    return row["sample_id"], final, metadata


def generate_from_plan(args):
    plan_path = Path(args.plan_output)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    rows = load_jsonl(plan_path)
    validate_plan(rows)

    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    if args.model_config:
        model_config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    model_config["model"] = args.model
    model_config["max_completion_tokens"] = args.max_completion_tokens
    model_config.setdefault("temperature", args.temperature)
    model_config.setdefault("top_p", args.top_p)
    model_config.setdefault("extra_body", {"thinking": {"type": "disabled"}})

    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else Path(str(output) + ".meta.jsonl")
    completed = load_completed_sample_ids(output, metadata_output, rows)
    pending = [row for row in rows if row["sample_id"] not in completed]
    opener_buckets = assign_opener_buckets(rows, args.opener_seed)
    print(f"Stage III pending rows: {len(pending)} / {len(rows)}")

    failures = []
    pbar = tqdm(total=len(pending), desc="Stage III LLM")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(generate_one_from_plan, row, opener_buckets[row["sample_id"]], model_config, args)
            for row in pending
        ]
        for future in as_completed(futures):
            sample_id, final, metadata = future.result()
            if final is None:
                failures.append(sample_id)
            else:
                append_jsonl(output, final)
                append_jsonl(metadata_output, metadata)
            pbar.update(1)
    pbar.close()

    if failures:
        raise RuntimeError(
            f"Stage III failed for {len(failures)} samples. "
            f"First failures: {', '.join(failures[:5])}"
        )
    print(f"Wrote final rows to {output}")


def summarize_jsonl(path: Path, is_plan: bool):
    rows = load_jsonl(path)
    if not rows:
        print(f"{path}: empty")
        return
    if is_plan:
        validate_plan(rows)
        print(f"{path}: {len(rows)} plan rows")
        print("product counts:", dict(Counter(row["sampling_buckets"]["n_products"] for row in rows)))
        print("voucher types:", dict(Counter(row["voucher"]["voucher_type"] for row in rows)))
        print("discount types:", dict(Counter(row["voucher"]["discount_type"] for row in rows)))
        print("style buckets:", dict(Counter(row["sampling_buckets"].get("style_bucket") for row in rows)))
        first = rows[0]
        preview = {
            "sample_id": first["sample_id"],
            "sampled_product_ids": first["sampled_product_ids"],
            "requirements": first["requirements"],
            "voucher": first["voucher"],
            "prompt": first["prompt"][:500],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            validate_final_item(row)
        print(f"{path}: {len(rows)} final rows")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])


def parse_args():
    parser = argparse.ArgumentParser(
        description="ShoppingBench Coupon & Budget stage1/2 plan and stage3 query generation."
    )
    parser.add_argument("--stage", choices=["plan", "generate", "both", "summarize-plan", "summarize-final"], default="both")
    parser.add_argument("--documents-file", default=str(ROOT / "resources" / "documents.jsonl"))
    parser.add_argument("--prompt-file", default=str(ROOT / "src" / "agent" / "prompt" / "synthesize.md"))
    parser.add_argument("--plan-output", default=str(ROOT / "data" / "synthesize_voucher_train_plan.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "synthesize_voucher_train.jsonl"))
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--total", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voucher-type-weights", default="platform=0.40,shop=0.60")
    parser.add_argument("--max-docs", type=int, default=100000)
    parser.add_argument("--max-attempts-per-item", type=int, default=1000)
    parser.add_argument("--max-title-only-product-ratio", type=float, default=0.10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--base-url", default="https://token-plan-cn.xiaomimimo.com/v1")
    parser.add_argument("--api-key-env", default="MIMO_API_KEY")
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--opener-seed", type=int, default=20260614)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "plan":
        generate_plan(args)
    elif args.stage == "generate":
        generate_from_plan(args)
    elif args.stage == "both":
        generate_plan(args)
        generate_from_plan(args)
    elif args.stage == "summarize-plan":
        summarize_jsonl(Path(args.plan_output), is_plan=True)
    elif args.stage == "summarize-final":
        summarize_jsonl(Path(args.output), is_plan=False)


if __name__ == "__main__":
    main()
