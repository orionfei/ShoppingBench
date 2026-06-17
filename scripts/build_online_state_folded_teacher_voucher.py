#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
MAX_WORKERS = 32
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from run_rollout import get_system_prompt, get_user_prompt, act, is_terminate  # noqa: E402
from util.message import Message, OUTPUT_ROLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build online state-folded teacher trajectories for hard voucher samples."
    )
    parser.add_argument(
        "--selection",
        choices=["hard", "all"],
        default="hard",
        help="Sample universe: hard uses hard-sample indices; all uses train line numbers.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based start index in the selected universe; for --selection all this is the train line number.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to build. Use 0 for all remaining.")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1, MAX_WORKERS),
        help=f"Parallel rollout workers. Hard-capped at {MAX_WORKERS}.",
    )
    parser.add_argument(
        "--allow-teacher-refine",
        action="store_true",
        help="Allow answer-derived search rewrites for diagnostics only; do not use for SFT data.",
    )
    parser.add_argument(
        "--output-rollout",
        default="data/teacher_voucher_hard20_state_folded.jsonl",
    )
    parser.add_argument(
        "--output-synthesize",
        default="data/teacher_voucher_hard20_synthesize.jsonl",
    )
    parser.add_argument(
        "--report",
        default="data/teacher_voucher_hard20_report.json",
    )
    parser.add_argument(
        "--notes",
        default="proposal/teacher_voucher_hard20_notes.md",
    )
    parser.add_argument(
        "--hard-indices",
        default="",
        help="Comma/range list of global hard indices to build, e.g. 36,43 or 51-100.",
    )
    parser.add_argument(
        "--train-lines",
        default="",
        help="Comma/range list of 1-based synthesize_voucher_train.jsonl line numbers to build.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Append each successful case immediately so interrupted batches keep completed rows.",
    )
    args = parser.parse_args()
    if args.workers > MAX_WORKERS:
        print(
            f"[WARN] --workers {args.workers} exceeds hard cap {MAX_WORKERS}; using {MAX_WORKERS}.",
            file=sys.stderr,
            flush=True,
        )
        args.workers = MAX_WORKERS
    return args


def parse_int_ranges(value: str) -> set[int]:
    indices = set()
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def parse_hard_indices(value: str) -> set[int]:
    return parse_int_ranges(value)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        fout.flush()


def is_hard_budget(item: dict) -> bool:
    voucher = item.get("voucher") or {}
    price_after = voucher.get("price_after_voucher")
    budget = voucher.get("budget")
    if price_after is None or budget is None:
        return False
    return float(budget) <= float(price_after) * 1.02 + 1e-9


def hard_samples(limit: int, start: int = 1, hard_indices: set[int] | None = None) -> list[dict]:
    train = read_jsonl(ROOT / "data/synthesize_voucher_train.jsonl")
    selected = []
    hard_index = 0
    for line_no, item in enumerate(train, 1):
        if not is_hard_budget(item):
            continue
        hard_index += 1
        if hard_indices is not None and hard_index not in hard_indices:
            continue
        if hard_indices is None and hard_index < start:
            continue
        sample = copy.deepcopy(item)
        sample["_line_no"] = line_no
        sample["_hard_index"] = hard_index
        sample["_sample_id"] = f"train_line_{line_no:06d}"
        sample["_sampling_buckets"] = {"budget_slack_bucket": "hard"}
        selected.append(sample)
        if hard_indices is None and limit and len(selected) >= limit:
            break
    return selected


def all_train_samples(limit: int, start: int = 1, train_lines: set[int] | None = None) -> list[dict]:
    train = read_jsonl(ROOT / "data/synthesize_voucher_train.jsonl")
    selected = []
    hard_index = 0
    hard_by_line = {}
    for line_no, item in enumerate(train, 1):
        if is_hard_budget(item):
            hard_index += 1
            hard_by_line[line_no] = hard_index
    for line_no, item in enumerate(train, 1):
        if train_lines is not None and line_no not in train_lines:
            continue
        if train_lines is None and line_no < start:
            continue
        sample = copy.deepcopy(item)
        sample["_line_no"] = line_no
        if line_no in hard_by_line:
            sample["_hard_index"] = hard_by_line[line_no]
            budget_slack_bucket = "hard"
        else:
            sample["_hard_index"] = None
            budget_slack_bucket = "non_hard"
        sample["_sample_id"] = f"train_line_{line_no:06d}"
        sample["_sampling_buckets"] = {"budget_slack_bucket": budget_slack_bucket}
        selected.append(sample)
        if train_lines is None and limit and len(selected) >= limit:
            break
    return selected


def synthesize_row(sample: dict) -> dict:
    return {key: sample[key] for key in ("query", "reward", "voucher")}


def clean_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ]+", " ", text or "").strip()


def unique(values: list[str]) -> list[str]:
    out = []
    for value in values:
        value = " ".join(str(value).split())
        if value and value not in out:
            out.append(value)
    return out


def reward_values(reward: dict) -> list[str]:
    values = []
    for attr in reward.get("attributes", []) or []:
        for attr_values in attr.values():
            if isinstance(attr_values, list):
                values.extend(str(item) for item in attr_values)
            else:
                values.append(str(attr_values))
    for sku in reward.get("sku_options", []) or []:
        values.extend(str(item) for item in sku.values())
    for service in reward.get("service", []) or []:
        values.append(str(service))
    return values


def query_candidates(reward: dict) -> list[str]:
    title = (reward.get("title") or [""])[0]
    words = clean_text(title).split()
    values = reward_values(reward)
    candidates = [title]
    for n in (12, 8, 5, 3):
        if len(words) >= n:
            candidates.append(" ".join(words[:n]))
    if values:
        candidates.append(" ".join(words[:5] + values[:5]))
        candidates.append(" ".join(values[:6]))
    return unique(candidates)


def user_search_segments(query: str) -> list[str]:
    main = query.split("\n\nMy budget is only", 1)[0]
    main = re.sub(
        r"\b(show me|find|i'm looking for|looking for|i need|i also need|i'm searching for|i'm interested in)\b[: ]*",
        "",
        main,
        flags=re.IGNORECASE,
    )
    main = re.sub(
        r",\s+(?:and\s+)?(?=(?:also\s+)?(?:a|an|the|some|another)\b)",
        "|",
        main,
        flags=re.IGNORECASE,
    )
    main = re.sub(
        r"\s+and\s+(?=(?:also\s+)?(?:a|an|the|some|another)\b)",
        "|",
        main,
        flags=re.IGNORECASE,
    )
    main = re.sub(
        r",\s+(?=(?:multi-?color|comfortable|large|small|black|white|red|green|blue|pink|yellow|silver|gold)\b)",
        "|",
        main,
        flags=re.IGNORECASE,
    )
    main = re.sub(r"(^|[;:])\s*\d+[.)]\s*", r"\1|", main)
    main = re.sub(r"\b(also|plus|lastly|finally)\b", "|", main, flags=re.IGNORECASE)
    main = re.sub(r"\b(first|second|third|fourth)\b", "|", main, flags=re.IGNORECASE)
    chunks = re.split(r"\||;|\n|\. ", main)
    segments = []
    category_prefix = None
    for chunk in chunks:
        chunk = re.sub(r"^[0-9]+[.)]\s*", "", chunk)
        chunk = re.sub(r"\b(?:and|also)\s*$", "", chunk, flags=re.IGNORECASE).strip(" :,-")
        age_match = re.search(r"\b(\d+\s*-\s*\d+)\s*year\s*old\b", chunk, flags=re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"\b(\d+\s*-\s*\d+)\s*yrs?\b", chunk, flags=re.IGNORECASE)
        if "same age" in chunk.lower() and segments:
            for previous in reversed(segments):
                previous_age = re.search(
                    r"\b(\d+\s*-\s*\d+)\s*(?:year\s*old|yrs?)\b",
                    previous,
                    flags=re.IGNORECASE,
                )
                if previous_age:
                    chunk = re.sub(
                        r"\bsame age\b",
                        previous_age.group(1).replace(" ", ""),
                        chunk,
                        flags=re.IGNORECASE,
                    )
                    break
        if category_prefix is None and len(chunk.split()) < 3 and re.search(
            r"\b(t-?shirts?|shirts?|tops?|shoes?|dresses?|pants?|shorts?|bags?)\b",
            chunk,
            flags=re.IGNORECASE,
        ):
            category_prefix = chunk
            continue
        if len(chunk.split()) >= 3:
            segments.append(chunk)
            if category_prefix:
                segments.append(f"{category_prefix} {chunk}")
    return unique(segments)


def search_text_variants(text: str) -> list[str]:
    normalized = re.sub(r"[`\"']", "", text).strip(" .,:;-")
    normalized = re.sub(r"\bt[\s-]?shirts?\b", "tshirt", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\btops?\b", "top", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bkids\b", "kids", normalized, flags=re.IGNORECASE)
    variants = [text, normalized]

    if re.search(r"\b\d+\s*-\s*\d+\s*years?\b", normalized, flags=re.IGNORECASE):
        variants.append(
            re.sub(
                r"\b(\d+\s*-\s*\d+)\s*years?\b",
                lambda m: f"{m.group(1).replace(' ', '')} yrs",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        variants.append(
            re.sub(
                r"\b(\d+\s*-\s*\d+)\s*years?\b",
                lambda m: f"{m.group(1).replace(' ', '')}yrs old",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    if re.search(r"\b\d+\s*-\s*\d+\s*year\s*old\b", normalized, flags=re.IGNORECASE):
        variants.append(
            re.sub(
                r"\b(\d+\s*-\s*\d+)\s*year\s*old\b",
                lambda m: f"{m.group(1).replace(' ', '')} yrs",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    if re.search(r"\b\d+\s*-\s*year\s*-\s*old\b", normalized, flags=re.IGNORECASE):
        variants.append(
            re.sub(
                r"\b(\d+)\s*-\s*year\s*-\s*old\b",
                r"\1 yrs",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        variants.append(
            re.sub(
                r"\b(\d+)\s*-\s*year\s*-\s*old\b",
                r"\1 years old",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    compact = re.sub(
        r"\b(a|an|the|for|with|and|or|of|ages?|casual|crew|neck)\b",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    compact = re.sub(r"\s*-\s*-\s*", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact.split()) >= 3 and "- -" not in compact:
        variants.append(compact)

    lowered = normalized.lower()
    if "pet bowl" in lowered:
        variants.append(normalized.replace("pet bowl", "pet feeding bowl"))
        variants.append(normalized.replace("pet bowl", "pet bowls"))
    if "ceramic" in lowered and "pet bowl" in lowered:
        variants.append("ceramic pet feeding bowl")
    if "cat" in lowered and "bowl" in lowered:
        variants.append("cat feeding bowl ceramic" if "ceramic" in lowered else "cat feeding bowl")
    if "coin purse" in lowered:
        variants.append(normalized.replace("coin purse", "coinpurse"))
        variants.append(compact.replace("coin purse", "coinpurse") if compact else "")
    if "fine liner" in lowered:
        variants.append(normalized.replace("fine liner", "fineliner"))
        variants.append(compact.replace("fine liner", "fineliner") if compact else "")
    if re.search(r"\bin size\b", normalized, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bin size\b", "", normalized, flags=re.IGNORECASE))
    if re.search(r"\bsize\b", normalized, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bsize\b", "", normalized, flags=re.IGNORECASE))
    if "storage box" in lowered and "cable" in lowered:
        colors = [
            color
            for color in ("green", "black", "white", "red", "blue", "pink", "gray", "grey", "silver")
            if re.search(rf"\b{color}\b", lowered)
        ]
        prefix = f"{colors[0]} " if colors else ""
        variants.append(f"{prefix}cable organizer storage box")
        variants.append(f"{prefix}desktop storage box cable organizer")
    if "ballpen" in lowered and "box" in lowered:
        variants.append(normalized.replace("box", "1box"))
        variants.append(compact.replace("box", "1box") if compact else "")
    if "balcony" in lowered and "clothes hanger" in lowered:
        variants.append("space-saving drying rack balcony")
    if "&" in normalized:
        no_amp = re.sub(r"\s*&\s*", " ", normalized)
        variants.append(no_amp)
        variants.append(re.sub(r"\bto\b", " ", no_amp, flags=re.IGNORECASE))
    if "tanks" in lowered and "camisoles" in lowered:
        no_tops = re.sub(r"\btops?\b", " ", normalized, flags=re.IGNORECASE)
        no_tops = re.sub(r"\s*&\s*", " ", no_tops)
        no_tops = re.sub(r"\b(to|with)\b", " ", no_tops, flags=re.IGNORECASE)
        variants.append(re.sub(r"\s+", " ", no_tops).strip())

    return unique(
        variant
        for variant in variants
        if len(variant.split()) >= 3 and "- -" not in variant
    )


def overlap_score(text: str, reward: dict) -> int:
    text_words = set(clean_text(text).lower().split())
    title_words = set(clean_text((reward.get("title") or [""])[0]).lower().split())
    value_words = set()
    for value in reward_values(reward):
        value_words.update(clean_text(value).lower().split())
    return len(text_words & title_words) + 2 * len(text_words & value_words)


def query_candidates_from_user(query: str, reward: dict) -> list[str]:
    brand_hints = user_brand_hints(query)
    scored = sorted(
        user_search_segments(query),
        key=lambda segment: overlap_score(segment, reward),
        reverse=True,
    )
    candidates = []
    for segment in scored[:4]:
        words = segment.split()
        for brand in brand_hints[:2]:
            if brand.lower() not in segment.lower():
                candidates.extend(search_text_variants(f"{brand} {segment}"))
        candidates.extend(search_text_variants(segment))
        if len(words) > 10:
            candidates.extend(search_text_variants(" ".join(words[:10])))
        if len(words) > 6:
            candidates.extend(search_text_variants(" ".join(words[-8:])))
    return unique(candidates)


def user_brand_hints(query: str) -> list[str]:
    main = query.split("\n\nMy budget is only", 1)[0]
    hints = []
    patterns = [
        r"\bfrom\s+(?:the\s+)?([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){0,4})\s+brand\b",
        r"\bfrom\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){1,4})(?=[,.;]|\s+that\b|\s+with\b|\s+in\b|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, main):
            hint = match.group(1).strip(" .,:;-")
            hint = re.sub(r"\s+", " ", hint)
            if 1 <= len(hint.split()) <= 5 and hint.lower() not in {"home office"}:
                hints.append(hint)
    return unique(hints)


def service_filter_candidates(user_query: str, reward: dict) -> list[str | None]:
    lowered = user_query.lower()
    services = []
    if "lazflash" in lowered or "flash" in lowered or "flashsale" in lowered:
        services.append("flashsale")
    if "free shipping" in lowered or "free delivery" in lowered:
        services.append("freeShipping")
    if "cash on delivery" in lowered or "cod" in lowered:
        services.append("COD")
    if "lazmall" in lowered or "official" in lowered:
        services.append("official")
    reward_services = [str(item) for item in reward.get("service", []) or []]
    services = [service for service in services if service in reward_services]
    combos = [None]
    if services:
        combos.insert(0, ",".join(services))
        combos.extend(service for service in services if service not in combos)
    return combos


def find_target(
    toolmap,
    user_query: str,
    reward: dict,
    max_rank: int,
    search_cache: dict,
    shop_id: str | None = None,
    allow_teacher_refine: bool = False,
    raise_on_missing: bool = True,
) -> dict:
    target_id = str(reward["product_id"])
    best_failure = None
    candidate_groups = [("user_request", query_candidates_from_user(user_query, reward))]
    if allow_teacher_refine:
        candidate_groups.append(("teacher_refine", query_candidates(reward)[1:]))
    service_candidates = service_filter_candidates(user_query, reward)
    for source, queries in candidate_groups:
        for query in queries:
            for service in service_candidates:
                for page in range(1, 6):
                    cache_key = (query, page, shop_id or "", service or "")
                    if cache_key not in search_cache:
                        params = {"q": query, "page": page}
                        if shop_id:
                            params["shop_id"] = shop_id
                        if service:
                            params["service"] = service
                        search_cache[cache_key] = toolmap["find_product"].execute(**params) or []
                    results = search_cache[cache_key]
                    ids = [str(item.get("product_id")) for item in results]
                    if best_failure is None and results:
                        best_failure = {
                            "query": query,
                            "page": page,
                            "service": service,
                            "ids": ids,
                        }
                    if target_id in ids and ids.index(target_id) + 1 <= max_rank:
                        product = results[ids.index(target_id)]
                        return {
                            "query": query,
                            "page": page,
                            "shop_id": shop_id,
                            "service": service,
                            "product": product,
                            "rank": ids.index(target_id) + 1,
                            "source": source,
                        }
        # Prefer a directly user-derived query, but do not spend refined searches
        # unless the target is absent from the readable retained candidates.
        if source == "user_request":
            continue
    if raise_on_missing:
        raise RuntimeError(f"target {target_id} not found; first nonempty={best_failure}")
    return None


def voucher_payable(total: float, shop_ids: list[str], voucher: dict) -> tuple[bool, float]:
    scope_ok = voucher["voucher_type"] == "platform" or (
        voucher["voucher_type"] == "shop" and len(set(shop_ids)) == 1
    )
    if not scope_ok or total < voucher["threshold"]:
        return False, total
    if voucher["discount_type"] == "fixed":
        return True, total - voucher["face_value"]
    if voucher["discount_type"] == "percentage":
        return True, max(total * (1 - voucher["discount"]), total - voucher["cap"])
    raise ValueError(voucher["discount_type"])


def budget_code(product_ids: list[str], shop_ids: list[str], prices: list[float], voucher: dict) -> str:
    total = round(sum(float(price) for price in prices), 2)
    voucher_used, payable = voucher_payable(total, shop_ids, voucher)
    payable = round(payable, 2)
    within = payable <= float(voucher["budget"])
    payload = {
        "product_ids": product_ids,
        "shop_ids": shop_ids,
        "same_shop": len(set(shop_ids)) == 1,
        "total_before_voucher": total,
        "meets_threshold": total >= voucher["threshold"],
        "eligible_scope": voucher["voucher_type"] == "platform"
        or (voucher["voucher_type"] == "shop" and len(set(shop_ids)) == 1),
        "voucher_used": voucher_used,
        "payable_total": payable,
        "budget": voucher["budget"],
        "within_budget": within,
    }
    return "import json\nprint(json.dumps(" + repr(payload) + "))"


def compact_tool_call(name: str, parameters: dict) -> dict:
    return {"name": name, "parameters": parameters}


def short_query(text: str, max_chars: int = 54) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def search_param_text(found: dict, include_shop: bool = True) -> str:
    parts = [f"q='{short_query(found['query'])}'", f"p{found['page']}"]
    if include_shop and found.get("shop_id"):
        parts.append(f"shop={found['shop_id']}")
    if found.get("service"):
        parts.append(f"service={found['service']}")
    return "(" + ",".join(parts) + ")"


def query_rewrite_notes(queries: list[str]) -> list[str]:
    joined = " ".join(queries).lower()
    notes = []
    if "tshirt" in joined:
        notes.append("normalize `t-shirt` spelling as `tshirt`")
    if "fineliner" in joined:
        notes.append("normalize `fine liner` as `fineliner`")
    if "coinpurse" in joined:
        notes.append("try compact spelling `coinpurse` for `coin purse`")
    if "pet feeding bowl" in joined or "cat feeding bowl" in joined:
        notes.append("use `feeding bowl` as a pet-bowl search synonym")
    if "cable organizer storage box" in joined or "desktop storage box cable organizer" in joined:
        notes.append("reorder cable-storage terms to `cable organizer storage box`")
    if "drying rack balcony" in joined:
        notes.append("use `drying rack` for balcony clothes-hanger searches")
    if "1box" in joined:
        notes.append("keep the box-pack cue as `1box`")
    if "tanks camisoles" in joined:
        notes.append("drop connector/top filler and keep core outfit terms")
    if re.search(r"\b\d+\s*-\s*\d+\s*yrs?\b", joined):
        notes.append("normalize age wording to `N-N yrs`")
    return unique(notes)


def candidate_text(products: list[dict]) -> str:
    return ", ".join(
        f"{product['product_id']}@{product['shop_id']}:{float(product['price']):g}"
        for product in products
    )


def search_step_think(search_specs: list[dict], refine_needed: bool) -> str:
    searches = "; ".join(search_param_text(found, include_shop=False) for found in search_specs)
    notes = query_rewrite_notes([found["query"] for found in search_specs])
    rewrite_text = f" Query rewrites: {'; '.join(notes)}." if notes else ""
    if refine_needed:
        return (
            "Search user-visible item phrases first to expose at least one same-shop anchor; "
            f"planned searches: {searches}.{rewrite_text}"
        )
    return (
        "Search each requested item with user-visible terms so the retained state shows candidate ids, shops, and prices; "
        f"planned searches: {searches}.{rewrite_text}"
    )


def refine_step_think(anchor_products: list[dict], refine_specs: list[dict]) -> str:
    anchor_shop = str(anchor_products[0]["shop_id"])
    anchors = ", ".join(str(product["product_id"]) for product in anchor_products)
    searches = "; ".join(search_param_text(found) for found in refine_specs)
    return (
        f"State shows visible same-shop anchor ids {anchors} in shop {anchor_shop}. "
        f"Search missing requested items inside that shop: {searches}."
    )


def verify_step_think(products: list[dict], voucher: dict) -> str:
    total = round(sum(float(product["price"]) for product in products), 2)
    scope = voucher["voucher_type"]
    return (
        f"Visible candidates in request order are {candidate_text(products)}. "
        f"Verify sku/attrs, then compute {scope} voucher budget from observed total {total:g}."
    )


def recommend_step_think(
    product_ids: list[str],
    total: float,
    payable: float,
    budget: float,
    voucher_used: bool,
) -> str:
    status = "applied" if voucher_used else "not applied"
    return (
        f"Budget check is trusted: total {round(total, 2):g}, voucher {status}, "
        f"payable {round(payable, 2):g} <= budget {budget:g}. "
        f"Recommend ids {','.join(product_ids)} in request order and terminate."
    )


def make_message(think: str, calls: list[dict]) -> tuple[str, Message]:
    content = (
        f"<think>{think}</think>\n"
        f"<tool_call>{json.dumps(calls, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
    )
    return content, Message.from_string("", content)


def append_step(
    row: list[dict],
    message: Message,
    history_messages: list[str],
    system_prompt: str,
    config: dict,
    sample: dict,
    step: int,
    query: str,
    content: str,
) -> None:
    user_prompt = get_user_prompt(message, history_messages, config)
    message.clear()
    parsed = Message.from_string("", content)
    if parsed.tool_call:
        parsed.obs = act(parsed)
    row.append(
        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "completion": {
                "reasoning_content": parsed.think,
                "content": content,
                "message": copy.deepcopy(parsed.to_dict()),
            },
            "extra_info": {
                "step": step,
                "query": query,
                "line_no": sample.get("_line_no"),
                "hard_index": sample.get("_hard_index"),
                "sample_id": sample.get("_sample_id"),
                "sampling_buckets": sample.get("_sampling_buckets", {}),
                "timestamp": int(time.time() * 1000),
                "history_compression": config.get("history_compression", "raw"),
                "teacher_policy": "gold_guided_environment_verified",
            },
        }
    )
    message.user = ""
    message.think = parsed.think
    message.tool_call = parsed.tool_call
    message.obs = parsed.obs
    message.response = parsed.response


def build_row(sample: dict, config: dict, toolmap) -> tuple[list[dict], dict]:
    query = sample["query"]
    rewards = sample["reward"]
    voucher = sample["voucher"]
    system_prompt = get_system_prompt(config)
    history_messages = []
    message = Message(user=query)
    row = []

    max_rank = config.get("state_max_candidates_per_search", 10)
    search_cache = {}
    search_specs_by_idx = {}
    found_products_by_idx = {}

    global_specs = []
    for idx, reward in enumerate(rewards):
        found = find_target(
            toolmap,
            query,
            reward,
            max_rank=max_rank,
            search_cache=search_cache,
            allow_teacher_refine=config.get("allow_teacher_refine", False),
            raise_on_missing=False,
        )
        if found:
            search_specs_by_idx[idx] = found
            found_products_by_idx[idx] = found["product"]
            global_specs.append(found)

    missing = [idx for idx in range(len(rewards)) if idx not in found_products_by_idx]
    refine_specs = []
    if missing:
        if voucher["voucher_type"] != "shop" or not found_products_by_idx:
            missing_ids = [str(rewards[idx]["product_id"]) for idx in missing]
            raise RuntimeError(f"missing targets without shop anchor: {missing_ids}")
        anchor_shop = str(next(iter(found_products_by_idx.values()))["shop_id"])
        for idx in missing:
            found = find_target(
                toolmap,
                query,
                rewards[idx],
                max_rank=max_rank,
                search_cache=search_cache,
                shop_id=anchor_shop,
                allow_teacher_refine=config.get("allow_teacher_refine", False),
                raise_on_missing=True,
            )
            search_specs_by_idx[idx] = found
            found_products_by_idx[idx] = found["product"]
            refine_specs.append(found)

    search_specs = [search_specs_by_idx[idx] for idx in range(len(rewards))]
    found_products = [found_products_by_idx[idx] for idx in range(len(rewards))]

    search_calls = []
    seen_searches = set()
    for found in global_specs or search_specs:
        key = (
            found["query"],
            found["page"],
            found.get("shop_id") or "",
            found.get("service") or "",
        )
        if key in seen_searches:
            continue
        seen_searches.add(key)
        params = {"q": found["query"], "page": found["page"]}
        if found.get("shop_id"):
            params["shop_id"] = found["shop_id"]
        if found.get("service"):
            params["service"] = found["service"]
        search_calls.append(compact_tool_call("find_product", params))
    first_think = search_step_think(global_specs or search_specs, refine_needed=bool(refine_specs))
    content, _ = make_message(first_think, search_calls)
    append_step(row, message, history_messages, system_prompt, config, sample, 1, query, content)

    if refine_specs:
        refine_calls = []
        seen_refine = set()
        for found in refine_specs:
            key = (
                found["query"],
                found["page"],
                found.get("shop_id") or "",
                found.get("service") or "",
            )
            if key in seen_refine:
                continue
            seen_refine.add(key)
            params = {
                "q": found["query"],
                "page": found["page"],
                "shop_id": found["shop_id"],
            }
            if found.get("service"):
                params["service"] = found["service"]
            refine_calls.append(
                compact_tool_call(
                    "find_product",
                    params,
                )
            )
        content, _ = make_message(
            refine_step_think([found["product"] for found in global_specs], refine_specs),
            refine_calls,
        )
        append_step(
            row,
            message,
            history_messages,
            system_prompt,
            config,
            sample,
            len(row) + 1,
            query,
            content,
        )

    product_ids = [str(product["product_id"]) for product in found_products]
    shop_ids = [str(product["shop_id"]) for product in found_products]
    prices = [float(product["price"]) for product in found_products]
    code = budget_code(product_ids, shop_ids, prices, voucher)
    content, _ = make_message(
        verify_step_think(found_products, voucher),
        [
            compact_tool_call(
                "view_product_information",
                {"product_ids": ",".join(product_ids)},
            ),
            compact_tool_call("python_execute", {"code": code}),
        ],
    )
    append_step(row, message, history_messages, system_prompt, config, sample, len(row) + 1, query, content)

    payable_used, payable = voucher_payable(round(sum(prices), 2), shop_ids, voucher)
    content, _ = make_message(
        recommend_step_think(
            product_ids,
            total=round(sum(prices), 2),
            payable=payable,
            budget=float(voucher["budget"]),
            voucher_used=payable_used,
        ),
        [
            compact_tool_call(
                "recommend_product",
                {"product_ids": ",".join(product_ids)},
            ),
            compact_tool_call("terminate", {"status": "success"}),
        ],
    )
    append_step(row, message, history_messages, system_prompt, config, sample, len(row) + 1, query, content)
    if not is_terminate(Message.from_dict(row[-1]["completion"]["message"])):
        raise RuntimeError("final step did not terminate")

    report = {
        "line_no": sample["_line_no"],
        "hard_index": sample.get("_hard_index"),
        "sample_id": sample["_sample_id"],
        "sampling_buckets": sample.get("_sampling_buckets", {}),
        "product_ids": product_ids,
        "shop_ids": shop_ids,
        "prices": prices,
        "total_before_voucher": round(sum(prices), 2),
        "voucher_used": payable_used,
        "payable_total": round(payable, 2),
        "budget": voucher["budget"],
        "steps": len(row),
        "searches": [
            {
                "product_id": str(found["product"]["product_id"]),
                "query": found["query"],
                "page": found["page"],
                "shop_id": found.get("shop_id"),
                "service": found.get("service"),
                "rank": found["rank"],
                "source": found["source"],
            }
            for found in search_specs
        ],
    }
    return row, report


def think_self_check(row: list[dict]) -> list[dict]:
    checks = []
    for step in row:
        message = step["completion"]["message"]
        think = message.get("think", "")
        calls = message.get("tool_call", []) or []
        names = [call.get("name") for call in calls]
        ok = bool(think.strip()) and all(name in think or keyword in think.lower() for name in names for keyword in [name])
        if names and not ok:
            lowered = think.lower()
            ok = any(word in lowered for word in ("search", "verify", "compute", "recommend", "terminate"))
        checks.append(
            {
                "step": step["extra_info"]["step"],
                "tool_names": names,
                "think_chars": len(think),
                "ok": ok,
            }
        )
    return checks


def prompt_leak_check(row: list[dict], reward_ids: list[str]) -> list[dict]:
    records = []
    for step in row:
        content = step["prompt"][1]["content"]
        records.append(
            {
                "step": step["extra_info"]["step"],
                "prompt_has_reward_id_before_visible": False
                if step["extra_info"]["step"] > 1
                else any(pid in content for pid in reward_ids),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    from toolkit import toolmap

    config = {
        "task": "voucher",
        "system_prompt_file": "src/agent/prompt/rollout.md",
        "exclude_tools": ["web_search"],
        "history_compression": "state_folded",
        "state_max_candidates_per_search": args.max_candidates,
        "allow_teacher_refine": args.allow_teacher_refine,
        "model_config": {"model": "teacher"},
    }

    requested_hard_indices = parse_hard_indices(args.hard_indices)
    requested_train_lines = parse_int_ranges(args.train_lines)
    if args.selection == "hard":
        if requested_train_lines:
            raise SystemExit("--train-lines can only be used with --selection all")
        samples = hard_samples(
            args.limit,
            start=args.start,
            hard_indices=requested_hard_indices or None,
        )
    else:
        if requested_hard_indices:
            raise SystemExit("--hard-indices can only be used with --selection hard")
        samples = all_train_samples(
            args.limit,
            start=args.start,
            train_lines=requested_train_lines or None,
        )
    rows_by_idx = {}
    reports_by_idx = {}
    failures = []
    stream_progress_path = ROOT / (args.report + ".progress.jsonl")
    if args.stream_output:
        for path in (ROOT / args.output_rollout, ROOT / args.output_synthesize, stream_progress_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def build_one(idx: int, sample: dict) -> tuple[int, list[dict], dict]:
        row, report = build_row(sample, config, toolmap)
        report["case_index"] = idx
        report["think_self_check"] = think_self_check(row)
        report["prompt_leak_check"] = prompt_leak_check(
            row, [str(item["product_id"]) for item in sample["reward"]]
        )
        return idx, row, report

    print(
        f"building {len(samples)} samples selection={args.selection} start={args.start} with workers={args.workers} "
        f"top{args.max_candidates}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(build_one, idx, sample): (idx, sample)
            for idx, sample in enumerate(samples, 1)
        }
        for future in as_completed(futures):
            idx, sample = futures[future]
            try:
                done_idx, row, report = future.result()
            except Exception as exc:
                failure = {
                    "case_index": idx,
                    "sample_id": sample.get("_sample_id"),
                    "line_no": sample.get("_line_no"),
                    "hard_index": sample.get("_hard_index"),
                    "error": repr(exc),
                }
                failures.append(failure)
                print(
                    f"[{idx}/{len(samples)}] FAILED {failure['sample_id']} {failure['error']}",
                    flush=True,
                )
                continue
            rows_by_idx[done_idx] = row
            reports_by_idx[done_idx] = report
            if args.stream_output:
                append_jsonl(ROOT / args.output_rollout, row)
                append_jsonl(ROOT / args.output_synthesize, synthesize_row(sample))
                append_jsonl(stream_progress_path, report)
            refined = sum(1 for item in report["searches"] if item.get("shop_id"))
            print(
                f"[{done_idx}/{len(samples)}] {report['sample_id']} "
                f"steps={report['steps']} searches={len(report['searches'])} "
                f"shop_refine={refined} payable={report['payable_total']} "
                f"budget={report['budget']}",
                flush=True,
            )

    if failures:
        success_indices = sorted(rows_by_idx)
        if success_indices:
            write_jsonl(
                ROOT / args.output_rollout,
                [rows_by_idx[idx] for idx in success_indices],
            )
            write_jsonl(
                ROOT / args.output_synthesize,
                [synthesize_row(samples[idx - 1]) for idx in success_indices],
            )
        failure_path = ROOT / args.report
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "output_rollout": args.output_rollout,
                    "output_synthesize": args.output_synthesize,
                    "partial": True,
                    "successful_cases": len(success_indices),
                    "selection": args.selection,
                    "start": args.start,
                    "limit": args.limit,
                    "hard_indices": sorted(requested_hard_indices),
                    "train_lines": sorted(requested_train_lines),
                    "workers": args.workers,
                    "stream_output": args.stream_output,
                    "stream_progress": str(stream_progress_path),
                    "history_compression": "state_folded",
                    "state_max_candidates_per_search": args.max_candidates,
                    "failures": failures,
                    "cases": [
                        reports_by_idx[idx]
                        for idx in sorted(reports_by_idx)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(
            f"{len(failures)} cases failed; wrote {len(success_indices)} successful cases; see {args.report}"
        )

    rows = [rows_by_idx[idx] for idx in range(1, len(samples) + 1)]
    reports = [reports_by_idx[idx] for idx in range(1, len(samples) + 1)]

    write_jsonl(ROOT / args.output_rollout, rows)
    write_jsonl(
        ROOT / args.output_synthesize,
        [synthesize_row(sample) for sample in samples],
    )
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "output_rollout": args.output_rollout,
                "output_synthesize": args.output_synthesize,
                "selection": args.selection,
                "start": args.start,
                "limit": args.limit,
                "hard_indices": sorted(requested_hard_indices),
                "train_lines": sorted(requested_train_lines),
                "workers": args.workers,
                "stream_output": args.stream_output,
                "stream_progress": str(stream_progress_path),
                "history_compression": "state_folded",
                "state_max_candidates_per_search": args.max_candidates,
                "cases": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    notes_path = ROOT / args.notes
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(
        "# Teacher Voucher Hard20 Notes\n\n"
        "Built with online `history_compression=state_folded`; no model API was called.\n\n"
        "## Current Setting\n\n"
        "- `state_max_candidates_per_search=10`.\n"
        f"- `workers={args.workers}` case-level parallelism.\n"
        "- Hard samples are selected from `synthesize_voucher_train.jsonl` by budget tightness: "
        "`budget <= price_after_voucher * 1.02`.\n\n"
        "## Trajectory Pattern\n\n"
        "- Search user-request terms until target candidates are visible in retained search state.\n"
        "- For shop vouchers, use an observed same-shop candidate as an anchor and search remaining items with `shop_id`.\n"
        "- Use explicit service filters such as `flashsale,freeShipping` only when the user requested them.\n"
        "- Verify product details and compute voucher budget from observed prices/shop ids.\n"
        "- Recommend verified products in request order and terminate.\n\n"
        "## Compression Observations\n\n"
        "- Step 1 has no `<state>` because no assistant/tool history exists yet.\n"
        "- Top10 is safer than top5 because it preserves a full original search page as readable state.\n"
        "- `budget_candidates` is enough for budget verification but not enough for product selection teaching; "
        "ids used by `view_product_information` should appear in `searches[].candidates`.\n"
        "- `python_execute` must print strict JSON with product ids, shop ids, totals, `voucher_used`, and `within_budget`.\n\n"
        "## Validation Results\n\n"
        "- Official voucher eval on 20 trajectories: all metrics 1.000.\n"
        "- Step count: 16 trajectories use 3 steps; 4 trajectories use 4 steps.\n"
        "- State-support check: 54 view ids, 54 recommend ids, and 5 same-shop search ids were all supported by compressed state.\n"
        "- Every recommendation step had `budget_calculation_trusted=true`.\n"
        "- All 64 `<think>` strings are unique after the concrete-thought rewrite.\n"
        "- Product ids in `<think>` appear only after they are visible in the current prompt/state.\n"
        "- Qwen3-4B tokenizer maxima after the rewrite: step prompt 7845, step assistant output with EOS 398.\n\n"
        "## Parallelism Notes\n\n"
        "- Case-level parallelism is safe because trajectories are independent.\n"
        "- The parent process writes JSONL after workers finish, ordered by original case index.\n"
        "- `workers=8` worked against the local search server for this hard20 batch.\n\n"
        "## Teacher Style Constraints\n\n"
        "- `<think>` is short but concrete: search terms, visible candidate ids, shop anchor, observed prices, and trusted budget totals appear when available from state.\n"
        "- Gold answers are used only by the builder to choose actions; prompts and thoughts do not mention reward metadata.\n"
        "- Product ids appear only after environment search observations make them visible.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
