#!/usr/bin/env python3
import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import ujson as json
except ModuleNotFoundError:
    import json


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_FILTER_KEYS = ("service", "shop_id", "sort", "price")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_product(base_url: str, params: dict, timeout: float = 15.0) -> list[dict]:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
    url = f"{base_url.rstrip('/')}/find_product?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def product_ids(results: list[dict]) -> list[str]:
    return [
        str(product.get("product_id"))
        for product in results
        if isinstance(product, dict) and product.get("product_id") is not None
    ]


def variants(params: dict) -> list[tuple[str, dict]]:
    cleaned = {key: value for key, value in params.items() if value not in (None, "")}
    result = [("original", dict(cleaned))]
    try:
        page = int(cleaned.get("page", 1))
    except Exception:
        page = 1
    if page >= 1:
        variant = dict(cleaned)
        variant["page"] = page + 1
        result.append(("next_page", variant))
    for key in OPTIONAL_FILTER_KEYS:
        if key in cleaned:
            variant = dict(cleaned)
            variant.pop(key, None)
            result.append((f"drop_{key}", variant))
    if any(key in cleaned for key in OPTIONAL_FILTER_KEYS):
        variant = {key: value for key, value in cleaned.items() if key not in OPTIONAL_FILTER_KEYS}
        result.append(("drop_all_optional_filters", variant))
    return result


def audit_failure(failure: dict, base_url: str, sleep_s: float) -> dict:
    missing = set(str(item) for item in failure.get("gold_missing") or [])
    attempts = []
    recovered = set()
    for attempt in failure.get("search_attempts") or []:
        params = attempt.get("parameters") or {}
        variant_rows = []
        for name, variant_params in variants(params):
            try:
                results = find_product(base_url, variant_params)
                ids = product_ids(results)
                hits = sorted(missing & set(ids))
                if hits:
                    recovered.update(hits)
                variant_rows.append(
                    {
                        "variant": name,
                        "parameters": variant_params,
                        "result_count": len(results),
                        "missing_gold_hits": hits,
                        "top_product_ids": ids[:10],
                    }
                )
            except Exception as exc:
                variant_rows.append(
                    {
                        "variant": name,
                        "parameters": variant_params,
                        "error": type(exc).__name__,
                        "message": str(exc)[:200],
                    }
                )
            if sleep_s > 0:
                time.sleep(sleep_s)
        attempts.append(
            {
                "original_step": attempt.get("step"),
                "original_parameters": params,
                "variants": variant_rows,
            }
        )
    return {
        "idx": failure.get("idx"),
        "structured_failure_mode": failure.get("structured_failure_mode"),
        "gold_missing": sorted(missing),
        "recovered_missing_gold": sorted(recovered),
        "attempts": attempts,
    }


def summarize(rows: list[dict]) -> dict:
    recoverable = [row for row in rows if row.get("recovered_missing_gold")]
    variant_hits = {}
    for row in rows:
        for attempt in row.get("attempts") or []:
            for variant in attempt.get("variants") or []:
                if variant.get("missing_gold_hits"):
                    name = variant.get("variant") or "unknown"
                    variant_hits[name] = variant_hits.get(name, 0) + len(variant["missing_gold_hits"])
    return {
        "rows": len(rows),
        "rows_with_recovered_missing_gold": len(recoverable),
        "variant_hit_counts": variant_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay failed find_product searches with optional filter ablations.")
    parser.add_argument("--failure-audit", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5631")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    args = parser.parse_args()

    failure_audit = load_json(ROOT / args.failure_audit)
    rows = [
        audit_failure(failure, args.base_url, args.sleep_s)
        for failure in failure_audit.get("failures") or []
        if failure.get("gold_missing")
    ]
    report = {
        "failure_audit": args.failure_audit,
        "base_url": args.base_url,
        "summary": summarize(rows),
        "rows": rows,
    }
    output_path = ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
