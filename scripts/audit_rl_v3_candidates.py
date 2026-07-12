#!/usr/bin/env python3
"""Deterministically audit generated RL-v3 query candidates and update the manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def voucher_price(total: float, voucher: dict) -> float:
    if voucher["discount_type"] == "fixed":
        return total - float(voucher["face_value"])
    return max(total * (1 - float(voucher["discount"])), total - float(voucher["cap"]))


def percentiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    def pick(q: float) -> int:
        return ordered[round((len(ordered) - 1) * q)]
    return {"min": ordered[0], "p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1]}


def bucket_distribution(rows: list[dict]) -> dict[str, dict[str, int]]:
    def count(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row["sampling_buckets"].get(key)) for row in rows).items()))
    return {
        "product_count": count("n_products"),
        "voucher_type": count("voucher_type"),
        "discount_type": count("discount_type"),
        "budget_difficulty": count("difficulty_bucket"),
        "constraint_complexity": count("constraint_complexity"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--excluded-product-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    plans = {row["sample_id"]: row for row in read_jsonl(args.plan)}
    queries = read_jsonl(args.queries)
    all_metadata = read_jsonl(args.metadata)
    metadata = [row for row in all_metadata if row.get("status", "accepted") == "accepted"]
    if len(queries) != len(metadata):
        raise SystemExit(f"query/metadata count mismatch: {len(queries)} != {len(metadata)}")
    excluded = {line.strip() for line in args.excluded_product_ids.read_text().splitlines() if line.strip()}
    seen_queries: set[str] = set()
    audited, repair = [], []
    for query_row, meta in zip(queries, metadata):
        sample_id = meta.get("sample_id")
        plan = plans.get(sample_id)
        reasons: list[str] = []
        if plan is None:
            reasons.append("missing_plan")
            plan = {}
        query = str(query_row.get("query") or "")
        key = normalized(query)
        if not query.strip(): reasons.append("empty_query")
        if key in seen_queries: reasons.append("duplicate_query")
        seen_queries.add(key)
        if meta.get("model") != "qwen3.6-flash": reasons.append("wrong_generator_model")
        if meta.get("status") != "accepted": reasons.append("generator_status_not_accepted")
        if query_row.get("reward") != plan.get("reward"): reasons.append("reward_changed_by_generator")
        if query_row.get("voucher") != plan.get("voucher"): reasons.append("voucher_changed_by_generator")
        if "My budget is only" not in query or "voucher with the following rules" not in query:
            reasons.append("missing_programmatic_voucher_suffix")
        opener = meta.get("opener_bucket")
        expected = {"im_looking": "I'm looking for", "looking_for": "Looking for", "show_me": "Show me", "find": "Find"}.get(opener)
        if expected and not query.startswith(expected): reasons.append("opener_mismatch")
        product_ids = [str(pid) for pid in plan.get("sampled_product_ids", [])]
        if set(product_ids).intersection(excluded): reasons.append("test_product_overlap")
        if any(pid in query for pid in product_ids): reasons.append("product_id_leakage")
        for product in plan.get("sampled_products", []):
            title = normalized(str(product.get("title") or ""))
            if len(title) >= 20 and title in key:
                reasons.append("full_title_copied")
                break
        voucher = plan.get("voucher") or {}
        if voucher:
            total = float(plan["total_price_before_voucher"])
            expected_price = voucher_price(total, voucher)
            if not math.isclose(expected_price, float(voucher["price_after_voucher"]), rel_tol=0, abs_tol=1e-6):
                reasons.append("voucher_price_mismatch")
            if float(voucher["price_after_voucher"]) > float(voucher["budget"]):
                reasons.append("budget_infeasible")
            if float(voucher["threshold"]) > total:
                reasons.append("threshold_infeasible")
            if voucher["voucher_type"] == "shop" and len(plan.get("sampled_shop_ids", [])) != 1:
                reasons.append("shop_scope_infeasible")
        audit = {
            "sample_id": sample_id,
            "static_eligible": not reasons,
            "reasons": reasons,
            "query": query,
            "reward": query_row.get("reward"),
            "voucher": query_row.get("voucher"),
            "sampled_product_ids": product_ids,
            "sampling_buckets": plan.get("sampling_buckets"),
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "generator_model": meta.get("model"),
        }
        audited.append(audit)
        if reasons:
            repair.append(audit)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "candidates_1500.audit.jsonl"
    eligible_path = out / "candidates_1500.static_eligible.jsonl"
    repair_path = out / "candidates_1500.repair.jsonl"
    report_path = out / "candidates_1500.audit_report.json"
    write_jsonl(audit_path, audited)
    write_jsonl(eligible_path, [row for row in audited if row["static_eligible"]])
    write_jsonl(repair_path, repair)
    reason_counts = Counter(reason for row in repair for reason in row["reasons"])
    eligible = [row for row in audited if row["static_eligible"]]
    report = {
        "schema_version": 1,
        "created_unix": time.time(),
        "generated": len(queries),
        "static_eligible": len(audited) - len(repair),
        "repair": len(repair),
        "reason_counts": dict(sorted(reason_counts.items())),
        "generator_models": dict(Counter(meta.get("model") for meta in metadata)),
        "generator_failed_attempts": len(all_metadata) - len(metadata),
        "query_unique": len(seen_queries),
        "eligible_rate": len(eligible) / len(audited) if audited else 0.0,
        "eligible_distributions": bucket_distribution(eligible),
        "repair_distributions": bucket_distribution(repair),
        "eligible_query_words": percentiles([len(row["query"].split()) for row in eligible]),
        "eligible_query_characters": percentiles([len(row["query"]) for row in eligible]),
        "test_product_overlap": sum("test_product_overlap" in row["reasons"] for row in repair),
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "eligible_path": str(eligible_path),
        "eligible_sha256": sha256(eligible_path),
        "repair_path": str(repair_path),
        "repair_sha256": sha256(repair_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifest = json.loads(args.manifest.read_text())
    manifest.update({
        "status": "static_audit_complete",
        "query_path": str(args.queries), "query_sha256": sha256(args.queries),
        "query_metadata_path": str(args.metadata), "query_metadata_sha256": sha256(args.metadata),
        "static_audit_report": str(report_path), "static_eligible": report["static_eligible"],
        "static_eligible_path": str(eligible_path),
        "static_eligible_sha256": report["eligible_sha256"],
        "static_repair": report["repair"], "generator_model": "qwen3.6-flash",
        "generator_failed_attempts": report["generator_failed_attempts"],
    })
    root = Path(__file__).resolve().parents[1]
    manifest["code_sha256"].update({
        "scripts/sample_coupon_budget.py": sha256(root / "scripts/sample_coupon_budget.py"),
        "scripts/audit_rl_v3_candidates.py": sha256(root / "scripts/audit_rl_v3_candidates.py"),
        "config/rl/rl_v3_query_generator_qwen36_flash.json": sha256(
            root / "config/rl/rl_v3_query_generator_qwen36_flash.json"
        ),
    })
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
