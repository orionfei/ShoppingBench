#!/usr/bin/env python3
"""Prepare the deterministic RL-v3 candidate plan and product exclusions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/rl_v3")
    parser.add_argument("--test-parquet", type=Path, default=ROOT / "dataset/shoppingbench_query_rl_v2/test.parquet")
    parser.add_argument("--documents-file", type=Path, default=ROOT / "resources/documents.jsonl")
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-docs", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    exclusions = out / "excluded_test75_product_ids.txt"
    plan = out / f"candidates_{args.total}.plan.jsonl"
    manifest_path = out / "manifest.json"
    if plan.exists() and not args.force:
        raise SystemExit(f"plan already exists: {plan}; pass --force to replace it")

    test = pd.read_parquet(args.test_parquet)
    product_ids = sorted({
        str(product_id)
        for extra in test["extra_info"]
        for product_id in extra.get("reward_product_ids", [])
    })
    exclusions.write_text("\n".join(product_ids) + "\n", encoding="utf-8")

    command = [
        sys.executable, str(ROOT / "scripts/sample_coupon_budget.py"),
        "--stage", "plan", "--profile", "rl-v3-candidate",
        "--documents-file", str(args.documents_file), "--plan-output", str(plan),
        "--total", str(args.total), "--seed", str(args.seed), "--max-docs", str(args.max_docs),
        "--exclude-product-ids-file", str(exclusions),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    rows = read_jsonl(plan)
    sampled_ids = [str(pid) for row in rows for pid in row["sampled_product_ids"]]
    overlap = sorted(set(sampled_ids).intersection(product_ids))
    if overlap:
        raise RuntimeError(f"candidate/test product overlap: {overlap[:5]}")
    if len(sampled_ids) != len(set(sampled_ids)):
        raise RuntimeError("candidate plan reused a product id")

    manifest = {
        "schema_version": 1,
        "status": "plan_ready",
        "created_unix": time.time(),
        "profile": "rl-v3-candidate",
        "candidate_total": len(rows),
        "seed": args.seed,
        "max_documents_loaded": args.max_docs,
        "query_generator_model": "qwen3.6-flash",
        "query_generator_env_file": "/root/project/ResearchHarness/.env",
        "plan_path": str(plan),
        "plan_sha256": sha256(plan),
        "documents_path": str(args.documents_file.resolve()),
        "documents_bytes": args.documents_file.stat().st_size,
        "documents_sha256": sha256(args.documents_file),
        "test_parquet_path": str(args.test_parquet.resolve()),
        "test_parquet_sha256": sha256(args.test_parquet),
        "excluded_product_ids_path": str(exclusions),
        "excluded_product_ids": len(product_ids),
        "excluded_product_ids_sha256": sha256(exclusions),
        "sampled_unique_product_ids": len(sampled_ids),
        "test_product_overlap": 0,
        "distributions": {
            "product_count": dict(sorted(Counter(row["sampling_buckets"]["n_products"] for row in rows).items())),
            "voucher_type": dict(sorted(Counter(row["voucher"]["voucher_type"] for row in rows).items())),
            "discount_type": dict(sorted(Counter(row["voucher"]["discount_type"] for row in rows).items())),
            "budget_difficulty": dict(sorted(Counter(row["sampling_buckets"]["difficulty_bucket"] for row in rows).items())),
            "constraint_complexity": dict(sorted(Counter(row["sampling_buckets"]["constraint_complexity"] for row in rows).items())),
        },
        "code_sha256": {
            "scripts/sample_coupon_budget.py": sha256(ROOT / "scripts/sample_coupon_budget.py"),
            "scripts/prepare_rl_v3_candidates.py": sha256(ROOT / "scripts/prepare_rl_v3_candidates.py"),
            "config/rl/rl_v3_query_generator_qwen36_flash.json": sha256(ROOT / "config/rl/rl_v3_query_generator_qwen36_flash.json"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(plan), "candidates": len(rows), "products": len(sampled_ids), "test_overlap": 0}))


if __name__ == "__main__":
    main()
