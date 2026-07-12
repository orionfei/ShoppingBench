#!/usr/bin/env python3
"""Build the repaired RL-v3 train1414, diverse val64, and fixed test250.

The script has two explicit phases. ``plan`` deterministically samples two
replacement train rows and a product/shop-disjoint validation plan.  An API
launcher then rewrites those plans.  ``finalize`` audits the generated text and
emits VERL parquet files plus a self-contained reward product cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "data/rl_v3/formal"
DATASET = ROOT / "dataset/shoppingbench_query_rl_v3"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCB = load_module(ROOT / "scripts/sample_coupon_budget.py", "sample_coupon_budget")
PREP = load_module(ROOT / "scripts/prepare_verl_shoppingbench_data.py", "prepare_verl_data")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reward_ids(row: dict[str, Any]) -> set[str]:
    return {str(item["product_id"]) for item in row.get("reward", [])}


def find_products(documents: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with documents.open(encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(wanted):
                break
            item = json.loads(line)
            product = item.get("product") or {}
            product_id = str(product.get("product_id") or "")
            if product_id in wanted:
                found[product_id] = product
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"missing products in documents: {sorted(missing)[:10]}")
    return found


def exact_val_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs = [(n, voucher) for n in range(1, 5) for voucher in ("platform", "shop") for _ in range(8)]
    discounts = ["fixed"] * 32 + ["percentage"] * 32
    difficulties = ["easy"] * 16 + ["medium"] * 24 + ["hard"] * 24
    complexities = ["low"] * 16 + ["medium"] * 32 + ["high"] * 16
    for values in (pairs, discounts, difficulties, complexities):
        rng.shuffle(values)
    return [
        {
            "sample_id": f"rl_v3_val_{idx:04d}",
            "n_products": pairs[idx][0],
            "voucher_type": pairs[idx][1],
            "discount_type": discounts[idx],
            "difficulty_bucket": difficulties[idx],
            "constraint_complexity": complexities[idx],
        }
        for idx in range(64)
    ]


class TitleEmbedder:
    def __init__(self, model_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cpu").eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
        output = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"]
        last = mask.sum(dim=1) - 1
        vectors = output[torch.arange(output.shape[0]), last]
        return F.normalize(vectors.float(), dim=-1).cpu()


def candidate_product_sets(
    spec: dict[str, Any], products: list[dict[str, Any]], shop2products: dict[Any, list[dict[str, Any]]],
    used_products: set[str], excluded_shops: set[str], used_shops: set[str], rng: random.Random, count: int = 8,
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for _ in range(256):
        if len(result) >= count:
            break
        if spec["voucher_type"] == "shop":
            shops = [
                items for shop, items in shop2products.items()
                if str(shop) not in excluded_shops | used_shops
                and len([p for p in items if str(p["product_id"]) not in used_products]) >= spec["n_products"]
            ]
            if not shops:
                break
            available = [p for p in rng.choice(shops) if str(p["product_id"]) not in used_products]
            selected = rng.sample(available, spec["n_products"])
        else:
            available = [
                p for p in products
                if str(p["product_id"]) not in used_products
                and str(p.get("shop_id")) not in excluded_shops | used_shops
            ]
            rng.shuffle(available)
            selected, shops_seen = [], set()
            for product in available:
                shop = str(product.get("shop_id"))
                if shop in shops_seen:
                    continue
                selected.append(product)
                shops_seen.add(shop)
                if len(selected) == spec["n_products"]:
                    break
            if len(selected) != spec["n_products"]:
                break
        signature = tuple(sorted(str(p["product_id"]) for p in selected))
        if all(tuple(sorted(str(p["product_id"]) for p in old)) != signature for old in result):
            result.append(selected)
    if len(result) < count:
        raise RuntimeError(f"only {len(result)} candidates for {spec['sample_id']}")
    return result


def make_plan_item(spec: dict[str, Any], selected: list[dict[str, Any]], prompt: str, rng_seed: int) -> dict[str, Any] | None:
    random.seed(rng_seed)
    voucher = SCB.build_voucher(selected, spec["voucher_type"], spec["discount_type"], spec["difficulty_bucket"])
    if voucher is None:
        return None
    item = SCB.build_plan_item(spec, selected, prompt, voucher)
    SCB.validate_plan_item(item)
    return item


def plan_phase(args: argparse.Namespace) -> None:
    FORMAL.mkdir(parents=True, exist_ok=True)
    base_train = read_jsonl(ROOT / "data/rl_v3/candidates_1500.static_eligible.jsonl")
    base_plan = {row["sample_id"]: row for row in read_jsonl(ROOT / "data/rl_v3/candidates_1500.plan.jsonl")}
    test = read_jsonl(ROOT / "data/synthesize_voucher_test.jsonl")
    train_ids = set().union(*(reward_ids(row) for row in base_train))
    test_ids = set().union(*(reward_ids(row) for row in test))
    conflicts = [row for row in base_train if reward_ids(row) & test_ids]
    if len(conflicts) != 2:
        raise RuntimeError(f"expected two train/test conflicts, found {len(conflicts)}")

    products, shop2products = SCB.load_sampling_pool(ROOT / "resources/documents.jsonl", args.max_docs)
    product_by_id = {str(product["product_id"]): product for product in products}
    missing_test = test_ids - set(product_by_id)
    product_by_id.update(find_products(ROOT / "resources/documents.jsonl", missing_test))
    test_shops = {str(product_by_id[pid].get("shop_id")) for pid in test_ids}
    eligible_plan_rows = [base_plan[row["sample_id"]] for row in base_train]
    train_shops = {str(shop) for row in eligible_plan_rows for shop in row["sampled_shop_ids"]}
    prompt = (ROOT / "src/agent/prompt/synthesize.md").read_text(encoding="utf-8").strip()
    rng = random.Random(args.seed)

    repair_plan: list[dict[str, Any]] = []
    used_products = train_ids | test_ids
    for index, conflict in enumerate(conflicts):
        old = base_plan[conflict["sample_id"]]
        spec = {
            "sample_id": conflict["sample_id"],
            "n_products": old["sampling_buckets"]["n_products"],
            "voucher_type": old["sampling_buckets"]["voucher_type"],
            "discount_type": old["sampling_buckets"]["discount_type"],
            "difficulty_bucket": old["sampling_buckets"]["difficulty_bucket"],
            "constraint_complexity": old["sampling_buckets"]["constraint_complexity"],
        }
        selected = SCB.sample_products(products, shop2products, spec["voucher_type"], spec["n_products"], used_products)
        item = make_plan_item(spec, selected, prompt, args.seed + index)
        if item is None:
            raise RuntimeError(f"failed replacement plan for {spec['sample_id']}")
        repair_plan.append(item)
        used_products.update(map(str, item["sampled_product_ids"]))

    # Repaired products are also excluded from validation.
    repaired_train_ids = (train_ids - set().union(*(reward_ids(row) for row in conflicts))) | {
        str(pid) for row in repair_plan for pid in row["sampled_product_ids"]
    }
    excluded_shops = train_shops | test_shops | {
        str(shop) for row in repair_plan for shop in row["sampled_shop_ids"]
    }
    val_used_products = repaired_train_ids | test_ids
    val_used_shops: set[str] = set()
    embedder = TitleEmbedder(ROOT / "model/Qwen3-Embedding-0.6B")
    chosen_embeddings: list[torch.Tensor] = []
    val_plan: list[dict[str, Any]] = []
    for idx, spec in enumerate(exact_val_specs(args.seed + 1)):
        sets = candidate_product_sets(
            spec, products, shop2products, val_used_products, excluded_shops, val_used_shops, rng
        )
        feasible: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        for candidate_idx, selected in enumerate(sets):
            item = make_plan_item(spec, selected, prompt, args.seed + 1000 + idx * 32 + candidate_idx)
            if item is not None:
                feasible.append((selected, item))
        if not feasible:
            raise RuntimeError(f"no voucher-feasible candidates for {spec['sample_id']}")
        texts = [
            " ; ".join(str(product.get("title") or "") for product in selected)
            for selected, _item in feasible
        ]
        vectors = embedder.encode(texts)
        if chosen_embeddings:
            previous = torch.stack(chosen_embeddings)
            scores = 1.0 - torch.max(vectors @ previous.T, dim=1).values
            choice = int(torch.argmax(scores).item())
        else:
            choice = max(range(len(texts)), key=lambda i: len(set(texts[i].lower().split())))
        selected, item = feasible[choice]
        val_plan.append(item)
        chosen_embeddings.append(vectors[choice])
        val_used_products.update(map(str, item["sampled_product_ids"]))
        val_used_shops.update(map(str, item["sampled_shop_ids"]))

    SCB.validate_plan(repair_plan)
    SCB.validate_plan(val_plan)
    write_jsonl(FORMAL / "train_repair.plan.jsonl", repair_plan)
    write_jsonl(FORMAL / "val64.plan.jsonl", val_plan)
    write_json(FORMAL / "plan_report.json", {
        "repair_sample_ids": [row["sample_id"] for row in repair_plan],
        "val_rows": len(val_plan),
        "val_unique_products": len({str(pid) for row in val_plan for pid in row["sampled_product_ids"]}),
        "val_unique_shops": len({str(shop) for row in val_plan for shop in row["sampled_shop_ids"]}),
        "excluded_train_products": len(repaired_train_ids),
        "excluded_test_products": len(test_ids),
        "maximin_embedding_model": "Qwen3-Embedding-0.6B",
        "distributions": {
            "product_count": dict(Counter(str(row["sampling_buckets"]["n_products"]) for row in val_plan)),
            "voucher_type": dict(Counter(row["voucher"]["voucher_type"] for row in val_plan)),
            "discount_type": dict(Counter(row["voucher"]["discount_type"] for row in val_plan)),
            "difficulty": dict(Counter(row["sampling_buckets"]["difficulty_bucket"] for row in val_plan)),
            "complexity": dict(Counter(row["sampling_buckets"]["constraint_complexity"] for row in val_plan)),
        },
    })
    print(json.dumps({"repair": len(repair_plan), "val": len(val_plan)}, ensure_ascii=False))


def generated_map(query_path: Path, metadata_path: Path) -> dict[str, dict[str, Any]]:
    queries = read_jsonl(query_path)
    metadata = [row for row in read_jsonl(metadata_path) if row.get("status") == "accepted"]
    if len(queries) != len(metadata):
        raise RuntimeError(f"generated query/meta mismatch: {len(queries)} != {len(metadata)}")
    return {meta["sample_id"]: query for query, meta in zip(queries, metadata)}


def full_title_copy(query: str, plan: dict[str, Any]) -> bool:
    normalized = " ".join(query.lower().split())
    return any(
        len(" ".join(str(product.get("title") or "").lower().split())) >= 20
        and " ".join(str(product.get("title") or "").lower().split()) in normalized
        for product in plan["sampled_products"]
    )


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "product_count": dict(Counter(str(len(row["reward"])) for row in rows)),
        "voucher_type": dict(Counter(row["voucher"]["voucher_type"] for row in rows)),
        "discount_type": dict(Counter(row["voucher"]["discount_type"] for row in rows)),
    }


def finalize_phase(args: argparse.Namespace) -> None:
    base_train = read_jsonl(ROOT / "data/rl_v3/candidates_1500.static_eligible.jsonl")
    test = read_jsonl(ROOT / "data/synthesize_voucher_test.jsonl")
    repair_plan = {row["sample_id"]: row for row in read_jsonl(FORMAL / "train_repair.plan.jsonl")}
    val_plan = {row["sample_id"]: row for row in read_jsonl(FORMAL / "val64.plan.jsonl")}
    repairs = generated_map(FORMAL / "train_repair.query.jsonl", FORMAL / "train_repair.query.meta.jsonl")
    val_map = generated_map(FORMAL / "val64.query.jsonl", FORMAL / "val64.query.meta.jsonl")
    if set(repairs) != set(repair_plan) or set(val_map) != set(val_plan):
        raise RuntimeError("generated sample IDs do not match plans")
    copied = [sid for sid, row in {**repair_plan, **val_plan}.items() if full_title_copy((repairs | val_map)[sid]["query"], row)]
    if copied:
        raise RuntimeError(f"full titles copied in generated queries: {copied}")

    train = []
    train_meta = []
    for row in base_train:
        sample_id = row["sample_id"]
        final = repairs.get(sample_id) or {"query": row["query"], "reward": row["reward"], "voucher": row["voucher"]}
        SCB.validate_final_item(final)
        train.append(final)
        train_meta.append({"sample_id": sample_id, "sampling_buckets": (repair_plan.get(sample_id) or row)["sampling_buckets"]})
    val = [val_map[sid] for sid in sorted(val_map)]
    for row in val + test:
        SCB.validate_final_item(row)

    train_ids = set().union(*(reward_ids(row) for row in train))
    val_ids = set().union(*(reward_ids(row) for row in val))
    test_ids = set().union(*(reward_ids(row) for row in test))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("train/val/test product overlap remains")
    queries = [{" ".join(row["query"].lower().split()) for row in rows} for rows in (train, val, test)]
    if queries[0] & queries[1] or queries[0] & queries[2] or queries[1] & queries[2]:
        raise RuntimeError("train/val/test query overlap remains")
    if (len(train), len(val), len(test)) != (1414, 64, 250):
        raise RuntimeError(f"bad split counts: {len(train)}, {len(val)}, {len(test)}")

    # Gold cache: plan snapshots cover train/val; scan the fixed test products.
    cache: dict[str, dict[str, Any]] = {}
    base_plan = {row["sample_id"]: row for row in read_jsonl(ROOT / "data/rl_v3/candidates_1500.plan.jsonl")}
    for meta in train_meta:
        plan = repair_plan.get(meta["sample_id"]) or base_plan[meta["sample_id"]]
        for product in plan["sampled_products"]:
            cache[str(product["product_id"])] = product
    for plan in val_plan.values():
        for product in plan["sampled_products"]:
            cache[str(product["product_id"])] = product
    cache.update(find_products(ROOT / "resources/documents.jsonl", test_ids))
    write_json(DATASET / "product_cache.json", cache)

    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_path, trust_remote_code=True)
    system_prompt = PREP.build_system_prompt(ROOT / "src/agent/prompt/rollout.state_local.md")
    DATASET.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": train, "validation": val, "test": test}
    for split, rows in split_rows.items():
        converted = []
        for idx, row in enumerate(rows):
            meta = train_meta[idx] if split == "train" else {"source": "new_val64" if split == "validation" else "synthesize_voucher_test"}
            item = PREP.convert_query_row(row, idx, split, system_prompt, meta)
            length = PREP.token_len(tokenizer, item["prompt"])
            if length > 2048:
                raise RuntimeError(f"{split}[{idx}] prompt has {length} tokens")
            item["extra_info"]["prompt_tokens"] = length
            converted.append(item)
        pd.DataFrame(converted).to_parquet(DATASET / f"{split}.parquet", index=False)
        write_jsonl(FORMAL / f"{split}.jsonl", rows)

    report = {
        "schema_version": 1,
        "counts": {name: len(rows) for name, rows in split_rows.items()},
        "query_overlap": 0,
        "product_overlap": 0,
        "unique_products": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "distribution": {name: distribution(rows) for name, rows in split_rows.items()},
        "files": {},
    }
    for name in ("train", "validation", "test", "product_cache"):
        path = DATASET / (f"{name}.parquet" if name != "product_cache" else "product_cache.json")
        report["files"][name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(DATASET / "report.json", report)
    report["files"]["report"] = {"path": str((DATASET / "report.json").relative_to(ROOT)), "sha256": sha256(DATASET / "report.json")}
    write_json(DATASET / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "finalize"))
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-docs", type=int, default=100000)
    parser.add_argument("--model-path", default="checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108")
    args = parser.parse_args()
    (plan_phase if args.stage == "plan" else finalize_phase)(args)


if __name__ == "__main__":
    main()
