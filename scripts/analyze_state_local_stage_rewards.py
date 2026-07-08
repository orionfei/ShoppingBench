#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
REWARD_MODULE = ROOT / "src" / "rl" / "verl" / "utils" / "reward_score" / "shoppingbench_query.py"

if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from util.message import ASSISTANT_ROLES, Message  # noqa: E402


SUMMARY_KEYS = [
    "success",
    "progress",
    "find_correct",
    "view_confirmed",
    "budget_correct",
    "recommend_correct",
    "terminate_complete",
    "workflow_valid",
    "format",
    "tool_valid",
    "steps",
]

PER_QUERY_KEYS = [
    "score",
    "task",
    "structured_failure_mode",
    "success",
    "progress",
    "find_correct",
    "view_confirmed",
    "budget_correct",
    "recommend_correct",
    "terminate_complete",
    "workflow_valid",
    "format",
    "tool_valid",
    "search_gold_recall",
    "select_relevance_f1",
    "verify_relevance_f1",
    "budget_relevance_f1",
    "recommend_relevance_f1",
    "budget_ids_viewed",
    "recommended_ids_budgeted",
    "budget_numeric_alignment",
    "within_budget_correct",
    "steps",
    "recommended_ids",
    "expected_ids",
]


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_reward_module():
    spec = importlib.util.spec_from_file_location("shoppingbench_query", REWARD_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module: {REWARD_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_observed_products(rollout_rows: list) -> dict[str, dict]:
    products = {}
    for trajectory in rollout_rows:
        for step in trajectory:
            message = (step.get("completion") or {}).get("message") or {}
            for observation in message.get("obs") or []:
                results = observation.get("results")
                if not isinstance(results, list):
                    continue
                for product in results:
                    if not isinstance(product, dict) or product.get("product_id") is None:
                        continue
                    products[str(product["product_id"])] = product
    return products


def add_gold_products_from_documents(products: dict[str, dict], sample_rows: list, documents_path: Path) -> list[str]:
    missing = {
        str(item["product_id"])
        for row in sample_rows
        for item in row.get("reward", [])
        if isinstance(item, dict) and item.get("product_id") is not None
    } - set(products)
    if not missing:
        return []
    with documents_path.open(encoding="utf-8") as fin:
        for line in fin:
            if not missing:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            product = row.get("product") or {}
            product_id = str(product.get("product_id") or row.get("id"))
            if product_id in missing:
                products[product_id] = product
                missing.remove(product_id)
    return sorted(missing)


def write_product_cache(path: Path, products: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(products, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def trajectory_messages(trajectory: list) -> list[dict]:
    messages = []
    for step in trajectory:
        prompt = step.get("prompt") or []
        if prompt:
            messages.append({"role": "user", "content": prompt[-1].get("content", "")})
        message = (step.get("completion") or {}).get("message") or {}
        content = Message.from_dict(message).to_string(ASSISTANT_ROLES)
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_call": message.get("tool_call") or [],
                "obs": message.get("obs") or [],
            }
        )
    return messages


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    return {
        key: mean([float(row.get(key) or 0.0) for row in rows])
        for key in SUMMARY_KEYS
    }


def failure_modes(rows: list[dict]) -> dict:
    return dict(Counter(str(row.get("structured_failure_mode") or "missing") for row in rows))


def summarize_by_failure_mode(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        mode = str(row.get("structured_failure_mode") or "missing")
        grouped.setdefault(mode, []).append(row)
    return {
        mode: {
            "count": len(items),
            **summarize(items),
        }
        for mode, items in sorted(grouped.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze staged ShoppingBench reward components on state-local rollout JSONL.")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--rollout-file", required=True)
    parser.add_argument("--documents-file", default="resources/documents.jsonl")
    parser.add_argument("--product-cache-out", default="data/tmp/state_local_stage_reward_product_cache.json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    sample_path = ROOT / args.sample_file
    rollout_path = ROOT / args.rollout_file
    documents_path = ROOT / args.documents_file
    cache_path = ROOT / args.product_cache_out
    output_path = ROOT / args.output_json

    sample_rows = load_jsonl(sample_path)
    rollout_rows = load_jsonl(rollout_path)
    if len(sample_rows) != len(rollout_rows):
        raise ValueError(f"Sample/rollout length mismatch: {len(sample_rows)} vs {len(rollout_rows)}")

    products = iter_observed_products(rollout_rows)
    missing_gold = add_gold_products_from_documents(products, sample_rows, documents_path)
    write_product_cache(cache_path, products)
    os.environ["SHOPPINGBENCH_PRODUCT_CACHE"] = str(cache_path)

    reward_module = load_reward_module()
    per_query = []
    for index, (sample, trajectory) in enumerate(zip(sample_rows, rollout_rows, strict=True)):
        ground_truth = json.dumps(
            {"reward": sample.get("reward") or [], "voucher": sample.get("voucher") or {}},
            ensure_ascii=False,
        )
        score = reward_module.compute_score(
            "",
            ground_truth,
            extra_info={
                "messages": trajectory_messages(trajectory),
                "global_step": 0,
                "total_training_steps": 256,
            },
        )
        row = {key: score.get(key) for key in PER_QUERY_KEYS}
        row["idx"] = index
        row["query"] = str(sample.get("query") or "").split("\n", 1)[0]
        per_query.append(row)

    report = {
        "sample_file": args.sample_file,
        "rollout_file": args.rollout_file,
        "product_cache_out": args.product_cache_out,
        "missing_gold_product_ids": missing_gold,
        "summary": summarize(per_query),
        "failure_modes": failure_modes(per_query),
        "summary_by_failure_mode": summarize_by_failure_mode(per_query),
        "per_query": per_query,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
