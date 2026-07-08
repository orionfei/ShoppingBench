#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    import ujson as json
except ModuleNotFoundError:
    import json


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def product_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def collect_referenced_ids(sample_rows: list[dict], rollout_rows: list[list[dict]]) -> set[str]:
    ids = set()
    for row in sample_rows:
        for item in row.get("reward") or []:
            if isinstance(item, dict) and item.get("product_id") is not None:
                ids.add(str(item["product_id"]))
    for trajectory in rollout_rows:
        for step in trajectory:
            message = (step.get("completion") or {}).get("message") or {}
            for call in message.get("tool_call") or []:
                params = call.get("parameters") or {}
                ids.update(product_ids(params.get("product_ids")))
            for obs in message.get("obs") or []:
                results = obs.get("results")
                if isinstance(results, list):
                    for product in results:
                        if isinstance(product, dict) and product.get("product_id") is not None:
                            ids.add(str(product["product_id"]))
    return ids


def load_products(documents_path: Path, ids: set[str]) -> dict[str, dict]:
    products = {}
    missing = set(ids)
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
    return products


def compact_product(product_id: str, products: dict[str, dict]) -> dict:
    product = products.get(str(product_id), {})
    title = product.get("title")
    if isinstance(title, list):
        title = " ".join(str(item) for item in title)
    return {
        "product_id": str(product_id),
        "shop_id": product.get("shop_id"),
        "price": product.get("price"),
        "title": str(title or "")[:120],
    }


def compact_params(params: dict) -> dict:
    return {
        key: params.get(key)
        for key in ("q", "page", "shop_id", "price", "sort", "service")
        if params.get(key) not in (None, "")
    }


def audit_failure(index: int, sample: dict, trajectory: list[dict], stage_row: dict, products: dict[str, dict]) -> dict:
    gold_ids = [
        str(item.get("product_id"))
        for item in sample.get("reward") or []
        if isinstance(item, dict) and item.get("product_id") is not None
    ]
    observed_ids = set()
    search_attempts = []
    budgeted_ids = []
    recommended_ids = []
    final_only_used = False

    for step_index, step in enumerate(trajectory, 1):
        extra = step.get("extra_info") or {}
        if extra.get("allowed_tools") == ["recommend_product", "terminate"]:
            final_only_used = True
        message = (step.get("completion") or {}).get("message") or {}
        calls = message.get("tool_call") or []
        observations = message.get("obs") or []
        for call, obs in zip(calls, observations, strict=False):
            name = call.get("name")
            params = call.get("parameters") or {}
            results = obs.get("results")
            if name == "find_product":
                hits = []
                result_count = len(results) if isinstance(results, list) else None
                if isinstance(results, list):
                    for product in results:
                        if not isinstance(product, dict) or product.get("product_id") is None:
                            continue
                        product_id = str(product["product_id"])
                        observed_ids.add(product_id)
                        if product_id in gold_ids:
                            hits.append(product_id)
                search_attempts.append(
                    {
                        "step": step_index,
                        "parameters": compact_params(params),
                        "result_count": result_count,
                        "gold_hits": hits,
                    }
                )
            elif name == "budget_check":
                budgeted_ids = product_ids(params.get("product_ids"))
            elif name == "recommend_product":
                recommended_ids = product_ids(params.get("product_ids"))

    gold_seen = sorted(set(gold_ids) & observed_ids)
    gold_missing = sorted(set(gold_ids) - observed_ids)
    if gold_missing:
        failure_mode = "search_recall_gap"
    elif final_only_used:
        failure_mode = "final_selection_after_full_recall_gap"
    else:
        failure_mode = "selection_after_full_recall_gap"
    return {
        "idx": index,
        "success": stage_row.get("success"),
        "progress": stage_row.get("progress"),
        "steps": len(trajectory),
        "final_only_used": final_only_used,
        "structured_failure_mode": failure_mode,
        "query": str(sample.get("query") or "").split("\n", 1)[0],
        "voucher": sample.get("voucher") or {},
        "expected": [compact_product(product_id, products) for product_id in gold_ids],
        "recommended": [compact_product(product_id, products) for product_id in recommended_ids],
        "budgeted": [compact_product(product_id, products) for product_id in budgeted_ids],
        "gold_seen": gold_seen,
        "gold_missing": gold_missing,
        "search_attempts": search_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit failed state-local ShoppingBench trajectories.")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--rollout-file", required=True)
    parser.add_argument("--stage-report", required=True)
    parser.add_argument("--documents-file", default="resources/documents.jsonl")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    sample_rows = load_jsonl(ROOT / args.sample_file)
    rollout_rows = load_jsonl(ROOT / args.rollout_file)
    stage_report = json.loads((ROOT / args.stage_report).read_text(encoding="utf-8"))
    products = load_products(ROOT / args.documents_file, collect_referenced_ids(sample_rows, rollout_rows))

    failures = []
    for index, (sample, trajectory, stage_row) in enumerate(
        zip(sample_rows, rollout_rows, stage_report.get("per_query") or [], strict=True)
    ):
        if float(stage_row.get("success") or 0.0) >= 1.0:
            continue
        failures.append(audit_failure(index, sample, trajectory, stage_row, products))

    summary = {
        "failure_count": len(failures),
        "final_only_failures": sum(1 for item in failures if item["final_only_used"]),
        "all_gold_seen_failures": sum(1 for item in failures if not item["gold_missing"]),
        "any_gold_missing_failures": sum(1 for item in failures if item["gold_missing"]),
        "failure_modes": {},
    }
    for item in failures:
        mode = item["structured_failure_mode"]
        summary["failure_modes"][mode] = summary["failure_modes"].get(mode, 0) + 1
    report = {
        "sample_file": args.sample_file,
        "rollout_file": args.rollout_file,
        "stage_report": args.stage_report,
        "summary": summary,
        "failures": failures,
    }
    output_path = ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
