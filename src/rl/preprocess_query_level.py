import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def tool_schema_text() -> str:
    tools = [
        {
            "name": "find_product",
            "description": "Search for products and return up to 10 products, with each product including product_id, shop_id, title, price, service, and sold_count.",
            "parameters": {
                "q": "string, required. Search query.",
                "page": "integer, required. Page number from 1 to 5.",
                "shop_id": "string, optional. Restrict search to one shop.",
                "price": "string, optional. Price range such as 0-100.",
                "sort": "string, optional. One of priceasc, pricedesc, order, default.",
                "service": "string, optional. One or more of official, freeShipping, COD, flashsale, default.",
            },
        },
        {
            "name": "view_product_information",
            "description": "Fetch product descriptions, SKU options, and attributes for comma-separated product_ids.",
            "parameters": {"product_ids": "string, required. Comma-separated product ids."},
        },
        {
            "name": "recommend_product",
            "description": "Recommend selected product ids to the user.",
            "parameters": {"product_ids": "string, required. Comma-separated product ids in request order."},
        },
        {
            "name": "python_execute",
            "description": "Execute Python code string.",
            "parameters": {"code": "string, required. Python code that prints results."},
        },
        {
            "name": "terminate",
            "description": "End the dialogue with success or failure.",
            "parameters": {"status": "string, required. success or failure."},
        },
    ]
    lines = []
    for idx, tool in enumerate(tools, 1):
        lines.append(f"{idx}. Name: {tool['name']}")
        lines.append(f"Description: {tool['description']}")
        lines.append(f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_system_prompt(prompt_file: Path) -> str:
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    return prompt.replace("<|toolkit_description|>", tool_schema_text())


def load_needed_products(documents_file: Path, product_ids: set[str]) -> dict[str, dict]:
    products = {}
    with documents_file.open(encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Build reward product cache"):
            if len(products) == len(product_ids):
                break
            if not line.strip():
                continue
            item = json.loads(line)
            product = item.get("product") or {}
            product_id = str(product.get("product_id") or "")
            if product_id in product_ids:
                products[product_id] = product
    missing = sorted(product_ids - set(products))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} reward product ids in {documents_file}: {missing[:10]}")
    return products


def prompt_len(tokenizer, prompt: list[dict]) -> int:
    return len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "voucher_type": dict(Counter(row.get("voucher", {}).get("voucher_type", "unknown") for row in rows)),
        "product_count": dict(Counter(str(len(row.get("reward", []))) for row in rows)),
    }


def split_key(row: dict) -> tuple[str, int]:
    return row.get("voucher", {}).get("voucher_type", "unknown"), len(row.get("reward", []))


def stratified_split(rows: list[dict], val_size: float, rng) -> tuple[list[dict], list[dict]]:
    val_count = max(1, round(len(rows) * val_size))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault(split_key(row), []).append(row)

    allocations = {}
    remainders = []
    for key, group in groups.items():
        exact = len(group) * val_count / len(rows)
        base = min(len(group), int(np.floor(exact)))
        allocations[key] = base
        remainders.append((exact - base, len(group), key))

    remaining = val_count - sum(allocations.values())
    for _, _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1

    train, test = [], []
    for key, group in groups.items():
        group = list(group)
        rng.shuffle(group)
        test.extend(group[: allocations[key]])
        train.extend(group[allocations[key] :])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def convert_row(row: dict, idx: int, split: str, system_prompt: str) -> dict:
    ground_truth = {
        "reward": row["reward"],
        "voucher": row["voucher"],
    }
    return {
        "data_source": "shoppingbench_query",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["query"]},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "split": split,
            "index": idx,
            "query": row["query"],
            "reward": row["reward"],
            "voucher": row["voucher"],
        },
        "agent_name": "tool_agent",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query-level ShoppingBench RL parquet data.")
    parser.add_argument("--query-file", required=True, help="JSONL with query, reward, and voucher fields.")
    parser.add_argument("--local-dir", required=True, help="Output directory for train/test parquet.")
    parser.add_argument("--model-name", required=True, help="Tokenizer path/name.")
    parser.add_argument("--prompt-file", default=str(ROOT / "src" / "agent" / "prompt" / "rollout.md"))
    parser.add_argument("--documents-file", default=str(ROOT / "resources" / "documents.jsonl"))
    parser.add_argument("--product-cache-output", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.query_file))
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")

    system_prompt = build_system_prompt(Path(args.prompt_file))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    product_ids = {
        str(item["product_id"])
        for row in rows
        for item in row.get("reward", [])
        if item.get("product_id")
    }
    cache_output = Path(args.product_cache_output) if args.product_cache_output else Path(args.local_dir) / "product_cache.json"
    product_cache = load_needed_products(Path(args.documents_file), product_ids)
    write_json(cache_output, product_cache)

    rng = np.random.default_rng(args.seed)
    train_rows, test_rows = stratified_split(rows, args.val_size, rng)

    converted = []
    split_rows = {"train": [], "test": []}
    for split, rows_for_split in [("train", train_rows), ("test", test_rows)]:
        for idx, row in enumerate(rows_for_split):
            item = convert_row(row, idx, split, system_prompt)
            if prompt_len(tokenizer, item["prompt"]) <= args.max_length:
                converted.append(item)
                split_rows[split].append(row)

    train = [item for item in converted if item["extra_info"]["split"] == "train"]
    test = [item for item in converted if item["extra_info"]["split"] == "test"]
    rng.shuffle(train)

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(local_dir / "train.parquet")
    pd.DataFrame(test).to_parquet(local_dir / "test.parquet")
    report = {
        "source": args.query_file,
        "rows": len(rows),
        "kept": len(converted),
        "train": len(train),
        "test": len(test),
        "max_length": args.max_length,
        "product_cache": str(cache_output),
        "source_distribution": summarize_rows(rows),
        "train_distribution": summarize_rows(split_rows["train"]),
        "test_distribution": summarize_rows(split_rows["test"]),
    }
    write_json(local_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
