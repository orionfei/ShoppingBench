#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def maybe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def product_ids_from_query_rows(rows: list[dict]) -> set[str]:
    return {
        str(item["product_id"])
        for row in rows
        for item in row.get("reward", [])
        if item.get("product_id") is not None
    }


def open_documents(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def build_product_cache(documents_file: Path, product_ids: set[str], output: Path) -> dict[str, dict]:
    products = {}
    with open_documents(documents_file) as fin:
        for line in tqdm(fin, desc="Build product cache"):
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
        raise RuntimeError(f"Missing {len(missing)} product ids in {documents_file}: {missing[:10]}")
    write_json(output, products)
    return products


def validate_or_build_product_cache(
    cache_path: Path,
    product_ids: set[str],
    documents_file: Path,
    mode: str,
) -> dict:
    if mode == "skip":
        return {"mode": mode, "path": maybe_rel(cache_path), "checked": False}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fin:
            cache = json.load(fin)
        missing = sorted(product_ids - set(map(str, cache.keys())))
        if not missing:
            return {
                "mode": "validate",
                "path": maybe_rel(cache_path),
                "checked": True,
                "product_ids": len(product_ids),
                "cache_size": len(cache),
                "missing": 0,
            }
        if mode == "validate":
            raise RuntimeError(f"{cache_path} is missing {len(missing)} product ids: {missing[:10]}")
    if mode != "build":
        raise RuntimeError(f"{cache_path} does not exist. Use --product-cache-mode build or skip.")
    build_product_cache(documents_file, product_ids, cache_path)
    return {
        "mode": "build",
        "path": maybe_rel(cache_path),
        "checked": True,
        "product_ids": len(product_ids),
    }


def split_key(row: dict) -> tuple[str, int]:
    if "row" in row:
        row = row["row"]
    return row.get("voucher", {}).get("voucher_type", "unknown"), len(row.get("reward", []))


def stratified_split(rows: list[dict], val_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1).")
    if val_size == 0:
        return list(rows), []
    rng = np.random.default_rng(seed)
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

    train, val = [], []
    for key, group in groups.items():
        group = list(group)
        rng.shuffle(group)
        val.extend(group[: allocations[key]])
        train.extend(group[allocations[key] :])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def summarize_query_rows(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "voucher_type": dict(Counter(row.get("voucher", {}).get("voucher_type", "unknown") for row in rows)),
        "product_count": dict(Counter(str(len(row.get("reward", []))) for row in rows)),
    }


def convert_query_row(row: dict, idx: int, split: str, system_prompt: str, meta: dict | None) -> dict:
    ground_truth = {"reward": row["reward"], "voucher": row["voucher"]}
    extra_info = {
        "split": split,
        "index": idx,
        "query": row["query"],
        "reward": row["reward"],
        "voucher": row["voucher"],
    }
    if meta:
        extra_info["source_meta"] = meta
    return {
        "data_source": "shoppingbench_query",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["query"]},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": extra_info,
        "agent_name": "tool_agent",
    }


def token_len(tokenizer, messages: list[dict]) -> int:
    return count_chat_tokens(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))


def count_chat_tokens(tokenized) -> int:
    if hasattr(tokenized, "data") and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "numel"):
        return int(tokenized.numel())
    if tokenized and isinstance(tokenized[0], list):
        return len(tokenized[0])
    return len(tokenized)


def prepare_query(args, tokenizer) -> dict:
    rows = read_jsonl(ROOT / args.query_file)
    meta_rows = read_jsonl(ROOT / args.query_meta_file) if args.query_meta_file else [None] * len(rows)
    if len(meta_rows) != len(rows):
        raise ValueError(f"query/meta row count mismatch: {len(rows)} vs {len(meta_rows)}")

    cache_report = validate_or_build_product_cache(
        ROOT / args.product_cache,
        product_ids_from_query_rows(rows),
        ROOT / args.documents_file,
        args.product_cache_mode,
    )

    system_prompt = build_system_prompt(ROOT / args.prompt_file)
    paired = [{"row": row, "meta": meta} for row, meta in zip(rows, meta_rows)]
    train_pairs, val_pairs = stratified_split(paired, args.query_val_size, args.seed)

    output_dir = ROOT / args.query_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source": args.query_file,
        "meta": args.query_meta_file,
        "max_prompt_length": args.query_max_prompt_length,
        "product_cache": cache_report,
        "source_distribution": summarize_query_rows(rows),
    }
    for split, pairs in [("train", train_pairs), ("test", val_pairs)]:
        converted = []
        dropped = []
        for idx, pair in enumerate(pairs):
            item = convert_query_row(pair["row"], idx, split, system_prompt, pair["meta"])
            length = token_len(tokenizer, item["prompt"])
            if length <= args.query_max_prompt_length:
                item["extra_info"]["prompt_tokens"] = length
                converted.append(item)
            else:
                dropped.append({"index": idx, "prompt_tokens": length})
        pd.DataFrame(converted).to_parquet(output_dir / f"{split}.parquet")
        report[split] = {
            "rows": len(converted),
            "dropped_overlong": len(dropped),
            "distribution": summarize_query_rows([item["extra_info"] for item in converted]),
            "max_prompt_tokens": max((item["extra_info"]["prompt_tokens"] for item in converted), default=0),
        }
    write_json(output_dir / "report.json", report)
    return report


def read_state_folded_sft(path: Path) -> list[list[dict]]:
    return read_jsonl(path)


def split_trajectories(rows: list, val_size: float, seed: int) -> tuple[list, list]:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1).")
    indices = list(range(len(rows)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_n = 0 if val_size == 0 else max(1, round(len(rows) * val_size))
    val_idx = set(indices[:val_n])
    train, val = [], []
    for idx, row in enumerate(rows):
        (val if idx in val_idx else train).append(row)
    return train, val


def completion_content(step: dict) -> str:
    completion = step.get("completion", "")
    if isinstance(completion, dict):
        return completion.get("content") or completion.get("response") or json.dumps(completion, ensure_ascii=False)
    return str(completion)


def convert_sft_step(step: dict, idx: int, split: str) -> dict:
    messages = list(step["prompt"]) + [{"role": "assistant", "content": completion_content(step)}]
    extra_info = dict(step.get("extra_info") or {})
    extra_info.update({"split": split, "index": idx})
    return {
        "messages": messages,
        "enable_thinking": False,
        "extra_info": extra_info,
    }


def summarize_sft(rows: list[dict]) -> dict:
    steps = [row.get("extra_info", {}).get("step") for row in rows]
    return {
        "rows": len(rows),
        "step": dict(Counter(str(step) for step in steps)),
        "max_tokens": max((row.get("token_length", 0) for row in rows), default=0),
    }


def prepare_sft(args, tokenizer) -> dict:
    trajectories = read_state_folded_sft(ROOT / args.sft_state_folded_file)
    train_traj, val_traj = split_trajectories(trajectories, args.sft_val_size, args.seed)
    output_dir = ROOT / args.sft_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source": args.sft_state_folded_file,
        "trajectory_count": len(trajectories),
        "max_length": args.sft_max_length,
    }

    for split, split_traj in [("train", train_traj), ("test", val_traj)]:
        rows = []
        dropped = []
        for trajectory_idx, trajectory in enumerate(split_traj):
            for step_idx, step in enumerate(trajectory):
                item = convert_sft_step(step, len(rows), split)
                item["extra_info"]["trajectory_index"] = trajectory_idx
                item["extra_info"]["trajectory_step_index"] = step_idx
                length = count_chat_tokens(
                    tokenizer.apply_chat_template(item["messages"], tokenize=True, add_generation_prompt=False)
                )
                if length <= args.sft_max_length:
                    item["token_length"] = length
                    rows.append(item)
                else:
                    dropped.append({"trajectory_index": trajectory_idx, "step_index": step_idx, "tokens": length})
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet")
        report[split] = {
            "trajectories": len(split_traj),
            "rows": len(rows),
            "dropped_overlong": len(dropped),
            "summary": summarize_sft(rows),
        }
    write_json(output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShoppingBench parquet data for verl SFT and GRPO.")
    parser.add_argument("--model-name", default="model/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-query", action="store_true")

    parser.add_argument("--sft-state-folded-file", default="data/teacher_voucher_train_clean691_state_folded.jsonl")
    parser.add_argument("--sft-output-dir", default="dataset/shoppingbench_sft_state_folded")
    parser.add_argument("--sft-val-size", type=float, default=0.05)
    parser.add_argument("--sft-max-length", type=int, default=20480)

    parser.add_argument("--query-file", default="data/rl_voucher_queries_750.jsonl")
    parser.add_argument("--query-meta-file", default="data/rl_voucher_queries_750.meta.jsonl")
    parser.add_argument("--query-output-dir", default="dataset/shoppingbench_query")
    parser.add_argument("--query-val-size", type=float, default=0.1)
    parser.add_argument("--query-max-prompt-length", type=int, default=4096)
    parser.add_argument("--prompt-file", default="src/agent/prompt/rollout.md")
    parser.add_argument("--product-cache", default="dataset/shoppingbench_query/product_cache.json")
    parser.add_argument("--product-cache-mode", choices=["validate", "build", "skip"], default="validate")
    parser.add_argument("--documents-file", default="resources/documents.jsonl.gz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model_name, trust_remote_code=True)
    reports = {}
    if not args.skip_sft:
        reports["sft"] = prepare_sft(args, tokenizer)
    if not args.skip_query:
        reports["query"] = prepare_query(args, tokenizer)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
