#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
sys.path.insert(0, str(AGENT_SRC))

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from pyserini.search.lucene import LuceneSearcher

from rewards.orm import (
    ground_truth_reward,
    length_reward,
    rule_score_reward,
    web_response_score_reward,
    web_rule_score_reward,
)
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES


FIELDS = ["title", "price", "service", "sku & attrs"]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rollout_outputs(path: Path):
    outputs = {}
    for row in load_jsonl(path):
        if not row:
            continue
        query = row[0]["extra_info"]["query"]
        outputs[query] = row
    return outputs


def load_rewards(path: Path):
    rewards = {}
    vouchers = {}
    web_rewards = {}
    for row in load_jsonl(path):
        query = row["query"]
        rewards[query] = row["reward"]
        if row.get("voucher"):
            vouchers[query] = row["voucher"]
        if row.get("Knowledge_Attribute") is not None:
            web_rewards[query] = (row["reward"], str(row["Knowledge_Attribute"]))
    return rewards, vouchers, web_rewards


def extract_recommended_product_ids(output):
    product_ids = ""
    for step in output:
        message = step["completion"].get("message") or {}
        for call in message.get("tool_call") or []:
            if call.get("name") == "recommend_product":
                product_ids = call.get("parameters", {}).get("product_ids", "")
    return product_ids if isinstance(product_ids, str) else ""


def has_tool(output, tool_name):
    for step in output:
        message = step["completion"].get("message") or {}
        if any(call.get("name") == tool_name for call in message.get("tool_call") or []):
            return True
    return False


def set_product_score(product, reward):
    score = defaultdict(float)
    score["product"] = 1
    score["gt"] = ground_truth_reward(product, reward)
    rule_score, total_counter, hit_counter = rule_score_reward(product, reward)
    score["rule"] = rule_score
    for field in FIELDS:
        total = total_counter.get(field, 0)
        score[field] = hit_counter.get(field, 0) / total if total > 0 else 1
    return score


def format_score(output, model_name, mode):
    if model_name == "human":
        return 0
    if not output:
        return 0
    total = 0
    for step in output:
        message = Message.from_dict(step["completion"].get("message") or {})
        completion = message.to_string(OUTPUT_ROLES)
        total += format_reward(completion) if mode == "think" else format_reward(completion, ["tool_call"])
    return total / len(output)


def base_score(output, model_name, mode):
    return {
        "steps": len(output),
        "length": length_reward(output),
        "format": format_score(output, model_name, mode),
        "recommended_product_ids": extract_recommended_product_ids(output),
        "has_recommend": 1 if has_tool(output, "recommend_product") else 0,
        "has_terminate": 1 if has_tool(output, "terminate") else 0,
    }


def eval_product(output, reward, searcher):
    score = base_score(output, "", "think")
    product_id = score["recommended_product_ids"].split(",")[0]
    doc = searcher.doc(product_id) if product_id else None
    if doc:
        product = json.loads(doc.raw())["product"]
        score.update(set_product_score(product, reward))
    else:
        score.update({"product": 0, "gt": 0, "rule": 0, **{field: 0 for field in FIELDS}})
    score["success"] = 1 if score["rule"] >= 1 else 0
    return score


def eval_shop(output, reward, searcher):
    score = base_score(output, "", "think")
    product_ids = score["recommended_product_ids"].split(",") if score["recommended_product_ids"] else []
    shop_ids = set()
    subtotals = defaultdict(float)
    for idx, sub_reward in enumerate(reward):
        if idx >= len(product_ids):
            continue
        doc = searcher.doc(product_ids[idx])
        if not doc:
            continue
        product = json.loads(doc.raw())["product"]
        sub_score = set_product_score(product, sub_reward)
        for key, value in sub_score.items():
            subtotals[key] += value
        shop_ids.add(product["shop_id"])

    denom = len(reward) if reward else 1
    for key in ["product", "gt", "rule", *FIELDS]:
        score[key] = subtotals[key] / denom
    score["shop"] = 1 if score["product"] >= 1 and len(shop_ids) == 1 else 0
    score["success"] = 1 if score["rule"] >= 1 and score["shop"] >= 1 else 0
    return score


def eval_voucher(output, reward, voucher, searcher):
    score = base_score(output, "", "think")
    product_ids = score["recommended_product_ids"].split(",") if score["recommended_product_ids"] else []
    shop_ids = set()
    total_price = 0
    subtotals = defaultdict(float)
    num_hits = 0
    for idx, sub_reward in enumerate(reward):
        if idx >= len(product_ids):
            continue
        doc = searcher.doc(product_ids[idx])
        if not doc:
            continue
        product = json.loads(doc.raw())["product"]
        sub_score = set_product_score(product, sub_reward)
        for key, value in sub_score.items():
            subtotals[key] += value
        shop_ids.add(product["shop_id"])
        total_price += product["price"]
        num_hits += 1

    denom = len(reward) if reward else 1
    for key in ["product", "gt", "rule", *FIELDS]:
        score[key] = subtotals[key] / denom

    budget = 0
    if voucher and num_hits == len(reward):
        if total_price <= voucher["budget"]:
            budget = 1
        elif voucher["voucher_type"] == "platform" or (
            voucher["voucher_type"] == "shop" and len(shop_ids) == 1
        ):
            if total_price >= voucher["threshold"]:
                if voucher["discount_type"] == "fixed":
                    discounted = total_price - voucher["face_value"]
                elif voucher["discount_type"] == "percentage":
                    discounted = max(total_price * (1 - voucher["discount"]), total_price - voucher["cap"])
                else:
                    discounted = total_price
                budget = 1 if discounted <= voucher["budget"] else 0
    score["budget"] = budget
    score["success"] = 1 if score["rule"] >= 1 and score["budget"] >= 1 else 0
    return score


def eval_web(output, reward, key_attribute, searcher):
    score = base_score(output, "", "think")
    product_id = score["recommended_product_ids"].split(",")[0]
    score["gt"] = 1 if product_id and reward["product_id"] in product_id else 0
    doc = searcher.doc(product_id) if product_id else None
    if doc:
        product = json.loads(doc.raw())["product"]
        reward = dict(reward)
        reward["key_attribute"] = key_attribute
        score["kw"], score["title"] = web_rule_score_reward(product, reward)
    else:
        score["kw"], score["title"] = 0, 0
    response = "\n".join((step["completion"].get("message") or {}).get("response", "") for step in output)
    score["response"] = max(score["kw"], web_response_score_reward(response, key_attribute))
    score["rule"] = (score["kw"] + score["title"]) / 2
    score["success"] = 1 if score["rule"] >= 1 else 0
    return score


def numeric_summary(items, keys):
    summary = {}
    for key in keys:
        values = [float(item.get(key, 0)) for item in items]
        if not values:
            summary[key] = {"mean": 0, "variance": 0, "sample_variance": 0}
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "variance": statistics.pvariance(values),
            "sample_variance": statistics.variance(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def analyze(config, index_dir):
    rollout_file = ROOT / config["rollout_file"]
    synthesize_file = ROOT / config["synthesize_file"]
    rollout_outputs = load_rollout_outputs(rollout_file)
    rewards, vouchers, web_rewards = load_rewards(synthesize_file)
    searcher = LuceneSearcher(index_dir)
    mode = "no think" if "ablation_react" in config["rollout_file"] else "think"
    model_name = config["model_config"]["model"]

    per_query = []
    missing = []
    for query, output in rollout_outputs.items():
        if config["task"] == "web":
            if query not in web_rewards:
                missing.append(query)
                continue
            reward, key_attribute = web_rewards[query]
            score = eval_web(output, reward, key_attribute, searcher)
        else:
            if query not in rewards:
                missing.append(query)
                continue
            reward = rewards[query]
            if config["task"] == "product":
                score = eval_product(output, reward, searcher)
            elif config["task"] == "shop":
                score = eval_shop(output, reward, searcher)
            elif config["task"] == "voucher":
                score = eval_voucher(output, reward, vouchers.get(query), searcher)
            else:
                raise ValueError(f"Unsupported task: {config['task']}")
        score["format"] = format_score(output, model_name, mode)
        score["query"] = query
        per_query.append(score)

    keys = list(dict.fromkeys([
        "success",
        "gt",
        "rule",
        "format",
        "length",
        "steps",
        "product",
        "has_recommend",
        "has_terminate",
        *FIELDS,
        "shop",
        "budget",
        "kw",
        "title",
        "response",
    ]))
    return {
        "model": model_name,
        "task": config["task"],
        "rollout_file": config["rollout_file"],
        "synthesize_file": config["synthesize_file"],
        "index_dir": index_dir,
        "num_rollouts": len(rollout_outputs),
        "num_scored": len(per_query),
        "num_missing_rewards": len(missing),
        "summary": numeric_summary(per_query, keys),
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--index-dir", default=os.getenv("INDEX_DIR", "indexes"))
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with (ROOT / args.config).open(encoding="utf-8") as fin:
        config = json.load(fin)
    result = analyze(config, args.index_dir)

    if args.output_json:
        output_path = ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)
            fout.write("\n")

    print(json.dumps({k: v for k, v in result.items() if k != "per_query"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
