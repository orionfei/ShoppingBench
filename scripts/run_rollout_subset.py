import argparse
import os
import sys
from pathlib import Path

import ujson as json


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
sys.path.insert(0, str(AGENT_SRC))

from run_rollout import react_loop  # noqa: E402


TASK_FILES = {
    "product": "data/synthesize_product_test.jsonl",
    "shop": "data/synthesize_shop_test.jsonl",
    "voucher": "data/synthesize_voucher_test.jsonl",
}


def load_queries(path: Path, limit: int) -> list[str]:
    queries = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            queries.append(json.loads(line)["query"])
            if len(queries) >= limit:
                break
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_FILES), required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--base-url", default="http://35.220.164.252:3888/v1")
    parser.add_argument("--model", default="bailian/deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running.")

    output = args.output or f"data/rollout_{args.task}_10_deepseek-v4-flash.jsonl"
    config = {
        "task": args.task,
        "system_prompt_file": "src/agent/prompt/rollout.md",
        "synthesize_file": TASK_FILES[args.task],
        "rollout_file": output,
        "base_url": args.base_url,
        "api_key": api_key,
        "exclude_tools": ["web_search"],
        "model_config": {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    }

    out_path = ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    queries = load_queries(ROOT / TASK_FILES[args.task], args.limit)
    for i, query in enumerate(queries, start=1):
        print(f"[{args.task}] {i}/{len(queries)}", flush=True)
        react_loop(query, config)


if __name__ == "__main__":
    main()
