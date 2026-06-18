#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl_files(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    row = json.loads(line)
                    row["_source_file"] = str(path)
                    rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    scores = np.array([float(row.get("score", 0.0)) for row in rows], dtype=np.float64)
    by_input = defaultdict(list)
    for row in rows:
        by_input[row.get("input", "")].append(float(row.get("score", 0.0)))
    per_prompt = []
    for input_text, vals in by_input.items():
        arr = np.array(vals, dtype=np.float64)
        per_prompt.append(
            {
                "input": input_text,
                "n": int(arr.size),
                "mean": float(arr.mean()) if arr.size else 0.0,
                "var": float(arr.var(ddof=0)) if arr.size else 0.0,
                "std": float(arr.std(ddof=0)) if arr.size else 0.0,
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
            }
        )
    variances = np.array([item["var"] for item in per_prompt], dtype=np.float64)
    means = np.array([item["mean"] for item in per_prompt], dtype=np.float64)
    return {
        "rows": len(rows),
        "prompts": len(per_prompt),
        "score_mean": float(scores.mean()) if scores.size else 0.0,
        "score_var": float(scores.var(ddof=0)) if scores.size else 0.0,
        "prompt_reward_mean_mean": float(means.mean()) if means.size else 0.0,
        "prompt_reward_var_mean": float(variances.mean()) if variances.size else 0.0,
        "prompt_reward_var_max": float(variances.max()) if variances.size else 0.0,
        "top_variance_prompts": sorted(per_prompt, key=lambda item: item["var"], reverse=True)[:10],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize reward variance from verl rollout JSONL dumps.")
    parser.add_argument("paths", nargs="+", help="JSONL files or directories containing JSONL files.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def expand_paths(raw_paths: list[str]) -> list[Path]:
    files = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)
    return files


def main():
    args = parse_args()
    files = expand_paths(args.paths)
    rows = read_jsonl_files(files)
    report = {"files": [str(path) for path in files], **summarize(rows)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
