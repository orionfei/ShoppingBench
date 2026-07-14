#!/usr/bin/env python3
"""Build a query-unique DAPO proposal view from prior raw group outcomes.

The canonical train split is never modified.  This view changes only which
queries are proposed to the online mixed-group filter; reward and GSPO updates
remain fully on-policy.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd


QUERY_RE = re.compile(r'[,\{]"query":("(?:\\.|[^"\\])*")\s*,"state"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/shoppingbench_query_rl_v3/train.parquet")
    parser.add_argument(
        "--raw-dir",
        default="rollouts/step108_outcome_grpo_v3_dapo_20260711_054926/train/raw_dynamic",
    )
    parser.add_argument(
        "--output", default="dataset/shoppingbench_query_rl_v3/train_dapo_mixed_priority.parquet"
    )
    parser.add_argument("--report", default="dataset/shoppingbench_query_rl_v3/dapo_mixed_priority_report.json")
    parser.add_argument("--min-mixed-observations", type=int, default=1)
    return parser.parse_args()


def consume_run(rows: list[dict], stats: dict[str, collections.Counter]) -> int:
    groups = 0
    if len(rows) % 8:
        raise ValueError(f"raw prompt run has {len(rows)} trajectories, expected a multiple of G=8")
    for start in range(0, len(rows), 8):
        chunk = rows[start : start + 8]
        match = QUERY_RE.search(chunk[0]["input"])
        if not match:
            raise ValueError("could not extract query from raw rollout prompt")
        query = json.loads(match.group(1))
        stats[query][chunk[0]["dynamic_group_state"]] += 1
        groups += 1
    return groups


def collect_stats(raw_dir: str) -> tuple[dict[str, collections.Counter], int]:
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    groups = 0
    paths = sorted(
        glob.glob(os.path.join(raw_dir, "*.jsonl")),
        key=lambda path: int(Path(path).stem),
    )
    if not paths:
        raise FileNotFoundError(f"no raw dynamic rollout files under {raw_dir}")
    for path in paths:
        previous = None
        run: list[dict] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                current = row["input"]
                if previous is not None and current != previous:
                    groups += consume_run(run, stats)
                    run = []
                previous = current
                run.append(row)
        groups += consume_run(run, stats)
    return stats, groups


def main() -> None:
    args = parse_args()
    stats, group_count = collect_stats(args.raw_dir)
    raw_unique_queries = len(stats)
    train = pd.read_parquet(args.train)
    queries = train["extra_info"].map(lambda value: value["query"])
    selected = queries.map(lambda query: stats[query]["mixed"] >= args.min_mixed_observations)
    proposal = train[selected].copy().reset_index(drop=True)
    if proposal.empty:
        raise RuntimeError("priority proposal view is empty")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    proposal.to_parquet(output, index=False)

    selected_queries = set(queries[selected])
    selected_observations = sum(sum(stats[query].values()) for query in selected_queries)
    selected_mixed = sum(stats[query]["mixed"] for query in selected_queries)
    report = {
        "canonical_train": args.train,
        "canonical_rows": int(len(train)),
        "proposal_output": str(output),
        "proposal_rows": int(len(proposal)),
        "query_unique": int(proposal["extra_info"].map(lambda value: value["query"]).nunique()),
        "raw_dir": args.raw_dir,
        "raw_groups": group_count,
        "raw_unique_queries": raw_unique_queries,
        "min_mixed_observations": args.min_mixed_observations,
        "selected_historical_groups": selected_observations,
        "selected_historical_mixed_groups": selected_mixed,
        "selected_historical_mixed_rate": selected_mixed / selected_observations,
        "semantics": (
            "proposal distribution only; online G=8 outcome scoring and mixed-group filtering remain required"
        ),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
