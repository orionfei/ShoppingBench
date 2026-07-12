#!/usr/bin/env python3
"""Remove selected accepted sample IDs from paired generator JSONL outputs."""

import argparse
import json
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, values):
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    queries = rows(args.queries)
    metadata_all = rows(args.metadata)
    accepted = [row for row in metadata_all if row.get("status", "accepted") == "accepted"]
    if len(queries) != len(accepted):
        raise RuntimeError(f"query/accepted metadata mismatch: {len(queries)} != {len(accepted)}")
    rejected = set(args.sample_id)
    pairs = [(query, meta) for query, meta in zip(queries, accepted) if meta.get("sample_id") not in rejected]
    failed = [row for row in metadata_all if row.get("status") != "accepted"]
    found = {meta.get("sample_id") for _query, meta in zip(queries, accepted)} & rejected
    if found != rejected:
        raise RuntimeError(f"requested IDs not found: {sorted(rejected - found)}")
    write(args.queries, [query for query, _meta in pairs])
    write(args.metadata, failed + [meta for _query, meta in pairs])
    print({"removed": sorted(rejected), "remaining": len(pairs)})


if __name__ == "__main__":
    main()
