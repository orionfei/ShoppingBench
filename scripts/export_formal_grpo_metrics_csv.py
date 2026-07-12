#!/usr/bin/env python3
"""Export the supervisor's trainer-metrics JSONL as a flat machine-readable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    columns = ["step", "captured_unix"] + sorted({key for row in rows for key in row} - {"step", "captured_unix"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}: {len(rows)} records, {len(columns)} columns")


if __name__ == "__main__":
    main()
