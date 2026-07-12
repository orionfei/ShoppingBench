#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter successful teacher rollouts by audited token length.")
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queries")
    parser.add_argument("--length-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="total_tokens")
    parser.add_argument("--max-length", type=int, default=10240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = ROOT / args.rollout
    manifest_path = ROOT / args.manifest
    length_report_path = ROOT / args.length_report
    output_dir = ROOT / args.output_dir

    rollouts = read_jsonl(rollout_path)
    manifests = read_jsonl(manifest_path)
    queries = read_jsonl(ROOT / args.queries) if args.queries else None
    length_report = json.loads(length_report_path.read_text(encoding="utf-8"))
    records = length_report["records"]

    if len(rollouts) != len(manifests) or len(rollouts) != len(records):
        raise ValueError(
            f"length mismatch: rollouts={len(rollouts)} manifests={len(manifests)} records={len(records)}"
        )
    if queries is not None and len(queries) != len(rollouts):
        raise ValueError(f"query length mismatch: queries={len(queries)} rollouts={len(rollouts)}")

    kept_rollouts = []
    kept_manifests = []
    kept_queries = []
    kept_records = []
    dropped_records = []

    for idx, (trajectory, manifest, record) in enumerate(zip(rollouts, manifests, records, strict=True)):
        if record.get("clean_output_idx") != idx:
            raise ValueError(f"length report index mismatch at {idx}: {record.get('clean_output_idx')}")
        value = int(record[args.metric])
        if value <= args.max_length:
            kept_rollouts.append(trajectory)
            kept_manifests.append(manifest)
            kept_records.append(record)
            if queries is not None:
                kept_queries.append(queries[idx])
        else:
            dropped_records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "clean_success_rollout.jsonl", kept_rollouts)
    write_jsonl(output_dir / "clean_success_manifest.jsonl", kept_manifests)
    if queries is not None:
        write_jsonl(output_dir / "clean_success_queries.jsonl", kept_queries)

    report = {
        "input_rollout": args.rollout,
        "input_manifest": args.manifest,
        "input_queries": args.queries,
        "length_report": args.length_report,
        "output_dir": args.output_dir,
        "metric": args.metric,
        "max_length": args.max_length,
        "input_count": len(rollouts),
        "kept_count": len(kept_rollouts),
        "dropped_count": len(dropped_records),
        "kept_steps": sum(len(row) for row in kept_rollouts),
        "dropped_steps": sum(int(row["steps"]) for row in dropped_records),
        "max_kept_metric": max((int(row[args.metric]) for row in kept_records), default=None),
        "max_kept_response_tokens": max((int(row["response_tokens"]) for row in kept_records), default=None),
        "dropped": dropped_records,
        "files": {
            "rollout": str(output_dir / "clean_success_rollout.jsonl"),
            "manifest": str(output_dir / "clean_success_manifest.jsonl"),
            "queries": str(output_dir / "clean_success_queries.jsonl") if queries is not None else None,
            "report": str(output_dir / "filter_report.json"),
        },
    }
    write_json(output_dir / "filter_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
