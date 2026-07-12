#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fin:
        return json.load(fin)


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fout.write("\n")


def success_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report.get("per_query", [])
        if item.get("structured_failure_mode") == "success" or item.get("success") == 1.0
    ]


def has_empty_completion(trajectory: list[dict[str, Any]]) -> bool:
    for step in trajectory:
        content = (step.get("completion") or {}).get("content")
        if content is None or str(content).strip() == "":
            return True
    return False


def completion_text_tokens_proxy(trajectory: list[dict[str, Any]]) -> int:
    # Cheap stable proxy for report sorting/debugging; exact tokenizer stats are computed elsewhere.
    return sum(len(((step.get("completion") or {}).get("content") or "").split()) for step in trajectory)


def build_primary_query_index(primary_report: dict[str, Any]) -> dict[str, int]:
    query_to_idx = {}
    duplicates = []
    for item in primary_report.get("per_query", []):
        query = item.get("query")
        idx = item.get("idx")
        if query is None or idx is None:
            continue
        if query in query_to_idx:
            duplicates.append(query)
            continue
        query_to_idx[query] = int(idx)
    if duplicates:
        raise ValueError(f"Primary report has duplicate queries; cannot map retries safely: {len(duplicates)}")
    return query_to_idx


def load_run(run_dir: Path) -> dict[str, Any]:
    meta = read_json(run_dir / "meta.json")
    report = read_json(run_dir / "stage_reward_report.json")
    rollout_path = Path(meta.get("rollout_file") or run_dir / "rollout.jsonl")
    if not rollout_path.is_absolute():
        rollout_path = Path.cwd() / rollout_path
    rollouts = read_jsonl(rollout_path)
    if len(rollouts) != len(report.get("per_query", [])):
        raise ValueError(
            f"{run_dir}: rollout/report length mismatch: {len(rollouts)} vs {len(report.get('per_query', []))}"
        )
    sample_path = Path(meta.get("sample_file", ""))
    if sample_path and not sample_path.is_absolute():
        sample_path = Path.cwd() / sample_path
    samples = read_jsonl(sample_path) if sample_path.exists() else []
    return {"dir": run_dir, "meta": meta, "report": report, "rollouts": rollouts, "samples": samples}


def sample_for_original_idx(source_rows: list[Any], original_idx: int) -> Any:
    if 0 <= original_idx < len(source_rows):
        return source_rows[original_idx]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge successful ShoppingBench teacher rollouts from primary and retry runs.")
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--retry-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    primary = load_run(args.primary_run)
    retry = load_run(args.retry_run)
    query_to_original_idx = build_primary_query_index(primary["report"])

    source_file = Path(primary["meta"].get("source_file", ""))
    if source_file and not source_file.is_absolute():
        source_file = Path.cwd() / source_file
    source_rows = read_jsonl(source_file) if source_file.exists() else []

    merged: dict[int, dict[str, Any]] = {}
    duplicate_original_indices = []

    def add_successes(run: dict[str, Any], *, source_kind: str) -> None:
        run_name = run["meta"].get("run_name") or run["dir"].name
        for item in success_items(run["report"]):
            source_idx = int(item["idx"])
            query = item["query"]
            original_idx = source_idx if source_kind == "primary" else query_to_original_idx.get(query)
            if original_idx is None:
                raise ValueError(f"Could not map retry query to primary original idx: {query[:120]}")
            trajectory = run["rollouts"][source_idx]
            if original_idx in merged:
                duplicate_original_indices.append(
                    {
                        "original_idx": original_idx,
                        "kept_source_run": merged[original_idx]["manifest"]["source_run"],
                        "skipped_source_run": run_name,
                    }
                )
                continue
            merged[int(original_idx)] = {
                "trajectory": trajectory,
                "manifest": {
                    "original_idx": int(original_idx),
                    "source_kind": source_kind,
                    "source_run": run_name,
                    "source_idx": source_idx,
                    "query": query,
                    "steps": len(trajectory),
                    "score": item.get("score"),
                    "task": item.get("task"),
                    "recommended_ids": item.get("recommended_ids"),
                    "expected_ids": item.get("expected_ids"),
                    "has_empty_completion": has_empty_completion(trajectory),
                    "completion_word_proxy": completion_text_tokens_proxy(trajectory),
                },
                "query_row": sample_for_original_idx(source_rows, int(original_idx)),
            }

    add_successes(primary, source_kind="primary")
    add_successes(retry, source_kind="retry")

    ordered = [merged[idx] for idx in sorted(merged)]
    manifests = []
    trajectories = []
    query_rows = []
    clean_manifests = []
    clean_trajectories = []
    clean_query_rows = []

    for output_idx, item in enumerate(ordered):
        manifest = dict(item["manifest"])
        manifest["output_idx"] = output_idx
        manifests.append(manifest)
        trajectories.append(item["trajectory"])
        query_rows.append(item["query_row"])
        if not manifest["has_empty_completion"]:
            clean_manifests.append(dict(manifest, clean_output_idx=len(clean_manifests)))
            clean_trajectories.append(item["trajectory"])
            clean_query_rows.append(item["query_row"])

    by_source = Counter(item["source_kind"] for item in manifests)
    step_values = [item["steps"] for item in manifests]
    clean_step_values = [item["steps"] for item in clean_manifests]
    report = {
        "output_dir": str(args.output_dir),
        "primary_run": str(args.primary_run),
        "retry_run": str(args.retry_run),
        "source_file": str(source_file),
        "total_source_rows": len(source_rows) or len(primary["report"].get("per_query", [])),
        "success_count": len(manifests),
        "clean_success_count": len(clean_manifests),
        "by_source_kind": dict(by_source),
        "primary_success_count": by_source.get("primary", 0),
        "retry_success_count": by_source.get("retry", 0),
        "duplicate_original_indices_skipped": duplicate_original_indices,
        "empty_completion_success_count": len(manifests) - len(clean_manifests),
        "step_summary": {
            "total": sum(step_values),
            "mean": sum(step_values) / len(step_values) if step_values else 0,
            "min": min(step_values) if step_values else None,
            "max": max(step_values) if step_values else None,
        },
        "clean_step_summary": {
            "total": sum(clean_step_values),
            "mean": sum(clean_step_values) / len(clean_step_values) if clean_step_values else 0,
            "min": min(clean_step_values) if clean_step_values else None,
            "max": max(clean_step_values) if clean_step_values else None,
        },
        "files": {
            "success_rollout": str(args.output_dir / "success_rollout.jsonl"),
            "success_manifest": str(args.output_dir / "success_manifest.jsonl"),
            "success_queries": str(args.output_dir / "success_queries.jsonl"),
            "clean_success_rollout": str(args.output_dir / "clean_success_rollout.jsonl"),
            "clean_success_manifest": str(args.output_dir / "clean_success_manifest.jsonl"),
            "clean_success_queries": str(args.output_dir / "clean_success_queries.jsonl"),
            "report": str(args.output_dir / "report.json"),
        },
    }

    write_jsonl(args.output_dir / "success_rollout.jsonl", trajectories)
    write_jsonl(args.output_dir / "success_manifest.jsonl", manifests)
    write_jsonl(args.output_dir / "success_queries.jsonl", query_rows)
    write_jsonl(args.output_dir / "clean_success_rollout.jsonl", clean_trajectories)
    write_jsonl(args.output_dir / "clean_success_manifest.jsonl", clean_manifests)
    write_jsonl(args.output_dir / "clean_success_queries.jsonl", clean_query_rows)
    write_json(args.output_dir / "report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
