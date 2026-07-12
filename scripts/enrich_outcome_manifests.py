#!/usr/bin/env python3
"""Finalize outcome-run manifests with artifact counts and content hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "scripts/reward_shoppingbench_asr_batch.py",
    "scripts/analyze_outcome_sampling_sweep.py",
    "scripts/run_step108_outcome_sampling_sweep.sh",
    "src/rl/run_grpo_qwen3_1_7b_query_verl.sh",
    "src/rl/run_grpo_qwen3_4b_state_folded_a800.sh",
    "src/rl/verl/workers/reward_manager/batch.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    current_hashes = {item: sha256(ROOT / item) for item in CODE_FILES if (ROOT / item).is_file()}
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    for raw in args.run_dirs:
        run_dir = Path(raw).resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        files = sorted(path for path in run_dir.glob("*.jsonl") if path.is_file())
        manifest["trajectory_files"] = {
            path.name: {"sha256": sha256(path), "rows": jsonl_rows(path), "bytes": path.stat().st_size}
            for path in files
        }
        manifest["actual_trajectories"] = sum(item["rows"] for item in manifest["trajectory_files"].values())
        manifest["git_dirty"] = dirty
        if "code_files_sha256" not in manifest:
            manifest["code_files_sha256"] = current_hashes
            manifest["code_hash_capture"] = "post_run; reward semantics unchanged, diagnostics may have been instrumented during coarse sweep"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"enriched {manifest_path}: {manifest['actual_trajectories']} trajectories")


if __name__ == "__main__":
    main()
