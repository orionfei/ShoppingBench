#!/usr/bin/env python3
"""Supervise the four-GPU Step108 outcome-only DAPO run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "scripts/supervise_step108_outcome_grpo.py"
spec = importlib.util.spec_from_file_location("formal_supervisor_common", LEGACY_PATH)
COMMON = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(COMMON)


def hardlink_tree(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, copy_function=os.link, symlinks=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def sha256(path: Path) -> str:
    return COMMON.sha256(path)


def make_manifest(run_root: Path, experiment: str, workers: int) -> dict[str, Any]:
    files = [
        "dataset/shoppingbench_query_rl_v3/train.parquet",
        "dataset/shoppingbench_query_rl_v3/validation.parquet",
        "dataset/shoppingbench_query_rl_v3/test.parquet",
        "dataset/shoppingbench_query_rl_v3/product_cache.json",
        "scripts/run_step108_outcome_grpo_v3_dapo.sh",
        "scripts/supervise_step108_outcome_grpo_v3_dapo.py",
        "scripts/reward_shoppingbench_asr_batch.py",
        "src/rl/verl/trainer/ppo/ray_trainer.py",
    ]
    return {
        "schema_version": 1,
        "status": "starting",
        "experiment": experiment,
        "checkpoint": "global_step_108",
        "train_rows": 1414,
        "validation_rows": 64,
        "test_rows": 250,
        "gpus": 4,
        "group_size": 8,
        "effective_batch_groups": 32,
        "generation_batch_groups": 32,
        "max_generation_batches": 4,
        "effective_steps": 90,
        "save_steps": [23, 45, 68, 90],
        "validation_steps": [0, 11, 23, 34, 45, 56, 68, 79, 90],
        "agent_workers": workers,
        "temperature": 0.4,
        "top_p": 0.95,
        "validation_temperature": 0.2,
        "validation_top_p": 0.9,
        "learning_rate": 1e-6,
        "ratio_range": [0.8, 1.28],
        "use_kl_loss": False,
        "reward": "terminal_asr=paper_asr*terminate_success",
        "run_root": str(run_root),
        "hashes": {name: sha256(ROOT / name) for name in files},
        "started_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--agent-workers", type=int, default=8)
    parser.add_argument("--disk-warn-gib", type=float, default=18.0)
    parser.add_argument("--disk-stop-gib", type=float, default=12.0)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"step108_outcome_grpo_v3_dapo_{stamp}"
    run_root = (args.run_root or ROOT / "rollouts" / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    experiment = f"{run_id}_b32"
    checkpoint_root = ROOT / "checkpoints/shoppingbench-rl-v3-dapo" / experiment
    manifest = make_manifest(run_root, experiment, args.agent_workers)
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if not COMMON.search_server_ok():
        raise SystemExit("ShoppingBench search server is unavailable")
    if not COMMON.cleanup_ray_and_wait():
        raise SystemExit("GPUs are not clean")

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "EXPERIMENT_NAME": experiment,
        "ROLLOUT_AGENT_NUM_WORKERS": str(args.agent_workers),
        "ROLLOUT_DATA_DIR": str(run_root / "train"),
        "VALIDATION_DATA_DIR": str(run_root / "validation"),
    })
    process = subprocess.Popen(
        ["bash", "scripts/run_step108_outcome_grpo_v3_dapo.sh"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=COMMON.monitor_system,
        args=(process, run_root, checkpoint_root, stop_event, args.disk_stop_gib, args.disk_warn_gib),
        daemon=True,
    )
    monitor.start()
    metrics_path = run_root / "trainer_metrics.jsonl"
    log_path = run_root / "run.log"
    best_value = float("-inf")
    best_step = None
    saw_nan = False
    max_step = 0
    with log_path.open("w", encoding="utf-8") as log, metrics_path.open("w", encoding="utf-8") as metrics:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            clean = COMMON.ANSI.sub("", line)
            lowered = clean.lower()
            saw_nan = saw_nan or bool(re.search(r"(?:^|[=,: ])(?:nan|inf)(?:$|[, }])", lowered))
            record = COMMON.parse_step_record(clean)
            if record:
                max_step = max(max_step, int(record["step"]))
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
                print(COMMON.progress_summary(record), flush=True)
                dynamic = {
                    key: value for key, value in record.items()
                    if key.startswith("dynamic_sampling/")
                }
                if dynamic:
                    print(f"[supervisor:dynamic] step={record['step']} {json.dumps(dynamic)}", flush=True)
                value = COMMON.metric_value(record, "/terminal_asr/mean@8")
                step = int(record["step"])
                checkpoint = checkpoint_root / f"global_step_{step}"
                if value is not None and value > best_value and checkpoint.exists():
                    hardlink_tree(checkpoint, checkpoint_root / "best")
                    best_value, best_step = value, step
                    print(f"[supervisor:best] step={step} terminal_asr={value:.6f}", flush=True)
            elif any(token in clean for token in (
                "Initial validation metrics", "Total training steps", "Saved model", "Traceback",
                "dynamic-sampling", "out of memory",
            )):
                print(clean.rstrip()[-1600:], flush=True)
            if saw_nan:
                COMMON.stop_process_group(process, "NaN/Inf detected in trainer output")
                break
    code = process.wait()
    stop_event.set()
    monitor.join(timeout=10)
    COMMON.cleanup_ray_and_wait()
    checkpoints = sorted(str(path) for path in checkpoint_root.glob("global_step_*")) if checkpoint_root.exists() else []
    manifest.update({
        "status": "completed" if code == 0 and max_step >= 90 and not saw_nan else "failed",
        "return_code": code,
        "max_completed_step": max_step,
        "best_step": best_step,
        "best_validation_terminal_asr": None if best_step is None else best_value,
        "checkpoint_root": str(checkpoint_root),
        "checkpoints": checkpoints,
        "finished_unix": time.time(),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[supervisor:final] {json.dumps(manifest, ensure_ascii=False)}", flush=True)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
