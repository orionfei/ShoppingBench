#!/usr/bin/env python3
"""Launch and supervise the formal Step108 outcome-only GRPO run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RECORD = re.compile(r"\[step\s+(\d+)\]\s+(.*)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(raw: str) -> float | None:
    try:
        value = float(raw)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_step_record(raw_line: str) -> dict[str, Any] | None:
    line = ANSI.sub("", raw_line)
    match = STEP_RECORD.search(line)
    if not match:
        return None
    result: dict[str, Any] = {"step": int(match.group(1)), "captured_unix": time.time()}
    for item in match.group(2).split(", "):
        if "=" not in item:
            continue
        key, raw = item.rsplit("=", 1)
        value = finite(raw)
        if value is not None:
            result[key.strip()] = value
    return result


def metric_value(record: dict[str, Any], suffix: str) -> float | None:
    exact = [value for key, value in record.items() if key.endswith(suffix)]
    return exact[0] if exact else None


def progress_summary(record: dict[str, Any]) -> str:
    fields = {
        "train_reward": record.get("critic/score/mean"),
        "entropy": record.get("actor/entropy"),
        "clip": record.get("actor/pg_clipfrac"),
        "clip_low": record.get("actor/pg_clipfrac_lower"),
        "ppo_kl": record.get("actor/ppo_kl"),
        "grad": record.get("actor/grad_norm"),
        "resp": record.get("response_length/mean"),
        "val_terminal": metric_value(record, "/terminal_asr/mean@8"),
        "val_paper": metric_value(record, "/paper_asr/mean@8"),
        "val_format": metric_value(record, "/format/mean@8"),
        "val_trunc": metric_value(record, "/length_truncated/mean@8"),
    }
    values = " ".join(f"{key}={value:.5g}" for key, value in fields.items() if value is not None)
    return f"[supervisor:metrics] step={record['step']} {values}"


def search_server_ok() -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:5631/", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def stop_process_group(process: subprocess.Popen[str], reason: str) -> None:
    if process.poll() is not None:
        return
    print(f"[supervisor:stop] {reason}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def gpu_memory_used_mib() -> list[int]:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ], text=True)
        return [int(value.strip()) for value in output.splitlines() if value.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def cleanup_ray_and_wait(timeout_seconds: int = 90) -> bool:
    """Stop detached Ray workers and wait until both training GPUs are released."""
    ray = Path("/root/miniconda3/envs/shoppingbench-verl/bin/ray")
    if ray.exists():
        subprocess.run(
            [str(ray), "stop", "--force"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False, timeout=30,
        )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        used = gpu_memory_used_mib()
        if used and max(used) <= 1024:
            print(f"[supervisor:cleanup] Ray stopped; gpu_memory_mib={used}", flush=True)
            return True
        time.sleep(2)
    used = gpu_memory_used_mib()
    print(f"[supervisor:cleanup-failed] gpu_memory_mib={used}", flush=True)
    return False


def monitor_system(
    process: subprocess.Popen[str], attempt_dir: Path, checkpoint_dir: Path, stop_event: threading.Event,
    disk_stop_gib: float, disk_warn_gib: float,
) -> None:
    path = attempt_dir / "system_metrics.csv"
    columns = [
        "unix", "iso_utc", "gpu", "util_gpu_pct", "util_mem_pct", "memory_used_mib",
        "memory_total_mib", "power_w", "temperature_c", "disk_free_gib", "checkpoint_gib",
        "rollout_gib", "search_server_ok",
    ]
    warned = False
    server_failures = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        tick = 0
        while not stop_event.is_set() and process.poll() is None:
            now = time.time()
            disk = shutil.disk_usage(ROOT)
            free_gib = disk.free / 1024**3
            checkpoint_gib = directory_bytes(checkpoint_dir) / 1024**3
            rollout_gib = directory_bytes(attempt_dir) / 1024**3
            server = search_server_ok() if tick % 12 == 0 else True
            server_failures = server_failures + 1 if not server else 0
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ], text=True)
                rows = [line.split(", ") for line in output.strip().splitlines()]
            except Exception:
                rows = []
            for gpu in rows:
                writer.writerow({
                    "unix": now,
                    "iso_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "gpu": gpu[0], "util_gpu_pct": gpu[1], "util_mem_pct": gpu[2],
                    "memory_used_mib": gpu[3], "memory_total_mib": gpu[4], "power_w": gpu[5],
                    "temperature_c": gpu[6], "disk_free_gib": free_gib,
                    "checkpoint_gib": checkpoint_gib, "rollout_gib": rollout_gib,
                    "search_server_ok": int(server),
                })
            handle.flush()
            if free_gib < disk_warn_gib and not warned:
                warned = True
                print(f"[supervisor:disk-warning] free={free_gib:.2f}GiB checkpoint={checkpoint_gib:.2f}GiB", flush=True)
            if free_gib < disk_stop_gib:
                stop_process_group(process, f"disk free {free_gib:.2f}GiB < hard floor {disk_stop_gib:.2f}GiB")
                break
            if server_failures >= 3:
                stop_process_group(process, "search server failed three consecutive health checks")
                break
            if tick % 12 == 0:
                util = "/".join(row[1] for row in rows) if rows else "NA"
                mem = "/".join(row[3] for row in rows) if rows else "NA"
                print(
                    f"[supervisor:heartbeat] gpu_util={util}% gpu_mem={mem}MiB free_disk={free_gib:.1f}GiB ckpt={checkpoint_gib:.1f}GiB",
                    flush=True,
                )
            tick += 1
            stop_event.wait(5)


def manifest(attempt_dir: Path, experiment: str, batch_size: int, epochs: int) -> dict[str, Any]:
    steps_per_epoch = 643 // batch_size
    code_files = [
        ROOT / "scripts/run_step108_outcome_grpo_formal.sh",
        ROOT / "scripts/supervise_step108_outcome_grpo.py",
        ROOT / "scripts/reward_shoppingbench_asr_batch.py",
        ROOT / "src/rl/run_grpo_qwen3_4b_state_folded_a800.sh",
        ROOT / "src/rl/run_grpo_qwen3_1_7b_query_verl.sh",
    ]
    return {
        "schema_version": 1,
        "status": "starting",
        "experiment_name": experiment,
        "checkpoint": "global_step_108",
        "checkpoint_path": "checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108",
        "train_dataset": "dataset/shoppingbench_query_rl_v2/train.parquet",
        "train_dataset_sha256": sha256(ROOT / "dataset/shoppingbench_query_rl_v2/train.parquet"),
        "validation_dataset": "dataset/shoppingbench_query_rl_v2/validation.parquet",
        "validation_dataset_sha256": sha256(ROOT / "dataset/shoppingbench_query_rl_v2/validation.parquet"),
        "train_rows": 643, "validation_rows": 16, "batch_size": batch_size, "group_size": 8,
        "ppo_mini_batch_size": 16 if batch_size == 32 else 8,
        "steps_per_epoch": steps_per_epoch, "epochs": epochs,
        "total_training_steps": steps_per_epoch * epochs,
        "save_freq": steps_per_epoch // 2, "test_freq": steps_per_epoch // 4,
        "train_temperature": 0.4, "train_top_p": 0.95,
        "validation_temperature": 0.2, "validation_top_p": 0.9,
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.28,
        "probability_ratio_range": [0.8, 1.28], "learning_rate": 1e-6,
        "use_kl_loss": False, "use_kl_in_reward": False, "entropy_coeff": 0,
        "max_response_length": 10240, "max_num_seqs": 8, "engine_n": 1,
        "agent_workers": 8, "enable_prefix_caching": True, "stable_sampling": True,
        "free_cache_engine": True,
        "pytorch_cuda_alloc_conf": "expandable_segments:False",
        "ppo_max_token_len_per_gpu": 12288, "use_remove_padding": True,
        "checkpoint_contents": ["model", "extra"], "max_actor_ckpt_to_keep": 4,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "code_files_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_files},
        "started_unix": time.time(), "attempt_dir": str(attempt_dir),
    }


def run_attempt(args: argparse.Namespace, batch_size: int, run_id: str) -> tuple[int, bool, int, Path]:
    experiment = f"{run_id}_b{batch_size}"
    attempt_dir = args.run_root / f"attempt_b{batch_size}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / "checkpoints" / "shoppingbench-rl-formal" / experiment
    raw_log = attempt_dir / "run.log"
    metrics_path = attempt_dir / "trainer_metrics.jsonl"
    state = manifest(attempt_dir, experiment, batch_size, args.epochs)
    (attempt_dir / "manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1", "EXPERIMENT_NAME": experiment,
        "TRAIN_BATCH_SIZE": str(batch_size), "TOTAL_EPOCHS": str(args.epochs),
        "ROLLOUT_DATA_DIR": str(attempt_dir / "train"),
        "VALIDATION_DATA_DIR": str(attempt_dir / "validation"),
        "RESUME_MODE": "disable",
    })
    command = ["bash", "scripts/run_step108_outcome_grpo_formal.sh"]
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_system,
        args=(process, attempt_dir, checkpoint_dir, stop_event, args.disk_stop_gib, args.disk_warn_gib),
        daemon=True,
    )
    monitor.start()
    saw_oom = False
    max_completed_step = 0
    milestones = ("Total training steps", "Size of train dataloader", "Initial validation metrics", "local_global_step_folder", "Saved model to")
    with raw_log.open("w", encoding="utf-8") as log_handle, metrics_path.open("w", encoding="utf-8") as metrics_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            clean = ANSI.sub("", line)
            saw_oom = saw_oom or "out of memory" in clean.lower() or "CUDA OOM" in clean
            record = parse_step_record(clean)
            if record:
                max_completed_step = max(max_completed_step, int(record["step"]))
                metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                print(progress_summary(record), flush=True)
            elif any(token in clean for token in milestones) or "out of memory" in clean.lower() or "Traceback" in clean:
                print(clean.rstrip()[-1200:], flush=True)
    return_code = process.wait()
    stop_event.set()
    monitor.join(timeout=10)
    checkpoints = sorted(checkpoint_dir.glob("global_step_*")) if checkpoint_dir.exists() else []
    state.update({
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code, "saw_oom": saw_oom,
        "max_completed_step": max_completed_step,
        "finished_unix": time.time(), "checkpoint_dirs": [str(path) for path in checkpoints],
    })
    (attempt_dir / "manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(
        f"[supervisor:attempt-end] batch={batch_size} code={return_code} oom={saw_oom} "
        f"max_completed_step={max_completed_step} checkpoints={len(checkpoints)}",
        flush=True,
    )
    return return_code, saw_oom, max_completed_step, attempt_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches", default="32,16,8")
    parser.add_argument("--disk-warn-gib", type=float, default=18.0)
    parser.add_argument("--disk-stop-gib", type=float, default=12.0)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"step108_outcome_grpo_v2_{timestamp}"
    args.run_root = (args.run_root or ROOT / "rollouts" / run_id).resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "supervisor.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "status": "running",
        "batch_fallback_order": [int(item) for item in args.batches.split(",")],
        "run_root": str(args.run_root), "started_unix": time.time(),
    }, indent=2) + "\n")
    if not search_server_ok():
        raise SystemExit("ShoppingBench search server is unavailable")
    if not cleanup_ray_and_wait():
        raise SystemExit("Training GPUs are not clean before the first attempt")
    final: dict[str, Any] | None = None
    for batch_size in [int(item) for item in args.batches.split(",")]:
        code, oom, max_completed_step, attempt_dir = run_attempt(args, batch_size, run_id)
        if code == 0:
            final = {"status": "completed", "batch_size": batch_size, "attempt_dir": str(attempt_dir)}
            cleanup_ray_and_wait()
            break
        if not cleanup_ray_and_wait():
            final = {
                "status": "failed", "reason": "GPU memory was not released after failed attempt",
                "batch_size": batch_size, "attempt_dir": str(attempt_dir), "return_code": code,
            }
            break
        if not oom:
            final = {"status": "failed", "batch_size": batch_size, "attempt_dir": str(attempt_dir), "return_code": code}
            break
        print(
            f"[supervisor:fallback] capacity OOM at batch={batch_size} after completed_step="
            f"{max_completed_step}; restarting clean step108 with smaller logical/PPO batch",
            flush=True,
        )
    if final is None:
        final = {"status": "failed", "reason": "all batch fallbacks exhausted"}
    final.update({"schema_version": 1, "run_id": run_id, "run_root": str(args.run_root), "finished_unix": time.time()})
    (args.run_root / "supervisor.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"[supervisor:final] {json.dumps(final)}", flush=True)
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
