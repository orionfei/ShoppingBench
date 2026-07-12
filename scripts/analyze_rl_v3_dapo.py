#!/usr/bin/env python3
"""Aggregate and plot the RL-v3 DAPO training run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A live trainer may be in the middle of flushing the final line.
                continue
    return rows


def bootstrap_ci(values, seed=108, samples=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def validation_record(path: Path):
    rows = read_jsonl(path)
    if len(rows) != 512:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["input"]].append(int(float(row.get("terminal_asr", row.get("score", 0))) >= 1 - 1e-9))
    query_means = [sum(values) / len(values) for values in groups.values()]
    mixed = [int(0 < sum(values) < len(values)) for values in groups.values()]
    return {
        "step": int(path.stem), "trajectories": len(rows), "queries": len(groups),
        "terminal_asr": float(np.mean(query_means)), "paper_asr": float(np.mean([row["paper_asr"] for row in rows])),
        "mixed_rate": float(np.mean(mixed)), "terminal_ci95": bootstrap_ci(query_means, seed=108 + int(path.stem)),
        "format": float(np.mean([row.get("format", 0) for row in rows])),
        "workflow_valid": float(np.mean([row.get("workflow_valid", 0) for row in rows])),
        "truncation": float(np.mean([bool(row.get("length_truncated")) for row in rows])),
        "response_p50": float(np.percentile([row.get("response_tokens", 0) for row in rows], 50)),
        "response_p95": float(np.percentile([row.get("response_tokens", 0) for row in rows], 95)),
        "response_max": float(max(row.get("response_tokens", 0) for row in rows)),
        "server_error": float(np.mean([bool(row.get("server_error")) for row in rows])),
        "json_failure": float(np.mean([bool(row.get("json_decode_failure")) for row in rows])),
    }


def dynamic_record(path: Path):
    rows = read_jsonl(path)
    if not rows:
        return None
    states = Counter(row.get("dynamic_group_state") for row in rows[::8])
    lengths = [float(row.get("response_tokens", 0)) for row in rows]
    return {
        "step": int(path.stem), "trajectories": len(rows), "groups": len(rows) // 8,
        "all_fail": states["all_fail"], "mixed": states["mixed"], "all_success": states["all_success"],
        "mixed_yield": states["mixed"] / max(1, sum(states.values())),
        "raw_reward": float(np.mean([row.get("terminal_asr", row.get("score", 0)) for row in rows])),
        "format": float(np.mean([row.get("format", 0) for row in rows])),
        "truncation": float(np.mean([bool(row.get("length_truncated")) for row in rows])),
        "response_p50": float(np.percentile(lengths, 50)), "response_p95": float(np.percentile(lengths, 95)),
        "response_max": float(max(lengths)),
    }


def savefig(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_root / "trainer_metrics.jsonl"
    metrics = read_jsonl(metrics_path) if metrics_path.exists() else []
    validations = [validation_record(path) for path in sorted((args.run_root / "validation").glob("*.jsonl"), key=lambda p: int(p.stem))]
    dynamic = [dynamic_record(path) for path in sorted((args.run_root / "train/raw_dynamic").glob("*.jsonl"), key=lambda p: int(p.stem))]
    validations = [row for row in validations if row is not None]
    dynamic = [row for row in dynamic if row is not None]
    report = {"run_root": str(args.run_root), "metrics": metrics, "validation": validations, "dynamic": dynamic}
    (args.output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in (("validation", validations), ("dynamic", dynamic), ("trainer", metrics)):
        if rows:
            with (args.output_dir / f"{name}.csv").open("w", newline="") as handle:
                keys = sorted(set().union(*(row.keys() for row in rows)))
                writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)

    if validations:
        x = [row["step"] for row in validations]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, [row["terminal_asr"] for row in validations], marker="o", label="terminal ASR")
        ax.plot(x, [row["paper_asr"] for row in validations], marker="s", label="paper ASR")
        ax.plot(x, [row["mixed_rate"] for row in validations], marker="^", label="mixed-group rate")
        lo = [row["terminal_ci95"][0] for row in validations]
        hi = [row["terminal_ci95"][1] for row in validations]
        ax.fill_between(x, lo, hi, color="C0", alpha=.12, label="terminal ASR bootstrap 95% CI")
        ax.set(xlabel="effective optimizer step", ylabel="rate", ylim=(0, 1)); ax.grid(alpha=.25); ax.legend()
        savefig(fig, args.figure_dir / "01_validation_asr_mixed")

    if metrics:
        train = [row for row in metrics if row.get("step", 0) > 0]
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        x = [row["step"] for row in train]
        axes[0].plot(x, [row.get("critic/score/mean", np.nan) for row in train], label="accepted reward")
        axes[0].plot(x, [row.get("dynamic_sampling/acceptance_rate", np.nan) for row in train], label="buffer/raw")
        axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(x, [row.get("actor/entropy", np.nan) for row in train], label="entropy")
        axes[1].plot(x, [row.get("actor/pg_clipfrac", np.nan) for row in train], label="clip fraction")
        axes[1].set_xlabel("effective optimizer step"); axes[1].legend(); axes[1].grid(alpha=.25)
        savefig(fig, args.figure_dir / "02_train_reward_entropy")

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(x, [row.get("actor/grad_norm", np.nan) for row in train], label="grad norm")
        axes[0].plot(x, [row.get("actor/pg_clipfrac", np.nan) for row in train], label="upper clip fraction")
        axes[0].plot(x, [row.get("actor/pg_clipfrac_lower", np.nan) for row in train], label="lower clip fraction")
        axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(x, [row.get("actor/ppo_kl", np.nan) for row in train], label="diagnostic PPO KL")
        axes[1].plot(x, [row.get("actor/lr", np.nan) for row in train], label="learning rate")
        axes[1].axhline(0, color="black", linewidth=.6)
        axes[1].set_xlabel("effective optimizer step"); axes[1].legend(); axes[1].grid(alpha=.25)
        savefig(fig, args.figure_dir / "06_optimization_health")

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(x, [row.get("dynamic_sampling/gen_seconds", np.nan) for row in train], label="raw generation")
        axes[0].plot(x, [row.get("timing_s/update_actor", np.nan) for row in train], label="actor update")
        axes[0].plot(x, [row.get("timing_s/old_log_prob", np.nan) for row in train], label="old log-prob")
        axes[0].plot(x, [row.get("dynamic_sampling/reward_seconds", np.nan) for row in train], label="reward")
        axes[0].set_ylabel("seconds"); axes[0].legend(ncol=2); axes[0].grid(alpha=.25)
        axes[1].step(x, [row.get("dynamic_sampling/generation_batches", np.nan) for row in train], where="mid", label="generation chunks")
        axes[1].plot(x, [row.get("dynamic_sampling/mixed_groups", np.nan) / max(1, row.get("dynamic_sampling/generated_groups", 1)) for row in train], label="raw mixed yield")
        axes[1].set(xlabel="effective optimizer step", ylabel="chunks / rate"); axes[1].legend(); axes[1].grid(alpha=.25)
        savefig(fig, args.figure_dir / "07_timing_and_sampling_cost")

    if dynamic:
        x = [row["step"] for row in dynamic]
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(len(x))
        for key, color in (("all_fail", "#d95f5f"), ("mixed", "#55a868"), ("all_success", "#4c72b0")):
            values = np.array([row[key] for row in dynamic])
            ax.bar(x, values, bottom=bottom, label=key, color=color); bottom += values
        ax.set(xlabel="step", ylabel="raw query groups"); ax.legend(); ax.grid(axis="y", alpha=.2)
        savefig(fig, args.figure_dir / "03_dynamic_group_composition")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, [row["response_p50"] for row in dynamic], label="P50")
        ax.plot(x, [row["response_p95"] for row in dynamic], label="P95")
        ax.plot(x, [row["response_max"] for row in dynamic], label="max")
        ax2 = ax.twinx(); ax2.plot(x, [row["truncation"] for row in dynamic], color="red", label="truncation")
        ax.set(xlabel="step", ylabel="response tokens"); ax2.set_ylabel("truncation rate")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=.2)
        savefig(fig, args.figure_dir / "04_response_length")

    system_path = args.run_root / "system_metrics.csv"
    if system_path.exists():
        with system_path.open() as handle:
            system = list(csv.DictReader(handle))
        by_gpu = defaultdict(list)
        for row in system: by_gpu[row["gpu"]].append(row)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        start = min(float(row["unix"]) for row in system)
        for gpu, rows in sorted(by_gpu.items()):
            x = [(float(row["unix"]) - start) / 3600 for row in rows]
            axes[0].plot(x, [float(row["util_gpu_pct"]) for row in rows], label=f"GPU{gpu}")
            axes[1].plot(x, [float(row["memory_used_mib"]) / 1024 for row in rows], label=f"GPU{gpu}")
        axes[0].set_ylabel("GPU utilization %"); axes[1].set_ylabel("memory GiB"); axes[1].set_xlabel("hours")
        for ax in axes: ax.grid(alpha=.2); ax.legend(ncol=4)
        savefig(fig, args.figure_dir / "05_gpu_utilization")

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for gpu, rows in sorted(by_gpu.items()):
            x = [(float(row["unix"]) - start) / 3600 for row in rows]
            axes[0].plot(x, [float(row["power_w"]) for row in rows], label=f"GPU{gpu}")
            axes[1].plot(x, [float(row["temperature_c"]) for row in rows], label=f"GPU{gpu}")
        # Disk/checkpoint/rollout values are repeated once per GPU at each timestamp.
        rows0 = by_gpu[sorted(by_gpu)[0]]
        x0 = [(float(row["unix"]) - start) / 3600 for row in rows0]
        axes[2].plot(x0, [float(row["disk_free_gib"]) for row in rows0], label="disk free")
        axes[2].plot(x0, [float(row["checkpoint_gib"]) for row in rows0], label="checkpoint apparent size")
        axes[2].plot(x0, [float(row["rollout_gib"]) for row in rows0], label="rollout size")
        axes[0].set_ylabel("power W"); axes[1].set_ylabel("temperature C")
        axes[2].set(xlabel="hours", ylabel="GiB")
        for ax in axes: ax.grid(alpha=.2); ax.legend(ncol=4)
        savefig(fig, args.figure_dir / "08_hardware_and_storage")
    print(json.dumps({"steps": len(dynamic), "validation_points": len(validations), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
