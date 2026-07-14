#!/usr/bin/env python3
"""Analyze the GSPO pilot and compare it with the matched DAPO-GRPO run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from analyze_rl_v3_dapo import validation_record


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    ROOT
    / "reports/step108_outcome_grpo_v3_dapo_20260711_054926_b32"
    / "analysis/analysis.json"
)
DEFAULT_REUSED_STEP0 = (
    ROOT
    / "rollouts/step108_outcome_gspo_v3_dapo_fast64_20260714_063127"
    / "validation/0.jsonl"
)


def savefig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reused-step0", type=Path, default=DEFAULT_REUSED_STEP0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_rl_v3_dapo.py"),
            str(args.run_root),
            "--output-dir",
            str(args.output_dir),
            "--figure-dir",
            str(args.figure_dir),
        ],
        check=True,
    )
    current = json.loads((args.output_dir / "analysis.json").read_text())
    baseline = json.loads(args.baseline_analysis.read_text())

    if args.reused_step0.exists():
        step0 = validation_record(args.reused_step0)
        if step0 is not None:
            current["validation"] = [step0] + [row for row in current["validation"] if row["step"] != 0]
            current["reused_step0_source"] = str(args.reused_step0)
            (args.output_dir / "analysis.json").write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n"
            )
            with (args.output_dir / "validation.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(current["validation"][0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(current["validation"])

            rows = current["validation"]
            x = [row["step"] for row in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, [row["terminal_asr"] for row in rows], marker="o", label="terminal ASR")
            ax.plot(x, [row["paper_asr"] for row in rows], marker="s", label="paper ASR")
            ax.plot(x, [row["mixed_rate"] for row in rows], marker="^", label="mixed-group rate")
            ax.fill_between(
                x,
                [row["terminal_ci95"][0] for row in rows],
                [row["terminal_ci95"][1] for row in rows],
                color="C0",
                alpha=0.12,
                label="terminal ASR bootstrap 95% CI",
            )
            ax.set(xlabel="effective optimizer step", ylabel="rate", ylim=(0, 1))
            ax.grid(alpha=0.25)
            ax.legend()
            savefig(fig, args.figure_dir / "01_validation_asr_mixed")

    fig, ax = plt.subplots(figsize=(7, 4))
    for report, label, color in (
        (baseline, "DAPO-GRPO (4 GPU)", "C1"),
        (current, "GSPO + same dynamic sampler (8 GPU)", "C0"),
    ):
        rows = report["validation"]
        ax.plot(
            [row["step"] for row in rows],
            [row["terminal_asr"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    ax.set(xlabel="effective optimizer step", ylabel="validation terminal ASR", ylim=(0, 0.6))
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(fig, args.figure_dir / "09_gspo_vs_grpo_validation")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for report, label, color in (
        (baseline, "DAPO-GRPO", "C1"),
        (current, "GSPO", "C0"),
    ):
        rows = [row for row in report["metrics"] if row.get("step", 0) > 0]
        x = [row["step"] for row in rows]
        axes[0].plot(x, [row.get("actor/entropy", np.nan) for row in rows], label=label, color=color)
        axes[1].plot(x, [row.get("actor/grad_norm", np.nan) for row in rows], label=label, color=color)
        axes[2].plot(x, [row.get("actor/pg_clipfrac", np.nan) for row in rows], label=label, color=color)
    axes[0].set_ylabel("entropy")
    axes[1].set_ylabel("grad norm")
    axes[2].set(xlabel="effective optimizer step", ylabel="clip fraction", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    savefig(fig, args.figure_dir / "10_gspo_vs_grpo_optimization")

    summary = {
        "comparison": "GSPO versus matched outcome-only DAPO-GRPO",
        "important_caveat": "GPU count differs (8 versus 4); algorithmic metrics are comparable, wall time is not a pure algorithm comparison.",
        "baseline_validation": baseline["validation"],
        "gspo_validation": current["validation"],
        "gspo_reused_step0_source": current.get("reused_step0_source"),
        "gspo_last_step": max((row.get("step", 0) for row in current["metrics"]), default=0),
    }
    (args.output_dir / "gspo_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
