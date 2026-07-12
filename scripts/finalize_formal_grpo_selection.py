#!/usr/bin/env python3
"""Freeze the formal checkpoint decision and optionally prune non-selected RL weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_inventory(path: Path) -> list[dict[str, object]]:
    suffixes = {".pt", ".safetensors", ".bin", ".json"}
    return [
        {"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": file_sha256(item)}
        for item in sorted(path.rglob("*")) if item.is_file() and item.suffix in suffixes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-nonbest", action="store_true")
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    best_step = analysis.get("best_checkpoint_step")
    best_path = Path(analysis.get("best_checkpoint_path") or "")
    if best_step is None or not best_path.is_dir():
        raise SystemExit("analysis has no existing health-eligible best checkpoint")
    saved_steps = analysis["saved_steps"]
    validations = []
    for run in analysis["validation_runs"]:
        validations.append({
            key: run.get(key) for key in (
                "training_step", "paper_asr", "terminal_asr", "mixed_terminal_asr_group_rate",
                "terminal_pass_at_g", "format_mean", "workflow_valid_mean",
                "infrastructure_failure_rate", "token_limit_noncompletion_rate", "response_tokens_mean",
                "health_eligible", "health_reasons", "is_saved_checkpoint",
            )
        })
    step0 = next(item for item in validations if item["training_step"] == 0)
    step80 = next(item for item in validations if item["training_step"] == 80)
    selected = next(item for item in validations if item["training_step"] == best_step)
    report = {
        "schema_version": 1,
        "decision_unix": time.time(),
        "analysis": str(args.analysis.resolve()),
        "completed_epochs": 2,
        "extended_to_three_epochs": False,
        "extension_reason": (
            "not extended: selected checkpoint did not reach step0+5pp and the last two saved milestones "
            "improved by only 0.78pp (<2pp extension gate)"
        ),
        "success_threshold_terminal_asr": float(step0["terminal_asr"]) + 0.05,
        "registered_success_achieved": bool(analysis["satisfied_terminal_plus_5pp"]),
        "best_checkpoint_step": best_step,
        "best_checkpoint_path": str(best_path.resolve()),
        "selection_reason": (
            "highest validation terminal ASR among health-eligible saved checkpoints; then paper ASR, "
            "health/truncation, response cost, and earlier step"
        ),
        "selected_metrics": selected,
        "final_step80_metrics": step80,
        "validation_runs": validations,
        "saved_steps_before_prune": saved_steps,
        "selected_checkpoint_inventory": model_inventory(best_path / "actor"),
        "pruned_checkpoint_steps": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prune_nonbest:
        checkpoint_root = best_path.parent
        for step in saved_steps:
            if int(step) == int(best_step):
                continue
            path = checkpoint_root / f"global_step_{step}"
            if path.is_dir():
                shutil.rmtree(path)
                report["pruned_checkpoint_steps"].append(step)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md = args.output.with_suffix(".md")
    rows = "\n".join(
        f"| {v['training_step']} | {float(v['terminal_asr']):.2%} | "
        f"{float(v['mixed_terminal_asr_group_rate']):.2%} | {float(v['terminal_pass_at_g']):.2%} | "
        f"{float(v['infrastructure_failure_rate']):.2%} | "
        f"{'yes' if v['health_eligible'] else 'no'} | {'yes' if v['is_saved_checkpoint'] else 'no'} |"
        for v in validations
    )
    md.write_text(
        "# Formal GRPO checkpoint selection\n\n"
        f"Selected `global_step_{best_step}`. Registered +5pp success achieved: "
        f"**{report['registered_success_achieved']}**. Training stopped at two epochs because the last "
        "two saved milestones improved by only 0.78pp, below the 2pp extension gate.\n\n"
        "| Step | Terminal ASR | Mixed | Pass@8 | Infra failure | Healthy | Saved |\n"
        "|---:|---:|---:|---:|---:|:---:|:---:|\n" + rows + "\n\n"
        f"Pruned non-selected RL checkpoints: `{report['pruned_checkpoint_steps']}`. The original SFT "
        "checkpoint was not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_step": best_step, "pruned": report["pruned_checkpoint_steps"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
