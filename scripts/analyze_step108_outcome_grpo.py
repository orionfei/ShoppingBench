#!/usr/bin/env python3
"""Analyze one supervised formal GRPO attempt and select its best milestone."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_sweep():
    path = ROOT / "scripts/analyze_outcome_sampling_sweep.py"
    spec = importlib.util.spec_from_file_location("outcome_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_metrics(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        result.setdefault(step, {}).update({key: value for key, value in row.items() if isinstance(value, (int, float))})
    return result


def metric_at(metrics: dict[int, dict[str, float]], step: int, key: str) -> float | None:
    value = metrics.get(step, {}).get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def summarize_files(
    directory: Path, manifest: dict[str, Any], metrics: dict[int, dict[str, float]], sweep,
    *, validation: bool,
) -> list[dict[str, Any]]:
    args = SimpleNamespace(group_size=8, bootstrap=10000, bootstrap_seed=20260710)
    results = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda item: int(item.stem)):
        step = int(path.stem)
        metadata = {
            "run_id": manifest.get("experiment_name"), "group_size": 8, "seed": 108,
            "checkpoint": "global_step_108", "training_step": step, "source": str(path),
            "temperature": manifest.get("validation_temperature" if validation else "train_temperature"),
            "top_p": manifest.get("validation_top_p" if validation else "train_top_p"),
            "max_response_length": 10240, "rollout_max_num_seqs": 8,
        }
        summary = sweep.summarize_run(sweep.read_jsonl(path), metadata, args)
        summary["ppo_metrics"] = metrics.get(step, {})
        for output_key, metric_key in (
            ("entropy", "actor/entropy"), ("ppo_kl", "actor/ppo_kl"),
            ("clip_fraction", "actor/pg_clipfrac"), ("clip_fraction_lower", "actor/pg_clipfrac_lower"),
            ("grad_norm", "actor/grad_norm"), ("train_reward", "critic/score/mean"),
        ):
            summary[output_key] = metric_at(metrics, step, metric_key)
        results.append(summary)
    return results


def health(run: dict[str, Any], baseline_truncation: float) -> tuple[bool, list[str]]:
    reasons = []
    if (run.get("format_mean") or 0) < 0.98:
        reasons.append("format_below_0.98")
    if (run.get("infrastructure_failure_rate") or 0) > 0.01:
        reasons.append("infrastructure_failure_above_0.01")
    truncation = float(run.get("token_limit_noncompletion_rate") or 0)
    if truncation > 0.10 or truncation > baseline_truncation + 0.03:
        reasons.append("truncation_above_gate")
    for name in ("entropy", "ppo_kl", "clip_fraction", "clip_fraction_lower", "grad_norm"):
        value = run.get(name)
        if value is not None and not math.isfinite(float(value)):
            reasons.append(f"nonfinite_{name}")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve()
    manifest = json.loads((attempt / "manifest.json").read_text())
    metrics = load_metrics(attempt / "trainer_metrics.jsonl")
    sweep = load_sweep()
    validation = summarize_files(attempt / "validation", manifest, metrics, sweep, validation=True)
    training = summarize_files(attempt / "train", manifest, metrics, sweep, validation=False)
    baseline = next((run for run in validation if int(run["training_step"]) == 0), validation[0] if validation else {})
    baseline_terminal = float(baseline.get("terminal_asr") or 0)
    baseline_truncation = float(baseline.get("token_limit_noncompletion_rate") or 0)
    checkpoint_root = ROOT / "checkpoints/shoppingbench-rl-formal" / str(manifest["experiment_name"])
    saved_steps = sorted(int(path.name.rsplit("_", 1)[-1]) for path in checkpoint_root.glob("global_step_*") if path.is_dir())
    candidates = []
    for run in validation:
        ok, reasons = health(run, baseline_truncation)
        run["health_eligible"] = ok
        run["health_reasons"] = reasons
        run["is_saved_checkpoint"] = int(run["training_step"]) in saved_steps
        if run["is_saved_checkpoint"] and ok:
            candidates.append(run)
    candidates.sort(key=lambda run: (
        float(run.get("terminal_asr") or -1), float(run.get("paper_asr") or -1),
        -float(run.get("infrastructure_failure_rate") or 0), -float(run.get("token_limit_noncompletion_rate") or 0),
        -float(run.get("response_tokens_mean") or math.inf), -int(run["training_step"]),
    ), reverse=True)
    best = candidates[0] if candidates else baseline
    best_step = int(best.get("training_step") or 0)
    satisfied = bool(best_step > 0 and float(best.get("terminal_asr") or 0) >= baseline_terminal + 0.05)
    saved_terminal = [
        (int(run["training_step"]), float(run.get("terminal_asr") or 0))
        for run in validation if int(run["training_step"]) in saved_steps
    ]
    recent_improving = len(saved_terminal) >= 2 and saved_terminal[-1][1] - saved_terminal[-2][1] >= 0.02
    report = {
        "schema_version": 1,
        "attempt_dir": str(attempt), "manifest": manifest,
        "metric_note": "ppo_kl is updated policy versus rollout/old policy; reference-model KL loss/reward are disabled",
        "validation_runs": validation, "training_runs": training,
        "trainer_metrics": {str(key): value for key, value in sorted(metrics.items())},
        "saved_steps": saved_steps,
        "baseline": {"step": 0, "terminal_asr": baseline_terminal, "truncation": baseline_truncation},
        "best_checkpoint_step": best_step if best_step in saved_steps else None,
        "best_checkpoint_path": str(checkpoint_root / f"global_step_{best_step}") if best_step in saved_steps else None,
        "best_terminal_asr": best.get("terminal_asr"), "best_paper_asr": best.get("paper_asr"),
        "satisfied_terminal_plus_5pp": satisfied,
        "extension_eligible_recent_plus_2pp": recent_improving and not satisfied,
    }
    output = args.output or attempt / "analysis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation_steps": [run["training_step"] for run in validation],
        "training_steps": [run["training_step"] for run in training],
        "saved_steps": saved_steps, "baseline_terminal_asr": baseline_terminal,
        "best_checkpoint_step": report["best_checkpoint_step"], "best_terminal_asr": report["best_terminal_asr"],
        "satisfied": satisfied, "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
