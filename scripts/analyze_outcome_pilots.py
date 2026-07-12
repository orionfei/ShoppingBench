#!/usr/bin/env python3
"""Combine pilot validation rollouts and PPO console metrics into one report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_sweep_module():
    path = ROOT / "scripts" / "analyze_outcome_sampling_sweep.py"
    spec = importlib.util.spec_from_file_location("outcome_sweep_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def console_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Parse VERL's `[step N] key=value, ...` console records."""
    result: dict[int, dict[str, float]] = {}
    if not path.exists():
        return result
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ansi.sub("", raw_line)
        match = re.search(r"\[step\s+(\d+)\]\s+(.*)", line)
        if not match:
            continue
        step, body = int(match.group(1)), match.group(2)
        values = result.setdefault(step, {})
        for item in body.split(", "):
            if "=" not in item:
                continue
            key, raw = item.rsplit("=", 1)
            value = finite(raw)
            if value is not None:
                values[key.strip()] = value
    return result


def auc(points: list[tuple[int, float]]) -> float | None:
    points = sorted(set(points))
    if len(points) < 2:
        return None
    area = sum((right_x-left_x) * (left_y+right_y) / 2 for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]))
    width = points[-1][0] - points[0][0]
    return area / width if width else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dirs", nargs="+", help="Pilot directories containing manifest, log, and validation JSONL")
    parser.add_argument("--output", default="reports/step108_outcome_pilots_20260710/analysis.json")
    parser.add_argument("--sweep-analysis", default=None, help="Optional sweep report whose runs are retained for plotting")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    sweep = load_sweep_module()
    analysis_args = SimpleNamespace(
        group_size=8, bootstrap=args.bootstrap, bootstrap_seed=20260710,
        format_min=.98, max_failure_rate=.01, workflow_drop_max=.05,
        baseline_workflow=None, temperature=None, top_p=None, seed=None,
    )
    pilot_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for raw_dir in args.pilot_dirs:
        pilot_dir = Path(raw_dir).resolve()
        manifest_path = pilot_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        candidate = str(manifest.get("candidate") or pilot_dir.name)
        metrics = console_metrics(pilot_dir / "run.log")
        candidate_runs: list[dict[str, Any]] = []
        for path in sorted((pilot_dir / "validation").glob("*.jsonl"), key=lambda item: int(item.stem)):
            rows = sweep.read_jsonl(path)
            metadata = sweep.canonical_metadata(manifest, path, analysis_args)
            metadata.update({"candidate": candidate, "training_step": int(path.stem), "source": str(path)})
            run = sweep.summarize_run(rows, metadata, analysis_args)
            step_metrics = metrics.get(int(path.stem), {})
            run.update({
                "kl": step_metrics.get("actor/ppo_kl"),
                "entropy": step_metrics.get("actor/entropy"),
                "clip_fraction": step_metrics.get("actor/pg_clipfrac"),
                "ppo_metrics": step_metrics,
            })
            candidate_runs.append(run)
            pilot_runs.append(run)

        terminal_points = [(int(run["training_step"]), float(run["terminal_asr"])) for run in candidate_runs if run.get("terminal_asr") is not None]
        mixed_points = [(int(run["training_step"]), float(run["mixed_terminal_asr_group_rate"])) for run in candidate_runs if run.get("mixed_terminal_asr_group_rate") is not None]
        last = max(candidate_runs, key=lambda run: int(run["training_step"])) if candidate_runs else {}
        summaries.append({
            "candidate": candidate,
            "temperature": manifest.get("temperature"),
            "top_p": manifest.get("top_p"),
            "training_seed": manifest.get("training_seed"),
            "validation_steps": [run["training_step"] for run in candidate_runs],
            "terminal_asr_auc": auc(terminal_points),
            "mixed_terminal_asr_group_rate_auc": auc(mixed_points),
            "step12_terminal_asr": last.get("terminal_asr"),
            "step12_paper_asr": last.get("paper_asr"),
            "step12_mixed_terminal_asr_group_rate": last.get("mixed_terminal_asr_group_rate"),
            "max_truncation_rate": max(((run.get("failure_counts") or {}).get("truncation", 0)/(run.get("rows") or 1) for run in candidate_runs), default=None),
            "max_abs_ppo_kl": max((abs(run["kl"]) for run in candidate_runs if run.get("kl") is not None), default=None),
            "max_clip_fraction": max((run["clip_fraction"] for run in candidate_runs if run.get("clip_fraction") is not None), default=None),
        })

    summaries.sort(key=lambda item: (
        item["terminal_asr_auc"] if item["terminal_asr_auc"] is not None else -1,
        item["step12_terminal_asr"] if item["step12_terminal_asr"] is not None else -1,
        item["step12_paper_asr"] if item["step12_paper_asr"] is not None else -1,
        item["step12_mixed_terminal_asr_group_rate"] if item["step12_mixed_terminal_asr_group_rate"] is not None else -1,
    ), reverse=True)
    for rank, item in enumerate(summaries, 1):
        item["rank_by_registered_outcome_order"] = rank

    base: dict[str, Any] = {}
    if args.sweep_analysis:
        base = json.loads(Path(args.sweep_analysis).read_text())
    report = {
        **base,
        "schema_version": 1,
        "pilot_metric_note": "kl is actor/ppo_kl (updated policy versus rollout/old policy), not reference-model KL",
        "pilot_summaries": summaries,
        "pilot_runs": pilot_runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}: {len(summaries)} pilots, {len(pilot_runs)} validation snapshots")


if __name__ == "__main__":
    main()
