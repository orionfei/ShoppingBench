#!/usr/bin/env python3
"""Render formal GRPO learning, optimization, response, and system figures."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_plot_module():
    path = ROOT / "scripts/plot_outcome_sampling_sweep.py"
    spec = importlib.util.spec_from_file_location("outcome_plot", path)
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


def line_figure(plot, output: Path, name: str, heading: str, panels: list[tuple[str, list[tuple[str, list[tuple[float, float]]]]]], xlabel: str = "training step") -> None:
    width, panel_h = 1400, 390
    canvas = plot.Canvas(width, 80 + panel_h * len(panels), heading)
    plot.title(canvas, heading)
    colors = (plot.BLUE, plot.RED, "#45a675", "#8f63b8", "#e6ab3c")
    for index, (ylabel, series) in enumerate(panels):
        box = (90, 90 + index * panel_h, width - 40, 90 + index * panel_h + 300)
        values = [y for _, points in series for _, y in points]
        steps = [x for _, points in series for x, _ in points]
        if not values or not steps:
            canvas.text(width / 2, sum((box[1], box[3])) / 2, f"{ylabel}: no data", 16, plot.GREY, "mm")
            continue
        xmin, xmax = min(steps), max(steps)
        if xmin == xmax:
            xmax += 1
        ymin, ymax = min(values), max(values)
        padding = max((ymax - ymin) * 0.12, 0.01)
        if ymin >= 0 and ymax <= 1:
            yrange = (0, 1)
        else:
            yrange = (ymin - padding, ymax + padding)
        transform = plot.axes(canvas, box, ylabel, xlabel, ylabel, (xmin, xmax), yrange)
        for color, (label, points) in zip(colors, series):
            if not points:
                continue
            pixels = [transform(x, y) for x, y in sorted(points)]
            canvas.line(pixels, color, 3)
            for x, y in pixels:
                canvas.circle(x, y, 4, color)
            canvas.text(box[2] - 8, box[1] + 18 * list(series).index((label, points)), label, 11, color, "ra")
    canvas.save(output, name)


def group_composition(plot, output: Path, validation: list[dict[str, Any]]) -> None:
    usable = [run for run in validation if run.get("terminal_group_counts")]
    canvas = plot.Canvas(1200, 620, "Validation terminal-ASR group composition")
    plot.title(canvas, "Validation all-fail / mixed / all-success composition")
    left, top, bottom, right = 100, 90, 500, 1160
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        canvas.line([(left, y), (right, y)], plot.GRID, 1)
        canvas.text(left - 10, y, f"{tick / 5:.1f}", 12, anchor="ra")
    slot = (right - left) / max(1, len(usable))
    for index, run in enumerate(usable):
        counts = run["terminal_group_counts"]
        total = sum(counts.values()) or 1
        x0, x1, y = left + (index + .15) * slot, left + (index + .85) * slot, bottom
        for state in ("all_fail", "mixed", "all_success"):
            height = counts.get(state, 0) / total * (bottom - top)
            canvas.rect((x0, y - height, x1, y), plot.COLORS[state])
            y -= height
        canvas.text((x0 + x1) / 2, bottom + 24, str(run["training_step"]), 11, anchor="ma")
    for index, state in enumerate(("all_fail", "mixed", "all_success")):
        x = left + index * 180
        canvas.rect((x, 560, x + 18, 578), plot.COLORS[state])
        canvas.text(x + 25, 561, state.replace("_", " "), 13)
    canvas.text((left + right) / 2, bottom + 50, "training step", 13, anchor="ma")
    canvas.save(output, "05_validation_group_composition")


def points(runs: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    result = []
    for run in runs:
        value = finite(run.get(key))
        step = finite(run.get("training_step"))
        if value is not None and step is not None:
            result.append((step, value))
    return result


def metric_points(metrics: dict[str, dict[str, Any]], key: str) -> list[tuple[float, float]]:
    result = []
    for raw_step, row in metrics.items():
        value = finite(row.get(key))
        if value is not None:
            result.append((float(raw_step), value))
    return result


def system_points(path: Path, key: str, gpu: str | None = None) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return []
    start = float(rows[0]["unix"])
    result = []
    for row in rows:
        if gpu is not None and row["gpu"] != gpu:
            continue
        value = finite(row.get(key))
        if value is not None:
            result.append(((float(row["unix"]) - start) / 3600, value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    validation = report.get("validation_runs") or []
    training = report.get("training_runs") or []
    metrics = report.get("trainer_metrics") or {}
    output = args.output_dir or args.analysis.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot = load_plot_module()

    line_figure(plot, output, "01_validation_learning_curves", "Validation outcome learning curves", [
        ("ASR", [("terminal ASR", points(validation, "terminal_asr")), ("paper ASR", points(validation, "paper_asr")), ("pass@8", points(validation, "terminal_pass_at_g"))]),
        ("mixed group rate", [("mixed terminal", points(validation, "mixed_terminal_asr_group_rate"))]),
    ])
    line_figure(plot, output, "02_training_signal", "Training reward and group signal", [
        ("outcome", [("terminal ASR", points(training, "terminal_asr")), ("paper ASR", points(training, "paper_asr")), ("mixed", points(training, "mixed_terminal_asr_group_rate"))]),
        ("reward", [("console reward", metric_points(metrics, "critic/score/mean"))]),
    ])
    line_figure(plot, output, "03_optimization_health", "Optimization health (no reference KL loss)", [
        ("entropy", [("actor entropy", metric_points(metrics, "actor/entropy"))]),
        ("clip fraction", [("upper", metric_points(metrics, "actor/pg_clipfrac")), ("lower", metric_points(metrics, "actor/pg_clipfrac_lower"))]),
        ("diagnostics", [("PPO KL", metric_points(metrics, "actor/ppo_kl")), ("grad norm", metric_points(metrics, "actor/grad_norm"))]),
    ])
    line_figure(plot, output, "04_response_health", "Response length and truncation", [
        ("response tokens", [("train mean", points(training, "response_tokens_mean")), ("val mean", points(validation, "response_tokens_mean")), ("val P95", points(validation, "response_tokens_p95"))]),
        ("truncation", [("train", points(training, "token_limit_noncompletion_rate")), ("validation", points(validation, "token_limit_noncompletion_rate"))]),
    ])
    group_composition(plot, output, validation)
    line_figure(plot, output, "06_timing_and_throughput", "Training timing and throughput", [
        ("seconds", [("generation", metric_points(metrics, "timing_s/gen")), ("actor update", metric_points(metrics, "timing_s/update_actor")), ("reward", metric_points(metrics, "timing_s/reward"))]),
        ("tokens/s/GPU", [("throughput", metric_points(metrics, "perf/throughput"))]),
    ])
    system = Path(report["attempt_dir"]) / "system_metrics.csv"
    line_figure(plot, output, "07_gpu_and_disk", "GPU utilization and disk supervision", [
        ("GPU utilization %", [("GPU0", system_points(system, "util_gpu_pct", "0")), ("GPU1", system_points(system, "util_gpu_pct", "1"))]),
        ("GPU memory MiB", [("GPU0", system_points(system, "memory_used_mib", "0")), ("GPU1", system_points(system, "memory_used_mib", "1"))]),
        ("disk free GiB", [("free", system_points(system, "disk_free_gib", "0")), ("checkpoint", system_points(system, "checkpoint_gib", "0"))]),
    ], xlabel="elapsed hours")
    print(f"wrote PNG+SVG figures to {output}")


if __name__ == "__main__":
    main()
