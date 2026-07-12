#!/usr/bin/env python3
"""Plot the untouched Step108 versus selected formal-GRPO test75 comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def plot_module():
    path = ROOT / "scripts/plot_outcome_sampling_sweep.py"
    spec = importlib.util.spec_from_file_location("outcome_plot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(canvas, plot, runs, title, keys, labels, box, percentage=True):
    x0, y0, x1, y1 = box
    canvas.text((x0 + x1) / 2, y0 + 20, title, 18, anchor="ma")
    top, bottom, left, right = y0 + 55, y1 - 65, x0 + 65, x1 - 20
    maximum = 1.0 if percentage else max(float(run.get(key) or 0) for run in runs for key in keys) * 1.15
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        canvas.line([(left, y), (right, y)], plot.GRID, 1)
        canvas.text(left - 8, y, f"{value:.0%}" if percentage else f"{value:.0f}", 11, anchor="ra")
    group_w = (right - left) / len(runs)
    bar_w = group_w * 0.72 / len(keys)
    colors = (plot.BLUE, plot.RED, "#45a675")
    for run_index, run in enumerate(runs):
        center = left + (run_index + 0.5) * group_w
        for metric_index, key in enumerate(keys):
            value = float(run.get(key) or 0)
            xa = center - len(keys) * bar_w / 2 + metric_index * bar_w
            xb = xa + bar_w * 0.88
            ya = bottom - (bottom - top) * value / maximum
            canvas.rect((xa, ya, xb, bottom), colors[metric_index])
            canvas.text((xa + xb) / 2, ya - 6, f"{value:.1%}" if percentage else f"{value:.0f}", 10, anchor="ma")
        canvas.text(center, bottom + 22, "Untouched Step108" if run_index == 0 else "Best GRPO", 12, anchor="ma")
    legend_x = left
    for index, label in enumerate(labels):
        canvas.rect((legend_x, y1 - 28, legend_x + 13, y1 - 15), colors[index])
        canvas.text(legend_x + 19, y1 - 21, label, 11, anchor="la")
        legend_x += 125


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.analysis.read_text()).get("runs", [])
    if len(runs) != 2:
        raise SystemExit(f"expected exactly two test runs, found {len(runs)}")
    plot = plot_module()
    canvas = plot.Canvas(1400, 1030, "Product-disjoint Test75: Step108 vs selected outcome-only GRPO")
    plot.title(canvas, "Product-disjoint Test75: Step108 vs selected outcome-only GRPO")
    panels = [
        ("Outcome ASR", ["paper_asr", "terminal_asr"], ["Paper ASR", "Terminal ASR"], True),
        ("Group-level success", ["terminal_pass_at_g", "mixed_terminal_asr_group_rate"], ["Pass@8", "Mixed groups"], True),
        ("Response health", ["format_mean", "workflow_valid_mean", "token_limit_noncompletion_rate"], ["Format", "Workflow", "Truncation"], True),
        ("Response token cost", ["response_tokens_mean", "response_tokens_p95"], ["Mean", "P95"], False),
    ]
    boxes = ((20, 60, 690, 530), (710, 60, 1380, 530), (20, 545, 690, 1015), (710, 545, 1380, 1015))
    for values, box in zip(panels, boxes):
        panel(canvas, plot, runs, values[0], values[1], values[2], box, values[3])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.image.save(args.output)
    args.output.with_suffix(".svg").write_text("\n".join([*canvas.svg, "</svg>"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
