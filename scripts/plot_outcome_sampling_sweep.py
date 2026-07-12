#!/usr/bin/env python3
"""Render PNG+SVG outcome sweep figures using only Pillow and the stdlib."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


INK = "#243447"
GRID = "#d9e1e8"
BLUE = "#2878b5"
GREY = "#9aa5af"
RED = "#d95f5f"
COLORS = {"all_fail": RED, "mixed": "#e6ab3c", "all_success": "#45a675"}


def num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def font_path() -> str | None:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return path
    return None


class Canvas:
    def __init__(self, width: int, height: int, title: str):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font_file = font_path()
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<title>{html.escape(title)}</title>',
        ]

    def font(self, size: int) -> ImageFont.ImageFont:
        return ImageFont.truetype(self.font_file, size) if self.font_file else ImageFont.load_default()

    def text(self, x: float, y: float, value: Any, size: int = 14, color: str = INK, anchor: str = "la") -> None:
        value = str(value)
        pil_anchor = {"la": "la", "ma": "ma", "ra": "ra", "mm": "mm"}.get(anchor, "la")
        self.draw.text((x, y), value, fill=color, font=self.font(size), anchor=pil_anchor)
        svg_anchor = {"la": "start", "ma": "middle", "ra": "end", "mm": "middle"}.get(anchor, "start")
        baseline = "middle" if anchor == "mm" else "auto"
        self.svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" font-size="{size}" '
            f'fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="{baseline}">{html.escape(value)}</text>'
        )

    def line(self, points: list[tuple[float, float]], color: str = INK, width: int = 2) -> None:
        self.draw.line(points, fill=color, width=width, joint="curve")
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, box: tuple[float, float, float, float], fill: str, outline: str | None = None, width: int = 1) -> None:
        self.draw.rectangle(box, fill=fill, outline=outline, width=width)
        x0, y0, x1, y1 = box
        stroke = outline or "none"
        self.svg.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1-x0:.1f}" height="{y1-y0:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def circle(self, x: float, y: float, radius: float, fill: str, outline: str | None = None) -> None:
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=outline)
        self.svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{outline or "none"}"/>')

    def save(self, output_dir: Path, name: str) -> None:
        self.image.save(output_dir / f"{name}.png")
        (output_dir / f"{name}.svg").write_text("\n".join([*self.svg, "</svg>"]) + "\n", encoding="utf-8")


def title(canvas: Canvas, value: str) -> None:
    canvas.text(canvas.width / 2, 35, value, 23, anchor="ma")


def label(run: dict[str, Any], seed: bool = False) -> str:
    text = f"T{run.get('temperature', '?')}/p{run.get('top_p', '?')}"
    return text + (f"/s{run.get('seed', '?')}" if seed else "")


def palette(value: float) -> str:
    stops = ((68, 1, 84), (49, 104, 142), (53, 183, 121), (253, 231, 37))
    value = max(0.0, min(1.0, value)) * (len(stops) - 1)
    index = min(int(value), len(stops) - 2)
    fraction = value - index
    rgb = tuple(round(stops[index][i] * (1-fraction) + stops[index+1][i] * fraction) for i in range(3))
    return "#%02x%02x%02x" % rgb


def grouped_grid(runs: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float], dict[tuple[float, float], float]]:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for run in runs:
        temperature, top_p, value = num(run.get("temperature")), num(run.get("top_p")), num(run.get(metric))
        if None not in (temperature, top_p, value):
            grouped[(temperature, top_p)].append(value)
    values = {key: sum(items) / len(items) for key, items in grouped.items()}
    return sorted({key[0] for key in values}), sorted({key[1] for key in values}), values


def heatmap_panel(canvas: Canvas, runs: list[dict[str, Any]], metric: str, heading: str, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    temperatures, top_ps, values = grouped_grid(runs, metric)
    canvas.text((x0+x1)/2, y0, heading, 18, anchor="ma")
    if not values:
        canvas.text((x0+x1)/2, (y0+y1)/2, "No compatible data", 15, GREY, "mm")
        return
    grid_top, grid_bottom = y0 + 35, y1 - 42
    grid_left, grid_right = x0 + 65, x1 - 10
    cell_w = (grid_right-grid_left) / len(top_ps)
    cell_h = (grid_bottom-grid_top) / len(temperatures)
    for row, temperature in enumerate(reversed(temperatures)):
        for column, top_p in enumerate(top_ps):
            xa, ya = grid_left + column*cell_w, grid_top + row*cell_h
            value = values.get((temperature, top_p))
            color = palette(value) if value is not None else "#eeeeee"
            canvas.rect((xa, ya, xa+cell_w, ya+cell_h), color, "white")
            canvas.text(xa+cell_w/2, ya+cell_h/2, "NA" if value is None else f"{value:.3f}", 13, "white" if value is not None and value < .65 else INK, "mm")
        canvas.text(grid_left-8, grid_top+(row+.5)*cell_h, temperature, 13, anchor="ra")
    for column, top_p in enumerate(top_ps):
        canvas.text(grid_left+(column+.5)*cell_w, grid_bottom+18, top_p, 13, anchor="ma")
    canvas.text((grid_left+grid_right)/2, grid_bottom+38, "top-p", 13, anchor="ma")
    canvas.text(x0+8, grid_top-10, "temperature", 11, GREY)


def plot_heatmaps(runs: list[dict[str, Any]], output: Path) -> None:
    canvas = Canvas(760, 600, "Mixed terminal-ASR group rate")
    heatmap_panel(canvas, runs, "mixed_terminal_asr_group_rate", "Mixed terminal-ASR group rate", (20, 30, 740, 570))
    canvas.save(output, "01_mixed_terminal_asr_heatmap")
    canvas = Canvas(1740, 570, "Outcome heatmaps")
    for index, (metric, heading) in enumerate((("paper_asr", "Paper ASR"), ("terminal_asr", "Terminal ASR"), ("terminal_pass_at_g", "Terminal pass@G"))):
        heatmap_panel(canvas, runs, metric, heading, (20+index*570, 30, 570+index*570, 540))
    canvas.save(output, "02_outcome_heatmaps")


def plot_groups(runs: list[dict[str, Any]], output: Path) -> None:
    usable = [run for run in runs if sum((run.get("terminal_group_counts") or {}).values())]
    usable.sort(key=lambda run: (num(run.get("temperature")) or 0, num(run.get("top_p")) or 0, num(run.get("seed")) or 0))
    width = max(900, 160 + 92*len(usable))
    canvas = Canvas(width, 620, "Terminal-ASR group composition")
    title(canvas, "Terminal-ASR group composition")
    if not usable:
        canvas.text(width/2, 300, "No compatible group data", 17, GREY, "mm")
    else:
        left, top, bottom = 85, 85, 485
        plot_w = width-left-30
        for tick in range(6):
            y = bottom - tick*(bottom-top)/5
            canvas.line([(left, y), (width-30, y)], GRID, 1)
            canvas.text(left-10, y, f"{tick/5:.1f}", 12, anchor="ra")
        slot = plot_w/len(usable)
        show_seed = len({run.get("seed") for run in usable}) > 1
        for index, run in enumerate(usable):
            counts = run["terminal_group_counts"]
            total = sum(counts.values())
            x0, x1, y = left+index*slot+slot*.15, left+(index+1)*slot-slot*.15, bottom
            for state in ("all_fail", "mixed", "all_success"):
                fraction = counts.get(state, 0)/total
                height = fraction*(bottom-top)
                canvas.rect((x0, y-height, x1, y), COLORS[state])
                y -= height
            canvas.text((x0+x1)/2, bottom+24, label(run, show_seed), 10, anchor="ma")
        for index, state in enumerate(("all_fail", "mixed", "all_success")):
            x = left + index*165
            canvas.rect((x, 550, x+18, 568), COLORS[state])
            canvas.text(x+25, 551, state.replace("_", " "), 13)
    canvas.save(output, "03_terminal_group_composition")


def axes(canvas: Canvas, box: tuple[int, int, int, int], heading: str, xlabel: str, ylabel: str, x_range: tuple[float, float], y_range: tuple[float, float]) -> callable:
    x0, y0, x1, y1 = box
    canvas.text((x0+x1)/2, y0-28, heading, 18, anchor="ma")
    for tick in range(6):
        x = x0 + tick*(x1-x0)/5
        y = y1 - tick*(y1-y0)/5
        canvas.line([(x, y0), (x, y1)], GRID, 1)
        canvas.line([(x0, y), (x1, y)], GRID, 1)
        canvas.text(x, y1+18, f"{x_range[0]+tick*(x_range[1]-x_range[0])/5:.2g}", 11, anchor="ma")
        canvas.text(x0-8, y, f"{y_range[0]+tick*(y_range[1]-y_range[0])/5:.2g}", 11, anchor="ra")
    canvas.line([(x0, y0), (x0, y1), (x1, y1)], INK, 2)
    canvas.text((x0+x1)/2, y1+43, xlabel, 13, anchor="ma")
    canvas.text(x0, y0-8, ylabel, 12)
    def transform(x: float, y: float) -> tuple[float, float]:
        px = x0 + (x-x_range[0])/(x_range[1]-x_range[0])*(x1-x0)
        py = y1 - (y-y_range[0])/(y_range[1]-y_range[0])*(y1-y0)
        return px, py
    return transform


def plot_pareto(runs: list[dict[str, Any]], output: Path) -> None:
    canvas = Canvas(1420, 590, "Exploration Pareto")
    tokens = [value for run in runs if (value := num(run.get("response_tokens_mean"))) is not None]
    token_range = (0, max(tokens)*1.1) if tokens else (0, 1)
    panels = [
        ((80, 80, 680, 500), "Exploration vs outcome", "terminal ASR", (0, 1), "terminal_asr"),
        ((800, 80, 1400, 500), "Exploration vs token cost", "mean response tokens", token_range, "response_tokens_mean"),
    ]
    for box, heading, xlabel, x_range, metric in panels:
        transform = axes(canvas, box, heading, xlabel, "mixed group rate", x_range, (0, 1))
        for run in runs:
            x, y = num(run.get(metric)), num(run.get("mixed_terminal_asr_group_rate"))
            if x is None or y is None:
                continue
            px, py = transform(x, y)
            canvas.circle(px, py, 5, BLUE if run.get("gate_eligible") else GREY)
            canvas.text(px+7, py-4, label(run), 9)
    canvas.save(output, "04_exploration_pareto")


def plot_ci(runs: list[dict[str, Any]], output: Path, *, aggregated: bool = False) -> None:
    usable = [run for run in runs if num(run.get("mixed_terminal_asr_group_rate")) is not None]
    if aggregated and any(run.get("rank") is not None for run in usable):
        usable.sort(key=lambda run: int(run.get("rank") or 10**9))
    else:
        usable.sort(key=lambda run: num(run["mixed_terminal_asr_group_rate"]) or -1)
    height = max(500, 130 + len(usable)*34)
    canvas = Canvas(1100, height, "Mixed-group bootstrap confidence intervals")
    title(canvas, "Configuration-level bootstrap 95% confidence intervals" if aggregated else "Query bootstrap 95% confidence intervals")
    left, right, top, bottom = 260, 1060, 80, height-55
    for tick in range(6):
        x = left+tick*(right-left)/5
        canvas.line([(x, top), (x, bottom)], GRID, 1)
        canvas.text(x, bottom+20, f"{tick/5:.1f}", 12, anchor="ma")
    show_seed = len({run.get("seed") for run in usable}) > 1
    for index, run in enumerate(usable):
        y = top+(index+.5)*(bottom-top)/len(usable)
        center = num(run["mixed_terminal_asr_group_rate"])
        ci = run.get("mixed_terminal_asr_ci95") or [center, center]
        low, high = num(ci[0]), num(ci[1])
        low = center if low is None else low
        high = center if high is None else high
        scale = lambda value: left+value*(right-left)
        eligible = run.get("gate_eligible", run.get("all_runs_eligible", True))
        color = BLUE if eligible else GREY
        canvas.line([(scale(low), y), (scale(high), y)], color, 3)
        canvas.line([(scale(low), y-5), (scale(low), y+5)], color, 2)
        canvas.line([(scale(high), y-5), (scale(high), y+5)], color, 2)
        canvas.circle(scale(center), y, 5, color)
        canvas.text(left-12, y, label(run, show_seed), 11, anchor="ra")
    canvas.save(output, "05_mixed_group_bootstrap_ci")


def plot_lengths(runs: list[dict[str, Any]], output: Path) -> None:
    usable = [run for run in runs if num(run.get("response_tokens_p50")) is not None]
    usable.sort(key=lambda run: (num(run.get("temperature")) or 0, num(run.get("top_p")) or 0, num(run.get("seed")) or 0))
    canvas = Canvas(max(1300, 220+len(usable)*100), 620, "Length and truncation")
    title(canvas, "Response length and token-limit truncation")
    if not usable:
        canvas.text(canvas.width/2, 300, "Token counts unavailable in these reports", 17, GREY, "mm")
    else:
        show_seed = len({run.get("seed") for run in usable}) > 1
        max_token = max(num(run.get("response_tokens_max")) or 0 for run in usable) or 1
        left, top, bottom, middle = 85, 85, 490, int(canvas.width*.63)
        slot = (middle-left-25)/len(usable)
        for index, run in enumerate(usable):
            for offset, (key, color) in enumerate((("response_tokens_p50", BLUE), ("response_tokens_p95", "#e6ab3c"), ("response_tokens_max", RED))):
                value = num(run.get(key)) or 0
                bar_w = slot*.2
                x0 = left+index*slot+slot*.15+offset*bar_w
                canvas.rect((x0, bottom-value/max_token*(bottom-top), x0+bar_w, bottom), color)
            canvas.text(left+(index+.5)*slot, bottom+22, label(run, show_seed), 9, anchor="ma")
        canvas.text(left, 65, "P50 (blue), P95 (amber), max (red)", 12)
        transform = axes(canvas, (middle+90, top, canvas.width-35, bottom), "Truncation rate", "configuration index", "fraction", (0, max(1, len(usable)-1)), (0, max(.01, max(((run.get("failure_counts") or {}).get("truncation", 0)/(run.get("rows") or 1) for run in usable), default=0)*1.2)))
        for index, run in enumerate(usable):
            rate = (run.get("failure_counts") or {}).get("truncation", 0)/(run.get("rows") or 1)
            canvas.circle(*transform(index, rate), 5, RED)
    canvas.save(output, "06_length_and_truncation")


def plot_theory(runs: list[dict[str, Any]], output: Path) -> None:
    canvas = Canvas(900, 650, "Theory vs observed mixed groups")
    transform = axes(canvas, (90, 85, 850, 560), "Observed vs pooled-p Bernoulli heuristic", "pooled trajectory terminal ASR (p)", "mixed group probability", (0, 1), (0, 1))
    sizes = sorted({int(run.get("group_size") or 8) for run in runs}) or [8]
    curve_colors = (BLUE, "#45a675", "#8f63b8")
    for index, group_size in enumerate(sizes):
        points = []
        for step in range(201):
            p = step/200
            points.append(transform(p, 1-p**group_size-(1-p)**group_size))
        canvas.line(points, curve_colors[index % len(curve_colors)], 3)
        canvas.text(650, 95+index*20, f"theory G={group_size}", 12, curve_colors[index % len(curve_colors)])
    for run in runs:
        p, mixed = num(run.get("terminal_asr")), num(run.get("mixed_terminal_asr_group_rate"))
        if p is not None and mixed is not None:
            x, y = transform(p, mixed)
            canvas.circle(x, y, 5, RED)
            canvas.text(x+7, y-4, label(run), 9)
    canvas.save(output, "07_theory_vs_observed_mixed_rate")


def plot_reward_flow(output: Path) -> None:
    """Make the reward boundary explicit for reports and interview discussion."""
    canvas = Canvas(1500, 430, "Paper ASR to GRPO advantage")
    title(canvas, "Outcome-only reward path (G=8)")
    nodes = (
        (35, 125, 320, 275, "Official evaluator", "rule == 1 AND budget == 1", "paper_asr ∈ {0,1}"),
        (405, 125, 690, 275, "Terminate gate", "terminate(status=success)", "terminate_success ∈ {0,1}"),
        (775, 125, 1060, 275, "Training outcome", "paper_asr × terminate_success", "terminal_asr = score"),
        (1145, 125, 1430, 275, "Within-query GRPO", "standardize 8 binary outcomes", "mixed group → ± advantage"),
    )
    fills = ("#e8f1f8", "#f9edcf", "#e2f2e9", "#eee8f7")
    for (x0, y0, x1, y1, heading, detail, result), fill in zip(nodes, fills, strict=True):
        canvas.rect((x0, y0, x1, y1), fill, INK, 2)
        canvas.text((x0+x1)/2, y0+35, heading, 18, anchor="ma")
        canvas.text((x0+x1)/2, y0+78, detail, 13, anchor="ma")
        canvas.text((x0+x1)/2, y0+115, result, 14, BLUE, "ma")
    for left, right in zip(nodes, nodes[1:]):
        x0, x1, y = left[2], right[0], 200
        canvas.line([(x0+12, y), (x1-25, y)], INK, 3)
        canvas.text(x1-14, y, "▶", 20, INK, "mm")
    canvas.text(750, 330, "All-fail / all-success group: advantage = 0    |    Mixed group: positive and negative learning signal", 16, anchor="ma")
    canvas.text(750, 370, "format, workflow, tokens and errors are diagnostics/safety gates only; they never enter score", 14, RED, "ma")
    canvas.save(output, "09_outcome_reward_flow")


def plot_pilots(runs: list[dict[str, Any]], output: Path) -> bool:
    pilot_step = lambda run: num(run.get("training_step")) if num(run.get("training_step")) is not None else num(run.get("step"))
    pilots = [run for run in runs if pilot_step(run) is not None]
    if not pilots:
        return False
    canvas = Canvas(1500, 900, "GRPO pilot curves")
    metrics = (("terminal_asr", "Terminal ASR"), ("mixed_terminal_asr_group_rate", "Mixed rate"), ("kl", "KL"), ("entropy", "Entropy"), ("clip_fraction", "Clip fraction"))
    aliases = {
        "terminal_asr": ("terminal_asr", "validation_terminal_asr", "val_terminal_asr"),
        "mixed_terminal_asr_group_rate": ("mixed_terminal_asr_group_rate", "mixed_group_rate"),
        "kl": ("kl", "approx_kl", "policy_kl"),
        "entropy": ("entropy", "policy_entropy"),
        "clip_fraction": ("clip_fraction", "clipfrac", "clip_ratio"),
    }
    metric_value = lambda run, metric: next((value for key in aliases[metric] if (value := num(run.get(key))) is not None), None)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in pilots:
        grouped[str(run.get("candidate") or label(run))].append(run)
    colors = (BLUE, RED, "#45a675", "#8f63b8")
    for index, (metric, heading) in enumerate(metrics):
        row, column = divmod(index, 3)
        box = (70+column*490, 90+row*400, 460+column*490, 410+row*400)
        values = [metric_value(run, metric) for run in pilots]
        values = [value for value in values if value is not None]
        steps = [pilot_step(run) for run in pilots]
        steps = [value for value in steps if value is not None]
        if not values:
            canvas.text((box[0]+box[2])/2, (box[1]+box[3])/2, f"{heading}: no data", 14, GREY, "mm")
            continue
        yrange = (min(0, min(values)), max(values)*1.1 if max(values) else 1)
        step_min, step_max = min(steps), max(steps)
        if step_min == step_max:
            step_max = step_min + 1
        transform = axes(canvas, box, heading, "GRPO step", metric, (step_min, step_max), yrange)
        for color, (candidate, items) in zip(colors, grouped.items()):
            points = sorted((pilot_step(item), metric_value(item, metric)) for item in items)
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if points:
                pixels = [transform(x, y) for x, y in points]
                canvas.line(pixels, color, 3)
                for x, y in pixels:
                    canvas.circle(x, y, 4, color)
                canvas.text(box[2]-5, box[1]+18*list(grouped).index(candidate), candidate, 10, color, "ra")
    canvas.save(output, "08_grpo_pilot_curves")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", help="JSON produced by analyze_outcome_sampling_sweep.py")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    source = Path(args.analysis)
    report = json.loads(source.read_text(encoding="utf-8"))
    runs = report.get("runs") or []
    if not isinstance(runs, list):
        raise ValueError("analysis JSON field 'runs' must be a list")
    output = Path(args.output_dir) if args.output_dir else source.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(runs, output)
    plot_groups(runs, output)
    plot_pareto(runs, output)
    config_ranking = report.get("config_ranking") or []
    plot_ci(config_ranking or runs, output, aggregated=bool(config_ranking))
    plot_lengths(runs, output)
    plot_theory(runs, output)
    plot_reward_flow(output)
    pilot_runs = report.get("pilot_runs") or []
    pilots = plot_pilots([*runs, *pilot_runs], output)
    print(f"wrote figures to {output} ({'including' if pilots else 'without'} pilot curves)")


if __name__ == "__main__":
    main()
