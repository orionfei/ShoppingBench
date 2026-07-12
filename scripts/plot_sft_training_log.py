#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


METRIC_RE = re.compile(r"\[step\s+(\d+)\]\s+(.+)")
PAIR_RE = re.compile(r"([A-Za-z0-9_/\-\(\)]+)=([-+0-9.eE]+)")


def parse_log(path: Path) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    with path.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            match = METRIC_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            for name, value in PAIR_RE.findall(match.group(2)):
                series.setdefault(name, []).append({"step": step, "value": float(value)})
    return series


def plot_with_matplotlib(series: dict[str, list[dict[str, float]]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    loss_ax, lr_ax = axes

    for name in ("train/loss", "val/loss"):
        points = series.get(name, [])
        if points:
            loss_ax.plot([p["step"] for p in points], [p["value"] for p in points], marker="o", label=name)
    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    for name in ("train/lr(1e-3)", "train/lr"):
        points = series.get(name, [])
        if points:
            lr_ax.plot([p["step"] for p in points], [p["value"] for p in points], marker="o", label=name)
    lr_ax.set_xlabel("step")
    lr_ax.set_ylabel("lr")
    lr_ax.grid(True, alpha=0.3)
    lr_ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def plot_with_pil(series: dict[str, list[dict[str, float]]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 820
    margin_left, margin_right = 92, 42
    margin_top, margin_bottom = 58, 84
    gap = 64
    panel_h = (height - margin_top - margin_bottom - gap) // 2
    panels = [
        (margin_top, margin_top + panel_h, "loss", [("train/loss", (34, 99, 190)), ("val/loss", (204, 80, 62))]),
        (
            margin_top + panel_h + gap,
            margin_top + panel_h + gap + panel_h,
            "lr",
            [("train/lr(1e-3)", (54, 138, 88)), ("train/lr", (54, 138, 88))],
        ),
    ]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    def clean_range(values: list[float]) -> tuple[float, float]:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return 0.0, 1.0
        lo, hi = min(finite), max(finite)
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1e-6)
            return lo - pad, hi + pad
        pad = (hi - lo) * 0.08
        return lo - pad, hi + pad

    def transform(step: int, value: float, min_step: int, max_step: int, min_val: float, max_val: float, top: int, bottom: int):
        left = margin_left
        right = width - margin_right
        x_ratio = 0.0 if max_step == min_step else (step - min_step) / (max_step - min_step)
        y_ratio = 0.0 if max_val == min_val else (value - min_val) / (max_val - min_val)
        x = left + x_ratio * (right - left)
        y = bottom - y_ratio * (bottom - top)
        return int(round(x)), int(round(y))

    draw.text((margin_left, 22), "SFT training metrics", fill=(20, 20, 20), font=title_font)
    for top, bottom, label, metric_specs in panels:
        metric_points = [(name, series.get(name, []), color) for name, color in metric_specs if series.get(name)]
        draw.rectangle((margin_left, top, width - margin_right, bottom), outline=(210, 210, 210), width=1)
        draw.text((24, top + 8), label, fill=(40, 40, 40), font=font)
        if not metric_points:
            draw.text((margin_left + 12, top + 12), "no data", fill=(120, 120, 120), font=font)
            continue

        steps = [int(p["step"]) for _, points, _ in metric_points for p in points]
        values = [float(p["value"]) for _, points, _ in metric_points for p in points]
        min_step, max_step = min(steps), max(steps)
        min_val, max_val = clean_range(values)

        for i in range(5):
            y = top + round(i * (bottom - top) / 4)
            draw.line((margin_left, y, width - margin_right, y), fill=(238, 238, 238), width=1)
            value = max_val - i * (max_val - min_val) / 4
            draw.text((18, y - 6), f"{value:.3g}", fill=(90, 90, 90), font=font)
        for i in range(5):
            x = margin_left + round(i * (width - margin_right - margin_left) / 4)
            draw.line((x, top, x, bottom), fill=(246, 246, 246), width=1)
            step = min_step + i * (max_step - min_step) / 4
            draw.text((x - 10, bottom + 10), f"{step:.0f}", fill=(90, 90, 90), font=font)

        legend_x = margin_left + 8
        legend_y = top + 8
        for name, points, color in metric_points:
            coords = [
                transform(int(p["step"]), float(p["value"]), min_step, max_step, min_val, max_val, top + 12, bottom - 12)
                for p in points
            ]
            if len(coords) >= 2:
                draw.line(coords, fill=color, width=3)
            for x, y in coords:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 8), fill=color)
            draw.text((legend_x + 26, legend_y - 2), name, fill=(40, 40, 40), font=font)
            legend_x += 170

    image.save(output)


def plot(series: dict[str, list[dict[str, float]]], output: Path) -> None:
    try:
        plot_with_matplotlib(series, output)
    except ModuleNotFoundError:
        plot_with_pil(series, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot verl SFT console metrics from a training log.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    series = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(series, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if series:
        plot(series, args.output_dir / "loss_curve.png")
    print(json.dumps({"metrics": str(metrics_path), "plot": str(args.output_dir / "loss_curve.png")}, indent=2))


if __name__ == "__main__":
    main()
