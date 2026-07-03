#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


METRIC_PLOTS = [
    (
        "quality_metrics",
        [
            ("protocol_mean", "protocol"),
            ("workflow_valid_mean", "workflow"),
            ("format_mean", "format"),
            ("tool_valid_mean", "tool"),
            ("progress_mean", "progress"),
            ("final_success_mean", "final"),
        ],
        1.0,
    ),
    (
        "progress_components",
        [
            ("find_correct_mean", "find"),
            ("view_confirmed_mean", "view"),
            ("budget_correct_mean", "budget"),
            ("recommend_correct_mean", "recommend"),
            ("terminate_complete_mean", "terminate"),
        ],
        1.0,
    ),
    (
        "grpo_readiness",
        [
            ("per_prompt_valid_count_mean", "valid/prompt"),
            ("task_group_var_mean", "task var"),
            ("reward_group_var_mean", "reward var"),
            ("recommended_unique_mean", "unique rec"),
        ],
        None,
    ),
]


def load_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("reports") or [payload]
    rows = []
    for report in reports:
        agg = report.get("aggregate", {})
        checkpoint = str(report.get("checkpoint") or Path(str(report.get("path", ""))).name)
        row = {"checkpoint": checkpoint}
        row.update(agg)
        rows.append(row)
    return sorted(rows, key=lambda item: int(str(item["checkpoint"]).rsplit("_", 1)[-1]))


def font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def draw_grouped_bars(rows: list[dict[str, Any]], metrics: list[tuple[str, str]], y_max: float | None, title: str, output: Path):
    width, height = 1500, 850
    left, right, top, bottom = 105, 35, 95, 135
    plot_w, plot_h = width - left - right, height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(26)
    label_font = font(18)
    tiny = font(14)
    palette = [
        (38, 114, 182),
        (42, 157, 143),
        (233, 196, 106),
        (230, 111, 81),
        (118, 80, 160),
        (90, 101, 117),
    ]
    values = [as_float(row.get(metric)) for row in rows for metric, _ in metrics]
    max_value = y_max if y_max is not None else max(values + [1.0]) * 1.12
    max_value = max(max_value, 1e-6)

    def ymap(value: float) -> float:
        return top + plot_h - (value / max_value) * plot_h

    for i in range(6):
        y = top + i * plot_h / 5
        value = max_value - i * max_value / 5
        draw.line((left, y, left + plot_w, y), fill=(228, 232, 236), width=1)
        draw.text((22, y - 9), f"{value:.2f}", fill=(70, 78, 86), font=tiny)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(125, 132, 140), width=2)
    draw.text((left, 35), title, fill=(24, 30, 38), font=title_font)

    groups = len(rows)
    group_w = plot_w / max(groups, 1)
    bar_w = min(28, group_w / (len(metrics) + 1.7))
    for gi, row in enumerate(rows):
        center = left + gi * group_w + group_w / 2
        group_start = center - (len(metrics) * bar_w + (len(metrics) - 1) * 6) / 2
        for mi, (metric, _) in enumerate(metrics):
            value = as_float(row.get(metric))
            x0 = group_start + mi * (bar_w + 6)
            x1 = x0 + bar_w
            y0 = ymap(value)
            draw.rectangle((x0, y0, x1, top + plot_h), fill=palette[mi % len(palette)])
            draw.text((x0 - 5, y0 - 18), f"{value:.2f}", fill=(40, 48, 56), font=tiny)
        checkpoint = str(row["checkpoint"]).replace("global_step_", "")
        draw.text((center - 28, top + plot_h + 22), checkpoint, fill=(45, 52, 60), font=label_font)
    draw.text((left + plot_w // 2 - 45, height - 42), "checkpoint step", fill=(45, 52, 60), font=label_font)

    legend_x, legend_y = left, height - 88
    for mi, (_, label) in enumerate(metrics):
        x = legend_x + mi * 190
        draw.rectangle((x, legend_y, x + 24, legend_y + 16), fill=palette[mi % len(palette)])
        draw.text((x + 32, legend_y - 2), label, fill=(40, 48, 56), font=tiny)

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def write_csv(rows: list[dict[str, Any]], output: Path):
    keys = [
        "checkpoint",
        "rows",
        "queries",
        "protocol_mean",
        "workflow_valid_mean",
        "format_mean",
        "tool_valid_mean",
        "progress_mean",
        "find_correct_mean",
        "view_confirmed_mean",
        "budget_correct_mean",
        "recommend_correct_mean",
        "terminate_complete_mean",
        "final_success_mean",
        "task_mean",
        "task_group_var_mean",
        "reward_group_var_mean",
        "per_prompt_valid_count_mean",
        "rows_without_tool_block",
        "output_len_max",
        "recommended_unique_mean",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Stage-1 SFT checkpoint probe summaries.")
    parser.add_argument("summary_json")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = Path(args.summary_json)
    rows = load_summary(summary)
    output_dir = Path(args.output_dir) if args.output_dir else summary.parent / "figures"
    csv_path = output_dir / "stage1_metrics.csv"
    write_csv(rows, csv_path)
    outputs = [csv_path]
    for name, metrics, y_max in METRIC_PLOTS:
        output = output_dir / f"{name}.png"
        draw_grouped_bars(rows, metrics, y_max, name.replace("_", " ").title(), output)
        outputs.append(output)
    print(json.dumps({"rows": len(rows), "outputs": [str(item) for item in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
