#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_DIR = Path(__file__).resolve().parent
OUT_PNG = RUN_DIR / "checkpoint_probe_summary.png"
OUT_SVG = RUN_DIR / "checkpoint_probe_summary.svg"


COLORS = {
    "bg": "#f7f8fb",
    "panel": "#ffffff",
    "grid": "#d9dee8",
    "text": "#172033",
    "muted": "#687385",
    "best": "#fff3c4",
    "success": "#207a4c",
    "task": "#2563eb",
    "progress": "#c2410c",
    "protocol": "#7c3aed",
    "protocol_invalid": "#ef4444",
    "search_recall_gap": "#f59e0b",
    "workflow_invalid": "#8b5cf6",
    "final_selection_after_full_recall_gap": "#64748b",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rows() -> list[dict]:
    rows = []
    for path in RUN_DIR.glob("global_step_*.json"):
        data = json.loads(path.read_text())
        agg = data["aggregate"]
        step = int(re.search(r"global_step_(\d+)", data["checkpoint"]).group(1))
        rows.append(
            {
                "step": step,
                "checkpoint": data["checkpoint"],
                "success": float(agg["final_success_mean"]),
                "success_n": int(round(float(agg["final_success_mean"]) * int(agg["rows"]))),
                "rows": int(agg["rows"]),
                "task": float(agg["task_mean"]),
                "progress": float(agg["progress_mean"]),
                "protocol": float(agg["protocol_mean"]),
                "steps_mean": float(agg["steps_mean"]),
                "modes": dict(agg["structured_failure_modes"]),
            }
        )
    return sorted(rows, key=lambda x: x["step"])


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text(draw, xy, text, size=24, fill=None, bold=False, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=fill or COLORS["text"], anchor=anchor)


def line_chart(draw, rows, box):
    x0, y0, x1, y1 = box
    pad_l, pad_t, pad_r, pad_b = 82, 54, 30, 70
    cx0, cy0, cx1, cy1 = x0 + pad_l, y0 + pad_t, x1 - pad_r, y1 - pad_b
    rounded_rect(draw, box, 18, COLORS["panel"])
    draw_text(draw, (x0 + 28, y0 + 22), "Checkpoint metrics", 28, bold=True)
    draw_text(draw, (x1 - 28, y0 + 25), "8 test queries x G=4, temp=0.2", 18, COLORS["muted"], anchor="ra")

    for i in range(6):
        v = i / 5
        y = cy1 - v * (cy1 - cy0)
        draw.line((cx0, y, cx1, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (cx0 - 14, y), f"{v:.1f}", 16, COLORS["muted"], anchor="rm")

    steps = [r["step"] for r in rows]
    min_s, max_s = min(steps), max(steps)

    def px(step):
        return cx0 + (step - min_s) / (max_s - min_s) * (cx1 - cx0)

    def py(v):
        return cy1 - v * (cy1 - cy0)

    best_step = 108
    bx = px(best_step)
    draw.rectangle((bx - 42, cy0, bx + 42, cy1), fill=COLORS["best"])
    draw_text(draw, (bx, cy0 + 16), "best overall", 15, "#8a5b00", anchor="mm")

    series = [
        ("success", "Success"),
        ("task", "Task reward"),
        ("progress", "Progress"),
        ("protocol", "Protocol"),
    ]
    for key, label in series:
        pts = [(px(r["step"]), py(r[key])) for r in rows]
        draw.line(pts, fill=COLORS[key], width=4, joint="curve")
        for x, y in pts:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS[key], outline="white", width=2)
        lx = x0 + 360 + series.index((key, label)) * 230
        ly = y0 + 28
        draw.line((lx, ly + 8, lx + 32, ly + 8), fill=COLORS[key], width=5)
        draw_text(draw, (lx + 42, ly), label, 18, COLORS["muted"])

    for r in rows:
        x = px(r["step"])
        draw.line((x, cy1, x, cy1 + 8), fill=COLORS["muted"], width=1)
        draw_text(draw, (x, cy1 + 18), str(r["step"]), 18, COLORS["muted"], anchor="ma")
        draw_text(draw, (x, py(r["success"]) - 22), f"{r['success_n']}/{r['rows']}", 16, COLORS["success"], bold=True, anchor="mm")

    draw.line((cx0, cy0, cx0, cy1, cx1, cy1), fill=COLORS["muted"], width=2)
    draw_text(draw, ((cx0 + cx1) / 2, y1 - 28), "global step", 18, COLORS["muted"], anchor="mm")


def stacked_modes(draw, rows, box):
    x0, y0, x1, y1 = box
    rounded_rect(draw, box, 18, COLORS["panel"])
    draw_text(draw, (x0 + 28, y0 + 22), "Outcome distribution", 28, bold=True)
    draw_text(draw, (x0 + 28, y0 + 58), "Counts per checkpoint, total=32", 18, COLORS["muted"])

    modes = ["success", "protocol_invalid", "search_recall_gap", "workflow_invalid", "final_selection_after_full_recall_gap"]
    labels = {
        "success": "success",
        "protocol_invalid": "protocol_invalid",
        "search_recall_gap": "search_recall_gap",
        "workflow_invalid": "workflow_invalid",
        "final_selection_after_full_recall_gap": "final_selection_gap",
    }
    bx0, by0, bx1, by1 = x0 + 82, y0 + 106, x1 - 38, y1 - 58
    bar_w = 64
    gap = (bx1 - bx0 - len(rows) * bar_w) / (len(rows) - 1)
    max_total = max(r["rows"] for r in rows)

    for i in range(5):
        v = i * 8
        y = by1 - v / max_total * (by1 - by0)
        draw.line((bx0 - 12, y, bx1, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (bx0 - 20, y), str(v), 15, COLORS["muted"], anchor="rm")

    for idx, r in enumerate(rows):
        x = bx0 + idx * (bar_w + gap)
        bottom = by1
        for mode in modes:
            cnt = int(r["modes"].get(mode, 0))
            if cnt <= 0:
                continue
            h = cnt / max_total * (by1 - by0)
            draw.rectangle((x, bottom - h, x + bar_w, bottom), fill=COLORS.get(mode, "#94a3b8"))
            if h > 18:
                draw_text(draw, (x + bar_w / 2, bottom - h / 2), str(cnt), 14, "white", bold=True, anchor="mm")
            bottom -= h
        draw_text(draw, (x + bar_w / 2, by1 + 18), str(r["step"]), 18, COLORS["muted"], anchor="ma")

    lx, ly = x0 + 370, y0 + 25
    for mode in modes:
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=COLORS.get(mode, "#94a3b8"))
        draw_text(draw, (lx + 28, ly - 2), labels[mode], 16, COLORS["muted"])
        lx += 185 if mode != "final_selection_after_full_recall_gap" else 0
        if lx > x1 - 230:
            lx = x0 + 370
            ly += 28


def table_panel(draw, rows, box):
    x0, y0, x1, y1 = box
    rounded_rect(draw, box, 18, COLORS["panel"])
    draw_text(draw, (x0 + 28, y0 + 22), "Checkpoint table", 28, bold=True)

    headers = ["step", "success", "task", "progress", "protocol", "avg steps"]
    widths = [80, 165, 120, 140, 140, 130]
    sx, sy = x0 + 28, y0 + 82
    row_h = 44
    x = sx
    for h, w in zip(headers, widths):
        draw_text(draw, (x, sy), h, 16, COLORS["muted"], bold=True)
        x += w
    draw.line((sx, sy + 28, x1 - 28, sy + 28), fill=COLORS["grid"], width=1)

    best = max(rows, key=lambda r: (r["success"], r["task"], r["progress"], r["protocol"]))
    for i, r in enumerate(rows):
        y = sy + 42 + i * row_h
        if r is best:
            draw.rounded_rectangle((sx - 10, y - 9, x1 - 28, y + 27), radius=8, fill=COLORS["best"])
        vals = [
            str(r["step"]),
            f"{r['success_n']}/{r['rows']} ({r['success']:.1%})",
            f"{r['task']:.3f}",
            f"{r['progress']:.3f}",
            f"{r['protocol']:.3f}",
            f"{r['steps_mean']:.2f}",
        ]
        x = sx
        for val, w in zip(vals, widths):
            draw_text(draw, (x, y), val, 17, COLORS["text"], bold=(r is best))
            x += w

    draw_text(draw, (x0 + 28, y1 - 82), "Takeaway: step 108 ties highest success (5/32)", 18, COLORS["text"], bold=True)
    draw_text(draw, (x0 + 28, y1 - 57), "and has the best task/progress/protocol scores.", 18, COLORS["text"], bold=True)
    draw_text(draw, (x0 + 28, y1 - 30), "Main bottleneck: protocol_invalid dominates.", 18, COLORS["muted"])


def save_svg(rows):
    # Compact companion SVG for easy browser viewing. The PNG is the primary polished figure.
    w, h = 1200, 720
    steps = [r["step"] for r in rows]
    min_s, max_s = min(steps), max(steps)
    x0, y0, x1, y1 = 90, 110, 1120, 420

    def px(step):
        return x0 + (step - min_s) / (max_s - min_s) * (x1 - x0)

    def py(v):
        return y1 - v * (y1 - y0)

    def poly(key):
        return " ".join(f"{px(r['step']):.1f},{py(r[key]):.1f}" for r in rows)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{COLORS["bg"]}"/>',
        f'<text x="40" y="52" font-family="DejaVu Sans, Arial" font-size="28" font-weight="700" fill="{COLORS["text"]}">SFT checkpoint probe: 8 test queries x G=4</text>',
        f'<text x="40" y="82" font-family="DejaVu Sans, Arial" font-size="16" fill="{COLORS["muted"]}">temperature=0.2, max turns=15, response=10240; highlighted best overall checkpoint is global_step_108.</text>',
        f'<rect x="40" y="95" width="1120" height="370" rx="12" fill="white"/>',
    ]
    for i in range(6):
        v = i / 5
        y = py(v)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        lines.append(f'<text x="{x0-14}" y="{y+5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="13" fill="{COLORS["muted"]}">{v:.1f}</text>')
    bx = px(108)
    lines.append(f'<rect x="{bx-38:.1f}" y="{y0}" width="76" height="{y1-y0}" fill="{COLORS["best"]}"/>')
    for key, label in [("success", "Success"), ("task", "Task"), ("progress", "Progress"), ("protocol", "Protocol")]:
        lines.append(f'<polyline points="{poly(key)}" fill="none" stroke="{COLORS[key]}" stroke-width="4"/>')
        for r in rows:
            lines.append(f'<circle cx="{px(r["step"]):.1f}" cy="{py(r[key]):.1f}" r="5" fill="{COLORS[key]}" stroke="white" stroke-width="2"/>')
        lx = 120 + [("success","Success"),("task","Task"),("progress","Progress"),("protocol","Protocol")].index((key,label))*180
        lines.append(f'<line x1="{lx}" y1="448" x2="{lx+28}" y2="448" stroke="{COLORS[key]}" stroke-width="5"/>')
        lines.append(f'<text x="{lx+36}" y="453" font-family="DejaVu Sans, Arial" font-size="15" fill="{COLORS["muted"]}">{label}</text>')
    for r in rows:
        x = px(r["step"])
        lines.append(f'<text x="{x:.1f}" y="443" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="15" fill="{COLORS["muted"]}">{r["step"]}</text>')
        lines.append(f'<text x="{x:.1f}" y="{py(r["success"])-14:.1f}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="13" font-weight="700" fill="{COLORS["success"]}">{r["success_n"]}/{r["rows"]}</text>')
    lines.append(f'<text x="40" y="510" font-family="DejaVu Sans, Arial" font-size="22" font-weight="700" fill="{COLORS["text"]}">Failure modes</text>')
    modes = ["success", "protocol_invalid", "search_recall_gap", "workflow_invalid", "final_selection_after_full_recall_gap"]
    bar_x0, bar_y0 = 70, 550
    for idx, r in enumerate(rows):
        x = bar_x0 + idx * 210
        lines.append(f'<text x="{x+55}" y="690" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="15" fill="{COLORS["muted"]}">{r["step"]}</text>')
        y = 670
        for mode in modes:
            cnt = int(r["modes"].get(mode, 0))
            height = cnt / r["rows"] * 110
            y -= height
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="110" height="{height:.1f}" fill="{COLORS.get(mode, "#94a3b8")}"/>')
    lines.append("</svg>")
    OUT_SVG.write_text("\n".join(lines))


def main():
    rows = load_rows()
    image = Image.new("RGB", (1800, 1100), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw_text(draw, (54, 42), "SFT checkpoint probe results", 42, bold=True)
    draw_text(draw, (54, 91), "Test split: 8 queries, G=4 per checkpoint. Best overall checkpoint is global_step_108.", 23, COLORS["muted"])
    line_chart(draw, rows, (42, 130, 1758, 610))
    stacked_modes(draw, rows, (42, 640, 790, 1058))
    table_panel(draw, rows, (820, 640, 1758, 1058))
    image.save(OUT_PNG)
    save_svg(rows)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
