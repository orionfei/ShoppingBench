#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


KEYS = [
    "success",
    "gt",
    "rule",
    "format",
    "length",
    "steps",
    "product",
    "has_recommend",
    "has_terminate",
    "shop",
    "budget",
    "kw",
    "title",
    "response",
]


def metric(report, key, field="mean"):
    return report.get("summary", {}).get(key, {}).get(field, 0)


def load_reports(paths):
    reports = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.exists():
                continue
            with candidate.open(encoding="utf-8") as fin:
                report = json.load(fin)
            report["_path"] = str(candidate)
            reports.append(report)
    return reports


def aggregate(reports):
    rows = []
    for report in reports:
        row = {
            "path": report["_path"],
            "model": report.get("model", ""),
            "task": report.get("task", ""),
            "num_scored": report.get("num_scored", 0),
        }
        for key in KEYS:
            row[f"{key}_mean"] = metric(report, key, "mean")
            row[f"{key}_variance"] = metric(report, key, "variance")
        rows.append(row)

    by_model = {}
    for row in rows:
        model = row["model"]
        current = by_model.setdefault(
            model,
            {
                "model": model,
                "tasks": [],
                "num_scored": 0,
                "success_weighted_sum": 0,
                "rule_weighted_sum": 0,
                "gt_weighted_sum": 0,
            },
        )
        n = row["num_scored"]
        current["tasks"].append(row["task"])
        current["num_scored"] += n
        current["success_weighted_sum"] += row["success_mean"] * n
        current["rule_weighted_sum"] += row["rule_mean"] * n
        current["gt_weighted_sum"] += row["gt_mean"] * n

    model_summary = []
    for item in by_model.values():
        n = item["num_scored"]
        model_summary.append(
            {
                "model": item["model"],
                "tasks": sorted(set(item["tasks"])),
                "num_scored": n,
                "success_weighted_mean": item["success_weighted_sum"] / n if n else 0,
                "rule_weighted_mean": item["rule_weighted_sum"] / n if n else 0,
                "gt_weighted_mean": item["gt_weighted_sum"] / n if n else 0,
            }
        )

    return {"reports": rows, "model_summary": sorted(model_summary, key=lambda x: x["model"])}


def print_table(rows):
    headers = [
        "model",
        "task",
        "n",
        "success",
        "rule",
        "gt",
        "format",
        "steps",
        "steps_var",
    ]
    print("\t".join(headers))
    for row in rows:
        values = [
            row["model"],
            row["task"],
            str(row["num_scored"]),
            f"{row['success_mean']:.3f}",
            f"{row['rule_mean']:.3f}",
            f"{row['gt_mean']:.3f}",
            f"{row['format_mean']:.3f}",
            f"{row['steps_mean']:.2f}",
            f"{row['steps_variance']:.2f}",
        ]
        print("\t".join(values))


def main():
    parser = argparse.ArgumentParser(description="Aggregate ShoppingBench rollout reward reports.")
    parser.add_argument("reports", nargs="+", help="Report JSON files or directories containing report JSONs.")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    result = aggregate(load_reports(args.reports))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
    print_table(result["reports"])


if __name__ == "__main__":
    main()
