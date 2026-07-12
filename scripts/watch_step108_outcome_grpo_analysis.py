#!/usr/bin/env python3
"""Continuously refresh formal GRPO analysis and figures without affecting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def signature(attempt: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [attempt / "trainer_metrics.jsonl", attempt / "system_metrics.csv"]
    paths.extend((attempt / "train").glob("*.jsonl"))
    paths.extend((attempt / "validation").glob("*.jsonl"))
    return tuple(sorted((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths if path.exists()))


def terminal(attempt: Path) -> bool:
    path = attempt / "manifest.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("status") in {"completed", "failed"}


def refresh(attempt: Path) -> None:
    analysis = attempt / "analysis.partial.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_step108_outcome_grpo.py"), str(attempt), "--output", str(analysis)],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/plot_step108_outcome_grpo.py"), str(analysis),
         "--output-dir", str(attempt / "figures")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve()
    previous: tuple[tuple[str, int, int], ...] = ()
    while True:
        current = signature(attempt)
        if current != previous:
            try:
                refresh(attempt)
                print(f"[analysis-watcher] refreshed {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
                previous = current
            except Exception as error:
                print(f"[analysis-watcher] refresh failed; will retry: {error}", file=sys.stderr, flush=True)
        if terminal(attempt):
            if signature(attempt) != previous:
                continue
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
