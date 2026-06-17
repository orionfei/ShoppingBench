#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from run_rollout import react_loop  # noqa: E402


def main() -> None:
    subset_path = ROOT / "data" / "mimo_voucher_15_by_difficulty.jsonl"
    out_path = ROOT / "data" / "rollout_voucher_mimo_state_folded_15.jsonl"
    rows = [
        json.loads(line)
        for line in subset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    completed = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trajectory = json.loads(line)
            if trajectory:
                completed.add(trajectory[0]["extra_info"]["query"])

    config = {
        "task": "voucher",
        "system_prompt_file": "src/agent/prompt/rollout.md",
        "synthesize_file": str(subset_path.relative_to(ROOT)).replace("\\", "/"),
        "rollout_file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "base_url": "https://api.xiaomimimo.com/v1",
        "exclude_tools": ["web_search"],
        "history_compression": "state_folded",
        "state_max_candidates_per_search": 5,
        "model_config": {
            "model": "mimo-v2.5-pro-ultraspeed",
            "temperature": 0,
            "max_completion_tokens": 8192,
            "stream": True,
        },
    }

    print(
        f"existing={len(completed)} remaining={len(rows) - len(completed)}",
        flush=True,
    )
    for idx, row in enumerate(rows, 1):
        if row["query"] in completed:
            continue
        print(
            f"[{idx}/{len(rows)}] start {row.get('difficulty')} "
            f"{row.get('sample_id')} line={row.get('source_line')}",
            flush=True,
        )
        start = time.time()
        react_loop(row["query"], config)
        print(f"[{idx}/{len(rows)}] done elapsed={time.time() - start:.1f}s", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
