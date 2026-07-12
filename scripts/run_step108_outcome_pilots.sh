#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIRM_ANALYSIS="${CONFIRM_ANALYSIS:-reports/step108_outcome_sampling_20260710/confirm_analysis.json}"
PILOT_ROOT="${PILOT_ROOT:-rollouts/step108_outcome_pilots_20260710}"
REPORT_ROOT="${REPORT_ROOT:-reports/step108_outcome_pilots_20260710}"
FIGURE_ROOT="${FIGURE_ROOT:-docs/figures/grpo_outcome_sampling_step108/pilots}"
TRAIN_SEED="${TRAIN_SEED:-108}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$PILOT_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT"

select_candidates() {
  python - "$CONFIRM_ANALYSIS" <<'PY'
import json, sys
runs = json.load(open(sys.argv[1])).get("config_ranking") or []
if not runs:
    raise SystemExit("no gate-eligible confirmed candidate")
best = runs[0]
baseline = {"temperature": 0.2, "top_p": 0.9}
same_as_baseline = abs(float(best["temperature"])-.2) < 1e-9 and abs(float(best["top_p"])-.9) < 1e-9
comparison = next((item for item in runs[1:] if not (abs(float(item["temperature"])-.2) < 1e-9 and abs(float(item["top_p"])-.9) < 1e-9)), None) if same_as_baseline else baseline
if comparison is None:
    raise SystemExit("baseline won but no second eligible confirmed candidate")
for label, item in (("winner", best), ("baseline" if not same_as_baseline else "runner_up", comparison)):
    print(label, item["temperature"], item["top_p"])
PY
}

write_manifest() {
  local path="$1" candidate="$2" temperature="$3" top_p="$4"
  MANIFEST_PATH="$path" CANDIDATE="$candidate" TEMPERATURE="$temperature" TOP_P="$top_p" TRAIN_SEED="$TRAIN_SEED" python - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
files = [
    "scripts/reward_shoppingbench_asr_batch.py", "scripts/run_step108_outcome_rl_pilot.sh",
    "src/rl/run_grpo_qwen3_1_7b_query_verl.sh", "src/rl/verl/workers/reward_manager/batch.py",
]
payload = {
    "schema_version": 1, "candidate": os.environ["CANDIDATE"],
    "checkpoint": "global_step_108",
    "checkpoint_path": "checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108",
    "train_dataset": "dataset/shoppingbench_query_rl_v2/train.parquet",
    "train_dataset_sha256": sha("dataset/shoppingbench_query_rl_v2/train.parquet"),
    "validation_dataset": "dataset/shoppingbench_query_rl_v2/validation.parquet",
    "validation_dataset_sha256": sha("dataset/shoppingbench_query_rl_v2/validation.parquet"),
    "temperature": float(os.environ["TEMPERATURE"]), "top_p": float(os.environ["TOP_P"]),
    "validation_temperature": .2, "validation_top_p": .9,
    "training_seed": int(os.environ["TRAIN_SEED"]), "group_size": 8,
    "training_steps": 12, "validation_steps": [0,4,8,12], "learning_rate": 1e-6,
    "train_batch_size": 8, "ppo_mini_batch_size": 8, "ppo_max_token_len_per_gpu": 12288,
    "optimizer": "AdamW",
    "max_response_length": 10240, "max_assistant_turns": 15, "rollout_max_num_seqs": 8,
    "engine_n": 1, "reward_mode": "paper_asr_plus_terminate",
    "git_revision": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git","status","--porcelain"], text=True).strip()),
    "code_files_sha256": {item: sha(item) for item in files}, "started_unix": time.time(),
}
Path(os.environ["MANIFEST_PATH"]).write_text(json.dumps(payload, indent=2)+"\n")
PY
}

run_pilot() {
  local candidate="$1" temperature="$2" top_p="$3"
  local slug="${candidate}_t${temperature/./}_p${top_p/./}" pilot_dir="$PILOT_ROOT/${candidate}_t${temperature/./}_p${top_p/./}"
  mkdir -p "$pilot_dir/validation" "$pilot_dir/train"
  if [[ -s "$pilot_dir/.done" ]]; then
    echo "[pilot] skip completed $candidate T=$temperature top_p=$top_p"
    return
  fi
  echo "[pilot] candidate=$candidate T=$temperature top_p=$top_p seed=$TRAIN_SEED"
  [[ "$DRY_RUN" == "1" ]] && return
  write_manifest "$pilot_dir/manifest.json" "$candidate" "$temperature" "$top_p"
  local start end status
  start="$(date +%s)"
  set +e
  PILOT_NAME="$slug" EXPERIMENT_NAME="step108_outcome_pilot_${slug}" \
  TRAIN_ROLLOUT_TEMPERATURE="$temperature" TRAIN_ROLLOUT_TOP_P="$top_p" TRAIN_SEED="$TRAIN_SEED" \
  ROLLOUT_DATA_DIR="$pilot_dir/train" VALIDATION_DATA_DIR="$pilot_dir/validation" LOGGER=console \
  bash scripts/run_step108_outcome_rl_pilot.sh 2>&1 | tee "$pilot_dir/run.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  if [[ "$status" != "0" ]]; then
    echo "[pilot] failed $candidate with status=$status" >&2
    return "$status"
  fi
  ELAPSED="$((end-start))" MANIFEST_PATH="$pilot_dir/manifest.json" python - <<'PY'
import json, os, time
from pathlib import Path
p=Path(os.environ["MANIFEST_PATH"]); d=json.loads(p.read_text())
d.update(elapsed_seconds=int(os.environ["ELAPSED"]), finished_unix=time.time())
p.write_text(json.dumps(d, indent=2)+"\n")
PY
  touch "$pilot_dir/.done"
}

[[ -s "$CONFIRM_ANALYSIS" ]] || { echo "missing $CONFIRM_ANALYSIS" >&2; exit 2; }
mapfile -t candidates < <(select_candidates)
for candidate in "${candidates[@]}"; do
  read -r label temperature top_p <<<"$candidate"
  run_pilot "$label" "$temperature" "$top_p"
done

if [[ "$DRY_RUN" != "1" ]]; then
  mapfile -t completed < <(find "$PILOT_ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/.done' ';' -print | sort)
  python scripts/analyze_outcome_pilots.py "${completed[@]}" \
    --sweep-analysis "$CONFIRM_ANALYSIS" --output "$REPORT_ROOT/analysis.json"
  python scripts/plot_outcome_sampling_sweep.py "$REPORT_ROOT/analysis.json" --output-dir "$FIGURE_ROOT"
fi
