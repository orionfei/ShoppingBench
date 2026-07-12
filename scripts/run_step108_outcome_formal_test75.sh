#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ATTEMPT_DIR="${ATTEMPT_DIR:-${1:-}}"
[[ -n "$ATTEMPT_DIR" && -d "$ATTEMPT_DIR" ]] || {
  echo "usage: ATTEMPT_DIR=rollouts/.../attempt_b16 $0" >&2
  exit 2
}
ATTEMPT_DIR="$(realpath "$ATTEMPT_DIR")"
PYTHON_BIN="/root/miniconda3/envs/shoppingbench-verl/bin/python"
RAY_BIN="/root/miniconda3/envs/shoppingbench-verl/bin/ray"
STEP108="checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108"
TEST_DATA="dataset/shoppingbench_query_rl_v2/test.parquet"
ANALYSIS="${ANALYSIS:-$ATTEMPT_DIR/analysis.json}"

"$PYTHON_BIN" scripts/analyze_step108_outcome_grpo.py "$ATTEMPT_DIR" --output "$ANALYSIS"
read -r experiment best_step actor_dir < <("$PYTHON_BIN" - "$ANALYSIS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
step = d.get("best_checkpoint_step")
path = d.get("best_checkpoint_path")
if step is None or not path:
    raise SystemExit("no health-eligible saved checkpoint was selected")
print(d["manifest"]["experiment_name"], step, path + "/actor")
PY
)

[[ -d "$actor_dir" ]] || { echo "missing actor checkpoint: $actor_dir" >&2; exit 2; }
MERGED_DIR="${MERGED_DIR:-outputs/merged_hf_model/${experiment}_step${best_step}}"
RUN_ROOT="${RUN_ROOT:-rollouts/${experiment}_test75}"
REPORT_ROOT="${REPORT_ROOT:-reports/${experiment}_test75}"
FIGURE_ROOT="${FIGURE_ROOT:-docs/figures/${experiment}_test75}"
mkdir -p "$MERGED_DIR" "$RUN_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT"

if [[ ! -f "$MERGED_DIR/config.json" ]]; then
  PYTHONPATH="$ROOT_DIR/src/rl${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp --local_dir "$actor_dir" --target_dir "$MERGED_DIR"
fi

cleanup_ray() {
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

run_eval() {
  local label="$1" model="$2" run_dir="$RUN_ROOT/$1" report_dir="$REPORT_ROOT/$1"
  mkdir -p "$run_dir" "$report_dir"
  if [[ -f "$run_dir/.done" ]]; then
    echo "[formal-test75] already complete; refusing a second test pass: $label"
    return
  fi
  if find "$run_dir" -name '*.jsonl' -type f -print -quit | grep -q .; then
    echo "[formal-test75] partial test output exists; preserve it and inspect before retry: $run_dir" >&2
    exit 3
  fi
  LABEL="$label" MODEL="$model" DATA="$TEST_DATA" MANIFEST="$run_dir/manifest.json" \
    BEST_STEP="$best_step" "$PYTHON_BIN" - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
d = {
    "schema_version": 1, "run_id": os.environ["LABEL"], "checkpoint_path": os.environ["MODEL"],
    "best_training_step": int(os.environ["BEST_STEP"]), "dataset": os.environ["DATA"],
    "dataset_sha256": sha(os.environ["DATA"]), "temperature": 0.2, "top_p": 0.9,
    "seed": 0, "group_size": 8, "queries": 75, "expected_trajectories": 600,
    "max_response_length": 10240, "max_assistant_turns": 15, "rollout_max_num_seqs": 8,
    "engine_n": 1, "reward_mode": "paper_asr_plus_terminate", "test_used_for_selection": False,
    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "started_unix": time.time(), "status": "running",
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(d, indent=2) + "\n")
PY
  local start end
  start="$(date +%s)"
  MODEL_PATH="$model" PROJECT_NAME=shoppingbench-rl-formal-test75 EXPERIMENT_NAME="${experiment}_${label}" \
  TRAIN_FILES="$TEST_DATA" VAL_FILES="$TEST_DATA" NGPUS_PER_NODE=2 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=75 \
  PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
  MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=10240 PPO_MAX_TOKEN_LEN_PER_GPU=12288 \
  ROLLOUT_N=8 TRAIN_ROLLOUT_TEMPERATURE=.2 TRAIN_ROLLOUT_TOP_P=.9 \
  VAL_ROLLOUT_TEMPERATURE=.2 VAL_ROLLOUT_TOP_P=.9 MAX_ASSISTANT_TURNS=15 MAX_USER_TURNS=15 \
  ROLLOUT_MAX_MODEL_LEN=12288 ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288 ROLLOUT_MAX_NUM_SEQS=8 \
  ROLLOUT_APPLY_MAX_NUM_SEQS=True ROLLOUT_ENABLE_PREFIX_CACHING=True ROLLOUT_ENFORCE_EAGER=False \
  ROLLOUT_FREE_CACHE_ENGINE=False ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 ROLLOUT_AGENT_NUM_WORKERS=8 \
  STABLE_ROLLOUT_SAMPLING=True STABLE_ROLLOUT_SEED_REQUESTS=True STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True \
  STABLE_ROLLOUT_STABLE_SERVER_ORDER=True STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True STABLE_ROLLOUT_SEED_BASE=0 \
  STABLE_ROLLOUT_OFFSET_BASE=0 STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True \
  SHOPPINGBENCH_REWARD_MODE=asr_terminal VAL_ONLY=True VAL_BEFORE_TRAIN=True TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=1 \
  TEST_FREQ=1 SAVE_FREQ=100000 LOGGER=console REQUIRE_SEARCH_SERVER=1 VALIDATION_DATA_DIR="$run_dir" \
  ROLLOUT_DATA_DIR="$run_dir/train_unused" bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh \
  >"$report_dir/run.log" 2>&1
  end="$(date +%s)"
  local trajectories
  trajectories="$(find "$run_dir" -name '*.jsonl' -type f -exec cat {} + | wc -l)"
  [[ "$trajectories" -eq 600 ]] || { echo "expected 600 trajectories, got $trajectories" >&2; exit 4; }
  ELAPSED="$((end-start))" TRAJECTORIES="$trajectories" MANIFEST="$run_dir/manifest.json" "$PYTHON_BIN" - <<'PY'
import json, os, time
from pathlib import Path
p = Path(os.environ["MANIFEST"]); d = json.loads(p.read_text())
d.update(status="completed", trajectories=int(os.environ["TRAJECTORIES"]),
         elapsed_seconds=int(os.environ["ELAPSED"]), finished_unix=time.time())
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  "$PYTHON_BIN" scripts/analyze_outcome_sampling_sweep.py "$run_dir" \
    --output "$report_dir/analysis.json" --group-size 8 --allow-truncation-as-outcome
  touch "$run_dir/.done"
  cleanup_ray
}

run_eval untouched_step108 "$STEP108"
run_eval best_step${best_step} "$MERGED_DIR"

"$PYTHON_BIN" scripts/analyze_outcome_sampling_sweep.py \
  "$RUN_ROOT/untouched_step108" "$RUN_ROOT/best_step${best_step}" \
  --output "$REPORT_ROOT/analysis.json" --group-size 8 --allow-truncation-as-outcome
"$PYTHON_BIN" scripts/plot_outcome_sampling_sweep.py "$REPORT_ROOT/analysis.json" --output-dir "$FIGURE_ROOT"
"$PYTHON_BIN" scripts/plot_formal_test75_comparison.py "$REPORT_ROOT/analysis.json" \
  --output "$FIGURE_ROOT/formal_test75_comparison.png"
echo "[formal-test75] comparison=$REPORT_ROOT/analysis.json figures=$FIGURE_ROOT"
