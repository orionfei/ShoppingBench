#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_RUN="${TRAIN_RUN:-rollouts/step108_outcome_grpo_v3_dapo_20260711_054926}"
TRAIN_MANIFEST="$TRAIN_RUN/manifest.json"
PYTHON_BIN="/root/miniconda3/envs/shoppingbench-verl/bin/python"
RAY_BIN="/root/miniconda3/envs/shoppingbench-verl/bin/ray"
STEP108="checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108"
TEST_DATA="dataset/shoppingbench_query_rl_v3/test.parquet"

[[ -f "$TRAIN_MANIFEST" ]] || { echo "missing training manifest: $TRAIN_MANIFEST" >&2; exit 2; }
read -r experiment best_step checkpoint_root < <("$PYTHON_BIN" - "$TRAIN_MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
step = d.get("best_step")
root = d.get("checkpoint_root")
if step is None or not root:
    raise SystemExit("training manifest has no validation-selected checkpoint")
print(d["experiment"], int(step), root)
PY
)

actor_dir="$checkpoint_root/best/actor"
[[ -d "$actor_dir" ]] || { echo "missing pinned best actor: $actor_dir" >&2; exit 2; }
merged_dir="${MERGED_DIR:-outputs/merged_hf_model/${experiment}_best_step${best_step}}"
run_root="${RUN_ROOT:-rollouts/${experiment}_test250}"
report_root="${REPORT_ROOT:-reports/${experiment}_test250}"
figure_root="${FIGURE_ROOT:-docs/figures/${experiment}_test250}"
mkdir -p "$run_root" "$report_root" "$figure_root" outputs/merged_hf_model

if [[ ! -f "$merged_dir/config.json" ]]; then
  PYTHONPATH="$ROOT_DIR/src/rl${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp --local_dir "$actor_dir" --target_dir "$merged_dir"
fi

cleanup_ray() { "$RAY_BIN" stop --force >/dev/null 2>&1 || true; }
trap cleanup_ray EXIT

run_eval() {
  local label="$1" model="$2" out="$run_root/$1" report="$report_root/$1"
  mkdir -p "$out" "$report"
  if [[ -f "$out/.done" ]]; then
    echo "[test250] already complete; refusing a second pass: $label"
    return
  fi
  if find "$out" -name '*.jsonl' -type f -print -quit | grep -q .; then
    echo "[test250] partial output exists; inspect before retry: $out" >&2
    exit 3
  fi
  LABEL="$label" MODEL="$model" DATA="$TEST_DATA" BEST_STEP="$best_step" MANIFEST="$out/manifest.json" \
    "$PYTHON_BIN" - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
d={"schema_version":1,"run_id":os.environ["LABEL"],"checkpoint_path":os.environ["MODEL"],
   "best_training_step":int(os.environ["BEST_STEP"]),"dataset":os.environ["DATA"],
   "dataset_sha256":sha(os.environ["DATA"]),"temperature":.2,"top_p":.9,"seed":0,
   "group_size":8,"queries":250,"expected_trajectories":2000,"max_response_length":10240,
   "max_assistant_turns":15,"rollout_max_num_seqs":8,"engine_n":1,
   "reward_mode":"terminal_asr=paper_asr*terminate_success","test_used_for_selection":False,
   "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
   "started_unix":time.time(),"status":"running"}
Path(os.environ["MANIFEST"]).write_text(json.dumps(d,indent=2)+"\n")
PY
  local start end trajectories
  start="$(date +%s)"
  MODEL_PATH="$model" PROJECT_NAME=shoppingbench-rl-v3-test250 EXPERIMENT_NAME="${experiment}_${label}" \
  TRAIN_FILES="$TEST_DATA" VAL_FILES="$TEST_DATA" NGPUS_PER_NODE=4 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=250 \
  PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
  MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=10240 PPO_MAX_TOKEN_LEN_PER_GPU=12288 ROLLOUT_N=8 \
  TRAIN_ROLLOUT_TEMPERATURE=.2 TRAIN_ROLLOUT_TOP_P=.9 VAL_ROLLOUT_TEMPERATURE=.2 VAL_ROLLOUT_TOP_P=.9 \
  MAX_ASSISTANT_TURNS=15 MAX_USER_TURNS=15 ROLLOUT_MAX_MODEL_LEN=12288 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288 ROLLOUT_MAX_NUM_SEQS=8 ROLLOUT_APPLY_MAX_NUM_SEQS=True \
  ROLLOUT_ENABLE_PREFIX_CACHING=True ROLLOUT_ENFORCE_EAGER=False ROLLOUT_FREE_CACHE_ENGINE=False \
  ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 ROLLOUT_AGENT_NUM_WORKERS=8 STABLE_ROLLOUT_SAMPLING=True \
  STABLE_ROLLOUT_SEED_REQUESTS=True STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True \
  STABLE_ROLLOUT_STABLE_SERVER_ORDER=True STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True \
  STABLE_ROLLOUT_SEED_BASE=0 STABLE_ROLLOUT_OFFSET_BASE=0 STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True \
  SHOPPINGBENCH_REWARD_MODE=asr_terminal VAL_ONLY=True VAL_BEFORE_TRAIN=True TOTAL_EPOCHS=1 \
  TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 SAVE_FREQ=100000 LOGGER=console REQUIRE_SEARCH_SERVER=1 \
  VALIDATION_DATA_DIR="$out" ROLLOUT_DATA_DIR="$out/train_unused" \
  bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh >"$report/run.log" 2>&1
  end="$(date +%s)"
  trajectories="$(find "$out" -name '*.jsonl' -type f -exec cat {} + | wc -l)"
  [[ "$trajectories" -eq 2000 ]] || { echo "expected 2000 trajectories, got $trajectories" >&2; exit 4; }
  ELAPSED="$((end-start))" TRAJECTORIES="$trajectories" MANIFEST="$out/manifest.json" "$PYTHON_BIN" - <<'PY'
import json, os, time
from pathlib import Path
p=Path(os.environ["MANIFEST"]); d=json.loads(p.read_text())
d.update(status="completed",trajectories=int(os.environ["TRAJECTORIES"]),
         elapsed_seconds=int(os.environ["ELAPSED"]),finished_unix=time.time())
p.write_text(json.dumps(d,indent=2)+"\n")
PY
  "$PYTHON_BIN" scripts/analyze_outcome_sampling_sweep.py "$out" --output "$report/analysis.json" \
    --group-size 8 --allow-truncation-as-outcome
  touch "$out/.done"
  cleanup_ray
}

run_eval untouched_step108 "$STEP108"
run_eval "best_step${best_step}" "$merged_dir"
"$PYTHON_BIN" scripts/analyze_outcome_sampling_sweep.py "$run_root/untouched_step108" \
  "$run_root/best_step${best_step}" --output "$report_root/analysis.json" --group-size 8 \
  --allow-truncation-as-outcome
"$PYTHON_BIN" scripts/plot_outcome_sampling_sweep.py "$report_root/analysis.json" --output-dir "$figure_root"
"$PYTHON_BIN" scripts/plot_formal_test75_comparison.py "$report_root/analysis.json" \
  --output "$figure_root/test250_comparison.png"
echo "[test250] comparison=$report_root/analysis.json figures=$figure_root"
