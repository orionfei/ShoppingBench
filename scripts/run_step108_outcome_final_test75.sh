#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PILOT_ANALYSIS="${PILOT_ANALYSIS:-reports/step108_outcome_pilots_20260710/analysis.json}"
PILOT_ROOT="${PILOT_ROOT:-rollouts/step108_outcome_pilots_20260710}"
RUN_ROOT="${RUN_ROOT:-rollouts/step108_outcome_test75_20260710}"
REPORT_ROOT="${REPORT_ROOT:-reports/step108_outcome_test75_20260710}"
FIGURE_ROOT="${FIGURE_ROOT:-docs/figures/grpo_outcome_sampling_step108/test75}"
TEST_DATA="${TEST_DATA:-dataset/shoppingbench_query_rl_v2/test.parquet}"
STEP108="checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108"
mkdir -p "$RUN_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT" outputs/merged_hf_model

read_winner() {
  python - "$PILOT_ANALYSIS" <<'PY'
import json, sys
items=json.load(open(sys.argv[1])).get("pilot_summaries") or []
if not items: raise SystemExit("no completed pilot summary")
w=min(items, key=lambda x:x.get("rank_by_registered_outcome_order",999))
print(w["candidate"], w["temperature"], w["top_p"])
PY
}

read -r winner_label winner_t winner_p < <(read_winner)
pilot_dir="$(find "$PILOT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${winner_label}_t*_p*" | head -1)"
[[ -n "$pilot_dir" ]] || { echo "winner pilot directory not found" >&2; exit 2; }
slug="$(basename "$pilot_dir")"
actor_dir="checkpoints/shoppingbench-rl-outcome-pilot/step108_outcome_pilot_${slug}/global_step_12/actor"
merged_dir="outputs/merged_hf_model/step108_outcome_pilot_${slug}_step12"
if [[ ! -f "$merged_dir/config.json" ]]; then
  [[ -d "$actor_dir" ]] || { echo "pilot actor checkpoint missing: $actor_dir" >&2; exit 2; }
  python -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$merged_dir"
fi

run_eval() {
  local label="$1" model="$2" run_dir="$RUN_ROOT/$1" report_dir="$REPORT_ROOT/$1"
  mkdir -p "$run_dir" "$report_dir"
  if [[ -s "$run_dir/.done" ]]; then
    echo "[test75] skip completed $label"
    return
  fi
  LABEL="$label" MODEL="$model" DATA="$TEST_DATA" MANIFEST="$run_dir/manifest.json" python - <<'PY'
import hashlib,json,os,subprocess,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
d={"schema_version":1,"run_id":os.environ["LABEL"],"checkpoint_path":os.environ["MODEL"],
"dataset":os.environ["DATA"],"dataset_sha256":sha(os.environ["DATA"]),"temperature":.2,"top_p":.9,
"seed":0,"group_size":8,"queries":75,"expected_trajectories":600,"max_response_length":10240,
"max_assistant_turns":15,"rollout_max_num_seqs":8,"engine_n":1,"reward_mode":"paper_asr_plus_terminate",
"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"started_unix":time.time()}
Path(os.environ["MANIFEST"]).write_text(json.dumps(d,indent=2)+"\n")
PY
  local start end
  start="$(date +%s)"
  MODEL_PATH="$model" PROJECT_NAME=shoppingbench-rl-outcome-test75 EXPERIMENT_NAME="$label" \
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
  ROLLOUT_DATA_DIR="$run_dir/train_unused" bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh 2>&1 | tee "$report_dir/run.log"
  end="$(date +%s)"
  ELAPSED="$((end-start))" MANIFEST="$run_dir/manifest.json" python - <<'PY'
import json,os,time
from pathlib import Path
p=Path(os.environ["MANIFEST"]);d=json.loads(p.read_text());d.update(elapsed_seconds=int(os.environ["ELAPSED"]),finished_unix=time.time());p.write_text(json.dumps(d,indent=2)+"\n")
PY
  python scripts/analyze_outcome_sampling_sweep.py "$run_dir" --output "$report_dir/analysis.json" --group-size 8
  touch "$run_dir/.done"
}

run_eval untouched_step108 "$STEP108"
run_eval winner_step12 "$merged_dir"
python scripts/analyze_outcome_sampling_sweep.py "$RUN_ROOT/untouched_step108" "$RUN_ROOT/winner_step12" \
  --output "$REPORT_ROOT/analysis.json" --group-size 8
python scripts/plot_outcome_sampling_sweep.py "$REPORT_ROOT/analysis.json" --output-dir "$FIGURE_ROOT"
