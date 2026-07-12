#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-coarse}"
CHECKPOINT="${CHECKPOINT:-checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108}"
PROBE_ROOT="${PROBE_ROOT:-dataset/probe/step108_outcome_sampling}"
RUN_ROOT="${RUN_ROOT:-rollouts/step108_outcome_sampling_20260710}"
REPORT_ROOT="${REPORT_ROOT:-reports/step108_outcome_sampling_20260710}"
FIGURE_ROOT="${FIGURE_ROOT:-docs/figures/grpo_outcome_sampling_step108}"
DRY_RUN="${DRY_RUN:-0}"
SWEEP_MAX_RESPONSE_LENGTH="${SWEEP_MAX_RESPONSE_LENGTH:-10240}"
SWEEP_MAX_MODEL_LEN="${SWEEP_MAX_MODEL_LEN:-12288}"
SWEEP_MAX_NUM_BATCHED_TOKENS="${SWEEP_MAX_NUM_BATCHED_TOKENS:-12288}"
ALLOW_TRUNCATION_AS_OUTCOME="${ALLOW_TRUNCATION_AS_OUTCOME:-0}"
CONFIRM_SOURCE_ANALYSIS="${CONFIRM_SOURCE_ANALYSIS:-$REPORT_ROOT/coarse_analysis.json}"
ANALYSIS_ARGS=()
if [[ "$ALLOW_TRUNCATION_AS_OUTCOME" == "1" ]]; then
  ANALYSIS_ARGS+=(--allow-truncation-as-outcome)
fi

TEMPERATURES=(0.2 0.4 0.6 0.8 1.0)
TOP_PS=(0.8 0.9 0.95)
CONFIRM_SEEDS=(0 1 2)

mkdir -p "$RUN_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT"

slug_float() {
  local value="$1"
  value="${value/./}"
  printf '%s' "$value"
}

write_manifest() {
  local path="$1" run_id="$2" split="$3" temperature="$4" top_p="$5" seed="$6" dataset="$7"
  RUN_ID="$run_id" SPLIT="$split" TEMPERATURE="$temperature" TOP_P="$top_p" SEED="$seed" \
  MAX_RESPONSE_LENGTH="$SWEEP_MAX_RESPONSE_LENGTH" MAX_MODEL_LEN="$SWEEP_MAX_MODEL_LEN" \
  MAX_BATCHED_TOKENS="$SWEEP_MAX_NUM_BATCHED_TOKENS" \
  ALLOW_TRUNCATION_AS_OUTCOME="$ALLOW_TRUNCATION_AS_OUTCOME" \
  DATASET="$dataset" CHECKPOINT="$CHECKPOINT" MANIFEST_PATH="$path" python - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

dataset = os.environ["DATASET"]
checkpoint = os.environ["CHECKPOINT"]
try:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    revision = None
try:
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
except Exception:
    dirty = None
code_files = [
    "scripts/reward_shoppingbench_asr_batch.py",
    "scripts/run_step108_outcome_sampling_sweep.sh",
    "src/rl/run_grpo_qwen3_1_7b_query_verl.sh",
    "src/rl/run_grpo_qwen3_4b_state_folded_a800.sh",
    "src/rl/verl/workers/reward_manager/batch.py",
]
payload = {
    "schema_version": 1,
    "run_id": os.environ["RUN_ID"],
    "stage": os.environ["SPLIT"],
    "checkpoint": "global_step_108",
    "checkpoint_path": checkpoint,
    "dataset": dataset,
    "dataset_sha256": sha256(dataset),
    "temperature": float(os.environ["TEMPERATURE"]),
    "top_p": float(os.environ["TOP_P"]),
    "seed": int(os.environ["SEED"]),
    "group_size": 8,
    "queries": 16,
    "expected_trajectories": 128,
    "gpus": 2,
    "max_response_length": int(os.environ["MAX_RESPONSE_LENGTH"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "max_num_batched_tokens": int(os.environ["MAX_BATCHED_TOKENS"]),
    "max_assistant_turns": 15,
    "max_user_turns": 15,
    "rollout_max_num_seqs": 8,
    "enable_prefix_caching": True,
    "enforce_eager": False,
    "stable_sampling": True,
    "engine_n": 1,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "reward_mode": "paper_asr_plus_terminate",
    "token_limit_policy": (
        "outcome_zero_diagnostic" if os.environ["ALLOW_TRUNCATION_AS_OUTCOME"] == "1" else "hard_failure_gate"
    ),
    "git_revision": revision,
    "git_dirty": dirty,
    "code_files_sha256": {
        item: sha256(item) for item in code_files if Path(item).is_file()
    },
    "started_unix": time.time(),
}
Path(os.environ["MANIFEST_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

finish_manifest() {
  local path="$1" elapsed="$2"
  MANIFEST_PATH="$path" ELAPSED="$elapsed" python - <<'PY'
import json, os, time
from pathlib import Path
path = Path(os.environ["MANIFEST_PATH"])
payload = json.loads(path.read_text())
payload["elapsed_seconds"] = float(os.environ["ELAPSED"])
payload["finished_unix"] = time.time()
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_one() {
  local stage="$1" dataset="$2" temperature="$3" top_p="$4" seed="$5"
  local t_slug p_slug run_id run_dir report_dir manifest start end
  t_slug="$(slug_float "$temperature")"
  p_slug="$(slug_float "$top_p")"
  run_id="${stage}_t${t_slug}_p${p_slug}_seed${seed}"
  run_dir="$RUN_ROOT/$run_id"
  report_dir="$REPORT_ROOT/$run_id"
  manifest="$run_dir/manifest.json"
  mkdir -p "$run_dir" "$report_dir"

  if [[ -s "$run_dir/.done" && -s "$report_dir/analysis.json" ]]; then
    echo "[sweep] skip completed $run_id"
    return 0
  fi

  echo "[sweep] run=$run_id dataset=$dataset T=$temperature top_p=$top_p seed=$seed"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  write_manifest "$manifest" "$run_id" "$stage" "$temperature" "$top_p" "$seed" "$dataset"
  start="$(date +%s)"

  MODEL_PATH="$CHECKPOINT" \
  PROJECT_NAME=shoppingbench-rl-outcome-sweep \
  EXPERIMENT_NAME="$run_id" \
  TRAIN_FILES="$dataset" VAL_FILES="$dataset" \
  NGPUS_PER_NODE=2 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=16 \
  PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
  MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH="$SWEEP_MAX_RESPONSE_LENGTH" PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  ROLLOUT_N=8 TRAIN_ROLLOUT_TEMPERATURE="$temperature" TRAIN_ROLLOUT_TOP_P="$top_p" \
  VAL_ROLLOUT_TEMPERATURE="$temperature" VAL_ROLLOUT_TOP_P="$top_p" \
  MAX_ASSISTANT_TURNS=15 MAX_USER_TURNS=15 \
  ROLLOUT_MAX_MODEL_LEN="$SWEEP_MAX_MODEL_LEN" ROLLOUT_MAX_NUM_BATCHED_TOKENS="$SWEEP_MAX_NUM_BATCHED_TOKENS" \
  ROLLOUT_MAX_NUM_SEQS=8 ROLLOUT_APPLY_MAX_NUM_SEQS=True \
  ROLLOUT_ENABLE_PREFIX_CACHING=True ROLLOUT_ENFORCE_EAGER=False ROLLOUT_FREE_CACHE_ENGINE=False \
  ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 ROLLOUT_AGENT_NUM_WORKERS=8 \
  STABLE_ROLLOUT_SAMPLING=True STABLE_ROLLOUT_SEED_REQUESTS=True \
  STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True STABLE_ROLLOUT_STABLE_SERVER_ORDER=True \
  STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True STABLE_ROLLOUT_SEED_BASE="$seed" \
  STABLE_ROLLOUT_OFFSET_BASE=0 STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True \
  SHOPPINGBENCH_REWARD_MODE=asr_terminal SHOPPINGBENCH_PRODUCT_CACHE=dataset/shoppingbench_query/product_cache.json \
  VAL_ONLY=True VAL_BEFORE_TRAIN=True TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 SAVE_FREQ=100000 \
  LOGGER=console REQUIRE_SEARCH_SERVER=1 \
  VALIDATION_DATA_DIR="$run_dir" ROLLOUT_DATA_DIR="$run_dir/train_unused" \
  bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh 2>&1 | tee "$report_dir/run.log"

  end="$(date +%s)"
  finish_manifest "$manifest" "$((end-start))"
  python scripts/analyze_outcome_sampling_sweep.py "$run_dir" \
    --output "$report_dir/analysis.json" --group-size 8 "${ANALYSIS_ARGS[@]}"
  touch "$run_dir/.done"
}

aggregate_stage() {
  local prefix="$1" output="$2"
  mapfile -t dirs < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" | sort)
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "[sweep] no runs found for prefix=$prefix" >&2
    return 1
  fi
  python scripts/enrich_outcome_manifests.py "${dirs[@]}"
  python scripts/analyze_outcome_sampling_sweep.py "${dirs[@]}" --output "$output" --group-size 8 "${ANALYSIS_ARGS[@]}"
  python scripts/plot_outcome_sampling_sweep.py "$output" --output-dir "$FIGURE_ROOT/$prefix"
}

coarse() {
  local dataset="$PROBE_ROOT/calibration16.parquet"
  for temperature in "${TEMPERATURES[@]}"; do
    for top_p in "${TOP_PS[@]}"; do
      run_one coarse "$dataset" "$temperature" "$top_p" 0
    done
  done
  [[ "$DRY_RUN" == "1" ]] || aggregate_stage coarse "$REPORT_ROOT/coarse_analysis.json"
}

top_three() {
  python - "$CONFIRM_SOURCE_ANALYSIS" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
ranking = report.get("config_ranking") or []
if len(ranking) < 3:
    ranking = [
        {"temperature": item["temperature"], "top_p": item["top_p"]}
        for item in report.get("ranking", [])
    ]
for item in ranking[:3]:
    print(item["temperature"], item["top_p"])
PY
}

confirm() {
  [[ -s "$CONFIRM_SOURCE_ANALYSIS" ]] || {
    echo "[sweep] missing confirm source analysis: $CONFIRM_SOURCE_ANALYSIS" >&2
    return 2
  }
  local dataset="$PROBE_ROOT/validation16.parquet"
  mapfile -t candidates < <(top_three)
  if [[ ${#candidates[@]} -lt 3 ]]; then
    echo "[sweep] fewer than three eligible coarse candidates" >&2
    return 2
  fi
  for candidate in "${candidates[@]}"; do
    read -r temperature top_p <<<"$candidate"
    for seed in "${CONFIRM_SEEDS[@]}"; do
      run_one confirm "$dataset" "$temperature" "$top_p" "$seed"
    done
  done
  [[ "$DRY_RUN" == "1" ]] || aggregate_stage confirm "$REPORT_ROOT/confirm_analysis.json"
}

length_recovery() {
  local dataset="$PROBE_ROOT/calibration16.parquet"
  # Chosen from the completed 15-point/10240 sweep: highest ASR/pass@8,
  # strongest alternate mixed signal, and the fixed baseline control.
  run_one len16384 "$dataset" 0.2 0.95 0
  run_one len16384 "$dataset" 0.6 0.8 0
  run_one len16384 "$dataset" 0.2 0.9 0
  [[ "$DRY_RUN" == "1" ]] || aggregate_stage len16384 "$REPORT_ROOT/length_recovery_analysis.json"
}

case "$STAGE" in
  coarse) coarse ;;
  confirm) confirm ;;
  length_recovery) length_recovery ;;
  analyze)
    aggregate_stage coarse "$REPORT_ROOT/coarse_analysis.json"
    if find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'confirm_*' | grep -q .; then
      aggregate_stage confirm "$REPORT_ROOT/confirm_analysis.json"
    fi
    ;;
  all) coarse; confirm ;;
  *)
    echo "Usage: $0 {coarse|confirm|length_recovery|analyze|all}" >&2
    exit 2
    ;;
esac
