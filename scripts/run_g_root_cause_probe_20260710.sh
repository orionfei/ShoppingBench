#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-g_root_cause_probe_20260710}"
G="${G:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
SEED_BASE="${SEED_BASE:-0}"
ROLLOUT_OFFSET_BASE="${ROLLOUT_OFFSET_BASE:-0}"

CHECKPOINT="${CHECKPOINT:-checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108}"
QUERY_VAL_FILES="${QUERY_VAL_FILES:-dataset/probe/sft_clean924_test8_state_local/probe.parquet}"
PRODUCT_CACHE="${PRODUCT_CACHE:-dataset/probe/sft_clean924_test8_state_local/product_cache.json}"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$QUERY_VAL_FILES" ]]; then
  echo "[ERROR] validation parquet not found: $QUERY_VAL_FILES" >&2
  exit 1
fi
if ! curl --noproxy "127.0.0.1,localhost" -fsS "${SEARCH_SERVER_URL:-http://127.0.0.1:5631/}" >/dev/null; then
  echo "[ERROR] ShoppingBench search server is unavailable" >&2
  exit 1
fi

REPORT_ROOT="${REPORT_ROOT:-reports/${RUN_ID}}"
OUT_ROOT="${OUT_ROOT:-rollouts/${RUN_ID}}"
SELECTED_ROOT="${REPORT_ROOT}/selected_checkpoints"
mkdir -p "$REPORT_ROOT" "$OUT_ROOT" "$SELECTED_ROOT"
ln -sfn "$(realpath "$CHECKPOINT")" "$SELECTED_ROOT/global_step_108"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS_PER_NODE=2
export MODEL_PATH="$CHECKPOINT"
export QUERY_VAL_FILES
export SHOPPINGBENCH_PRODUCT_CACHE="$PRODUCT_CACHE"
export OUT_ROOT REPORT_ROOT
export EXPERIMENT_PREFIX="$RUN_ID"

export ROLLOUT_N="$G"
export ROLLOUT_TEMPERATURE=0.2
export ROLLOUT_TOP_P=0.9
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=10240
export MAX_ASSISTANT_TURNS=15
export MAX_USER_TURNS=15
export VAL_BATCH_SIZE
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export PPO_MAX_TOKEN_LEN_PER_GPU=32768

export ROLLOUT_MAX_MODEL_LEN=12288
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export ROLLOUT_APPLY_MAX_NUM_SEQS="${ROLLOUT_APPLY_MAX_NUM_SEQS:-True}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"

# These controls are opt-in in the rollout implementation. Disabling stable
# sampling returns to the legacy UUID/router path; the engine-level n override
# is pinned to 1 here so G is represented only by outer prompt repetition.
export STABLE_ROLLOUT_SAMPLING=True
export STABLE_ROLLOUT_SEED_REQUESTS=True
export STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True
export STABLE_ROLLOUT_STABLE_SERVER_ORDER=True
export STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True
export STABLE_ROLLOUT_SEED_BASE="$SEED_BASE"
export STABLE_ROLLOUT_OFFSET_BASE="$ROLLOUT_OFFSET_BASE"
export STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True

export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=1
export TEST_FREQ=1
export SAVE_FREQ=100000
export LOGGER=console
export REQUIRE_SEARCH_SERVER=1

echo "[g-probe] run=$RUN_ID checkpoint=global_step_108 G=$G val_batch=$VAL_BATCH_SIZE seed=$SEED_BASE offset=$ROLLOUT_OFFSET_BASE eager=$ROLLOUT_ENFORCE_EAGER apply_max_seqs=$ROLLOUT_APPLY_MAX_NUM_SEQS max_seqs=$ROLLOUT_MAX_NUM_SEQS prefix_cache=$ROLLOUT_ENABLE_PREFIX_CACHING"
exec bash src/rl/evaluate_sft_checkpoints_query_variance_qwen3_4b_a800.sh "$SELECTED_ROOT"
