#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PILOT_NAME="${PILOT_NAME:-candidate}"
TRAIN_ROLLOUT_TEMPERATURE="${TRAIN_ROLLOUT_TEMPERATURE:?set TRAIN_ROLLOUT_TEMPERATURE}"
TRAIN_ROLLOUT_TOP_P="${TRAIN_ROLLOUT_TOP_P:?set TRAIN_ROLLOUT_TOP_P}"
TRAIN_SEED="${TRAIN_SEED:-1}"

export MODEL_PATH="${MODEL_PATH:-checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108}"
export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-rl-outcome-pilot}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-step108_outcome_pilot_${PILOT_NAME}_t${TRAIN_ROLLOUT_TEMPERATURE}_p${TRAIN_ROLLOUT_TOP_P}}"

export TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_query_rl_v2/train.parquet}"
export VAL_FILES="${VAL_FILES:-dataset/shoppingbench_query_rl_v2/validation.parquet}"
export NGPUS_PER_NODE=2
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
# With 10k multi-turn responses, a 32k dynamic token pack can require another
# ~8.7 GiB during backward.  12288 preserves the same logical mini-batch/loss
# and uses gradient accumulation, while bounding each activation pack.
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export ROLLOUT_N=8
export TRAIN_ROLLOUT_TEMPERATURE
export TRAIN_ROLLOUT_TOP_P
export VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.2}"
export VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-0.9}"
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=10240
export MAX_ASSISTANT_TURNS=15
export MAX_USER_TURNS=15
export ROLLOUT_MAX_MODEL_LEN=12288
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_APPLY_MAX_NUM_SEQS=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export ROLLOUT_ENFORCE_EAGER=False
export ROLLOUT_FREE_CACHE_ENGINE=False
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
export ROLLOUT_AGENT_NUM_WORKERS=8
export STABLE_ROLLOUT_SAMPLING=True
export STABLE_ROLLOUT_SEED_REQUESTS=True
export STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True
export STABLE_ROLLOUT_STABLE_SERVER_ORDER=True
export STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True
export STABLE_ROLLOUT_SEED_BASE="$TRAIN_SEED"
export STABLE_ROLLOUT_OFFSET_BASE=0
export STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True

export SHOPPINGBENCH_REWARD_MODE=asr_terminal
export SHOPPINGBENCH_PROTOCOL_WEIGHT_START=0
export SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS=0
export SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION=0
export SHOPPINGBENCH_STEP_PENALTY=0
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-12}"
export SAVE_FREQ="${SAVE_FREQ:-4}"
export TEST_FREQ="${TEST_FREQ:-4}"
export VAL_BEFORE_TRAIN=True
export ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACTOR_CHECKPOINT_SAVE_CONTENTS:-['model','optimizer','extra']}"
export ACTOR_CHECKPOINT_LOAD_CONTENTS="$ACTOR_CHECKPOINT_SAVE_CONTENTS"
export LOGGER="${LOGGER:-console}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/train}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/validation}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] step108 checkpoint missing: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILES" || ! -f "$VAL_FILES" ]]; then
  echo "[ERROR] train/validation parquet missing: $TRAIN_FILES / $VAL_FILES" >&2
  exit 2
fi

echo "[outcome-pilot] name=$PILOT_NAME train_t=$TRAIN_ROLLOUT_TEMPERATURE train_p=$TRAIN_ROLLOUT_TOP_P val_t=$VAL_ROLLOUT_TEMPERATURE val_p=$VAL_ROLLOUT_TOP_P seed=$TRAIN_SEED steps=$TOTAL_TRAINING_STEPS"
exec bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh \
  +data.seed="$TRAIN_SEED" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  "$@"
