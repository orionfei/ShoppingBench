#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MODEL_PATH="${MODEL_PATH:-checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108}"
export TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_query_rl_v2/train.parquet}"
export VAL_FILES="${VAL_FILES:-dataset/shoppingbench_query_rl_v2/validation.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-rl-formal}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-step108_outcome_grpo_v2_b32_clip08_128_lr1e6_2ep}"
export TRAIN_SEED="${TRAIN_SEED:-108}"

export NGPUS_PER_NODE=2
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
case "$TRAIN_BATCH_SIZE" in
  32) STEPS_PER_EPOCH=20; export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" ;;
  16) STEPS_PER_EPOCH=40; export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" ;;
  8)  STEPS_PER_EPOCH=80; export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" ;;
  *) echo "[ERROR] supported TRAIN_BATCH_SIZE values: 32, 16, 8" >&2; exit 2 ;;
esac
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$((STEPS_PER_EPOCH * TOTAL_EPOCHS))}"
export SAVE_FREQ="${SAVE_FREQ:-$((STEPS_PER_EPOCH / 2))}"
export TEST_FREQ="${TEST_FREQ:-$((STEPS_PER_EPOCH / 4))}"
export VAL_BATCH_SIZE=16
export VAL_BEFORE_TRAIN=True

export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export PPO_MAX_TOKEN_LEN_PER_GPU=12288
# vLLM free_cache_engine uses CuMemAllocator's memory pool. PyTorch expandable
# segments are explicitly incompatible with that pool (vLLM assertion).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export ACTOR_CHECKPOINT_SAVE_CONTENTS="['model','extra']"
export ACTOR_CHECKPOINT_LOAD_CONTENTS="['model','extra']"

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=10240
export ROLLOUT_N=8
export TRAIN_ROLLOUT_TEMPERATURE=0.4
export TRAIN_ROLLOUT_TOP_P=0.95
export VAL_ROLLOUT_TEMPERATURE=0.2
export VAL_ROLLOUT_TOP_P=0.9
export ROLLOUT_MAX_MODEL_LEN=12288
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288
export ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_APPLY_MAX_NUM_SEQS=True
export ROLLOUT_ENABLE_PREFIX_CACHING=True
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.55
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
export ROLLOUT_ENFORCE_EAGER=False
export ROLLOUT_FREE_CACHE_ENGINE=True
export ROLLOUT_AGENT_NUM_WORKERS=8
export MAX_ASSISTANT_TURNS=15
export MAX_USER_TURNS=15

export STABLE_ROLLOUT_SAMPLING=True
export STABLE_ROLLOUT_SEED_REQUESTS=True
export STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True
export STABLE_ROLLOUT_STABLE_SERVER_ORDER=True
export STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True
export STABLE_ROLLOUT_SEED_BASE="$TRAIN_SEED"
export STABLE_ROLLOUT_OFFSET_BASE=0
export STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True

export USE_REMOVE_PADDING=True
export ATTN_IMPLEMENTATION=flash_attention_2
export ACTOR_OPTIMIZER_OFFLOAD=True
export ACTOR_OPTIMIZER_FOREACH=false
export REF_PARAM_OFFLOAD=True
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=True
export LEARNING_RATE=1e-6
export ENTROPY_COEFF=0
export USE_KL_LOSS=False
export SHOPPINGBENCH_REWARD_MODE=asr_terminal
export LOGGER=console
export RESUME_MODE="${RESUME_MODE:-disable}"

export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/train}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/validation}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
FORMAL_RL_CONFIG=1
TRAIN_SEED=${TRAIN_SEED}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
TOTAL_EPOCHS=${TOTAL_EPOCHS}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
SAVE_FREQ=${SAVE_FREQ}
TEST_FREQ=${TEST_FREQ}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}
CLIP_RATIO=0.2
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
PROBABILITY_RATIO_RANGE=[0.8,1.28]
USE_KL_LOSS=False
USE_KL_IN_REWARD=False
ENTROPY_COEFF=0
LOSS_AGG_MODE=token-mean
PPO_EPOCHS=1
WEIGHT_DECAY=0.01
MAX_ACTOR_CKPT_TO_KEEP=4
CHECKPOINT_CONTENTS=model,extra
EOF
  exec env DRY_RUN=1 bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh
fi

exec bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh \
  +data.seed="$TRAIN_SEED" \
  data.validation_shuffle=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  trainer.max_actor_ckpt_to_keep=4 \
  trainer.resume_mode="$RESUME_MODE" \
  "$@"
