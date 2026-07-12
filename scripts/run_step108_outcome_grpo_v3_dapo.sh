#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MODEL_PATH="${MODEL_PATH:-checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108}"
export TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_query_rl_v3/train.parquet}"
export VAL_FILES="${VAL_FILES:-dataset/shoppingbench_query_rl_v3/validation.parquet}"
export SHOPPINGBENCH_PRODUCT_CACHE="${SHOPPINGBENCH_PRODUCT_CACHE:-dataset/shoppingbench_query_rl_v3/product_cache.json}"
export PROJECT_NAME="shoppingbench-rl-v3-dapo"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-step108_outcome_grpo_v3_dapo_b32_2ep}"
export TRAIN_SEED="${TRAIN_SEED:-108}"

export NGPUS_PER_NODE=4
export TRAIN_BATCH_SIZE=32
export GEN_BATCH_SIZE=32
export DYNAMIC_SAMPLING_ENABLE=True
export DYNAMIC_SAMPLING_METRIC=terminal_asr
export MAX_NUM_GEN_BATCHES=4
export PPO_MINI_BATCH_SIZE=16
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export PPO_MAX_TOKEN_LEN_PER_GPU=12288
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-9}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-90}"
export SAVE_FREQ=100000
export TEST_FREQ=100000
export VAL_BATCH_SIZE=16
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_STEPS="${SAVE_STEPS:-[23,45,68,90]}"
VALIDATION_STEPS="${VALIDATION_STEPS:-[11,23,34,45,56,68,79,90]}"

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
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
export ROLLOUT_ENFORCE_EAGER=False
export ROLLOUT_FREE_CACHE_ENGINE=True
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export ACTOR_CHECKPOINT_SAVE_CONTENTS="['model','extra']"
export ACTOR_CHECKPOINT_LOAD_CONTENTS="['model','extra']"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/train}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/validation}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
FORMAL_RL_V3_DAPO=1
TRAIN_ROWS=1414
VAL_ROWS=64
TEST_ROWS=250
NGPUS_PER_NODE=4
EFFECTIVE_TRAIN_BATCH_SIZE=32
GEN_BATCH_SIZE=32
MAX_NUM_GEN_BATCHES=4
GROUP_SIZE=8
TOTAL_EFFECTIVE_STEPS=90
SAVE_STEPS=23,45,68,90
VALIDATION_STEPS=0,11,23,34,45,56,68,79,90
LR=1e-6
RATIO_RANGE=[0.8,1.28]
KL_LOSS=False
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
  trainer.max_actor_ckpt_to_keep=2 \
  +trainer.save_steps="$SAVE_STEPS" \
  +trainer.test_steps="$VALIDATION_STEPS" \
  trainer.resume_mode=disable \
  "$@"
