#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-checkpoints/shoppingbench-sft/qwen3-4b_state_folded_4xa800_full_sft_lr1e-5_micro2_20260628_1139/global_step_256}"
export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-rl-eval}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_sft_step256_grpo_reward_variance}"
export TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_query/test.parquet}"
export VAL_FILES="${VAL_FILES:-dataset/shoppingbench_query/test.parquet}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
export VAL_ONLY="${VAL_ONLY:-True}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TEST_FREQ="${TEST_FREQ:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/train_unused}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-rollouts/${EXPERIMENT_NAME}/validation}"

"${SCRIPT_DIR}/run_grpo_qwen3_4b_state_folded_a800.sh" "$@"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

"${PYTHON_BIN:-/root/miniconda3/envs/shoppingbench-verl/bin/python}" "${ROOT_DIR}/scripts/reward_variance_report.py" \
  "$VALIDATION_DATA_DIR" \
  --output "${VALIDATION_DATA_DIR}/reward_variance_report.json"
