#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-model/Qwen3-4B}"
export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-sft}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_state_folded_4xa800_full_sft_lr1e-5}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
export MAX_LENGTH="${MAX_LENGTH:-9216}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export SAVE_FREQ="${SAVE_FREQ:-128}"
export TEST_FREQ="${TEST_FREQ:-64}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
export LOGGER="${LOGGER:-console}"

exec "${SCRIPT_DIR}/run_sft_qwen3_1_7b_verl.sh" "$@"
