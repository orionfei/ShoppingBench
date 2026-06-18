#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-shoppingbench-sft-verl}"

MODEL_PATH="${MODEL_PATH:-model/Qwen3-1.7B}"
TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_sft_state_folded/train.parquet}"
VAL_FILES="${VAL_FILES:-dataset/shoppingbench_sft_state_folded/test.parquet}"

PROJECT_NAME="${PROJECT_NAME:-shoppingbench-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-1.7b_state_folded_verl}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-20480}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:--1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LOGGER="${LOGGER:-console}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"

torchrun --standalone --nnodes="$NNODES" --nproc_per_node="$NGPUS_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=right \
  data.multiturn.enable=True \
  data.multiturn.messages_key=messages \
  data.multiturn.enable_thinking_key=enable_thinking \
  model.partial_pretrain="$MODEL_PATH" \
  model.trust_remote_code=True \
  model.fsdp_config.model_dtype="$MODEL_DTYPE" \
  model.strategy="$FSDP_STRATEGY" \
  model.enable_gradient_checkpointing=True \
  optim.lr="$LEARNING_RATE" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger="[$LOGGER]" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES"
