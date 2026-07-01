#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-shoppingbench-sft-verl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-model/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_sft_state_folded_hybrid_action_schemav3/train.parquet}"
VAL_FILES="${VAL_FILES:-dataset/shoppingbench_sft_state_folded_hybrid_action_schemav3/test.parquet}"

PROJECT_NAME="${PROJECT_NAME:-shoppingbench-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_state_folded_verl}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_LENGTH="${MAX_LENGTH:-9216}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-128}"
TEST_FREQ="${TEST_FREQ:-64}"
LEARNING_RATE="${LEARNING_RATE:-7e-6}"
LOGGER="${LOGGER:-console}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-True}"
WARMUP_STEPS_RATIO="${WARMUP_STEPS_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/shoppingbench-verl/bin/torchrun ]]; then
    TORCHRUN_BIN="/root/miniconda3/envs/shoppingbench-verl/bin/torchrun"
  else
    TORCHRUN_BIN="torchrun"
  fi
fi

"$TORCHRUN_BIN" --standalone --nnodes="$NNODES" --nproc_per_node="$NGPUS_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=right \
  data.num_workers="$DATA_NUM_WORKERS" \
  data.pin_memory="$PIN_MEMORY" \
  data.multiturn.enable=True \
  data.multiturn.messages_key=messages \
  data.multiturn.enable_thinking_key=enable_thinking \
  model.partial_pretrain="$MODEL_PATH" \
  model.trust_remote_code=True \
  model.fsdp_config.model_dtype="$MODEL_DTYPE" \
  model.strategy="$FSDP_STRATEGY" \
  model.enable_gradient_checkpointing=True \
  model.attn_implementation="$ATTN_IMPLEMENTATION" \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules="$TARGET_MODULES" \
  use_remove_padding="$USE_REMOVE_PADDING" \
  ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
  optim.lr="$LEARNING_RATE" \
  optim.warmup_steps_ratio="$WARMUP_STEPS_RATIO" \
  optim.weight_decay="$WEIGHT_DECAY" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger="[$LOGGER]" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  "$@"
