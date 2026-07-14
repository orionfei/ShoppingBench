#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_SEQS="${MAX_SEQS:-16}"
case "$MAX_SEQS" in
  8)
    DEFAULT_GPU_UTILIZATION=0.55
    DEFAULT_BATCHED_TOKENS=12288
    ;;
  16)
    DEFAULT_GPU_UTILIZATION=0.70
    DEFAULT_BATCHED_TOKENS=32768
    ;;
  32)
    DEFAULT_GPU_UTILIZATION=0.80
    DEFAULT_BATCHED_TOKENS=65536
    ;;
  *)
    echo "MAX_SEQS must be one of: 8, 16, 32" >&2
    exit 2
    ;;
esac

STAMP="$(date -u +%Y%m%d_%H%M%S)"
export ROLLOUT_MAX_NUM_SEQS="$MAX_SEQS"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$DEFAULT_BATCHED_TOKENS}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_UTILIZATION}"
export ROLLOUT_AGENT_NUM_WORKERS=64
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo_v3_throughput_maxseq${MAX_SEQS}_${STAMP}}"
export ROLLOUT_DATA_DIR="rollouts/${EXPERIMENT_NAME}/train"
export VALIDATION_DATA_DIR="rollouts/${EXPERIMENT_NAME}/validation"

echo "[throughput-probe] max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"

exec bash scripts/run_step108_outcome_gspo_v3_dapo_fast64.sh \
  trainer.val_only=True \
  trainer.val_before_train=True \
  trainer.resume_mode=disable
