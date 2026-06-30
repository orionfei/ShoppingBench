#!/bin/bash
set -euo pipefail

CHECKPOINT_ROOT="${1:-${CHECKPOINT_ROOT:-checkpoints/shoppingbench-sft/qwen3-4b_state_folded_4xa800_full_sft_lr1e-5_micro2_20260628_1139}}"
QUERY_VAL_FILES="${QUERY_VAL_FILES:-dataset/probe/qwen3_4b_voucher_probe_20260629.parquet}"
OUT_ROOT="${OUT_ROOT:-rollouts/qwen3_4b_voucher_sft_probe_20260629}"
REPORT_ROOT="${REPORT_ROOT:-reports/qwen3_4b_voucher_sft_probe_20260629}"
ROLLOUT_N="${ROLLOUT_N:-4}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-qwen3_4b_voucher_sft_probe}"

if [ ! -d "$CHECKPOINT_ROOT" ]; then
  echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
  exit 1
fi
if [ ! -f "$QUERY_VAL_FILES" ]; then
  echo "Probe parquet not found: $QUERY_VAL_FILES" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

for ckpt in "$CHECKPOINT_ROOT"/global_step_*; do
  [ -d "$ckpt" ] || continue
  step="$(basename "$ckpt")"
  export MODEL_PATH="$ckpt"
  export TRAIN_FILES="$QUERY_VAL_FILES"
  export VAL_FILES="$QUERY_VAL_FILES"
  export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${step}"
  export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-rl}"
  export VAL_ONLY=True
  export VAL_BEFORE_TRAIN=True
  export TOTAL_EPOCHS=1
  export TOTAL_TRAINING_STEPS=1
  export TEST_FREQ=1
  export SAVE_FREQ=100000
  export ROLLOUT_N="$ROLLOUT_N"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
  export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
  export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
  export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
  export VALIDATION_DATA_DIR="$OUT_ROOT/$step"
  export ROLLOUT_DATA_DIR="$OUT_ROOT/$step/train_unused"

  echo "[probe] checkpoint=$step val=$QUERY_VAL_FILES out=$VALIDATION_DATA_DIR n=$ROLLOUT_N"
  bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh "$ckpt" 2>&1 | tee "$REPORT_ROOT/$step.log"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    continue
  fi
  python scripts/analyze_verl_query_rollouts.py "$VALIDATION_DATA_DIR" --group-size "$ROLLOUT_N" --output "$REPORT_ROOT/$step.json"
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

python scripts/analyze_verl_query_rollouts.py "$OUT_ROOT"/global_step_* \
  --group-size "$ROLLOUT_N" \
  --selection-summary \
  --output "$REPORT_ROOT/summary.json"
