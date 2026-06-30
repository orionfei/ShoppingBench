#!/bin/bash
set -euo pipefail

CHECKPOINT_ROOT="${1:-${CHECKPOINT_ROOT:-checkpoints/shoppingbench-sft/qwen3-4b_state_folded_verl}}"
OUT_ROOT="${OUT_ROOT:-rollouts/sft_checkpoint_reward_variance}"
QUERY_VAL_FILES="${QUERY_VAL_FILES:-dataset/shoppingbench_query/test.parquet}"
ROLLOUT_N="${ROLLOUT_N:-8}"

if [ ! -d "$CHECKPOINT_ROOT" ]; then
  echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

for ckpt in "$CHECKPOINT_ROOT"/global_step_*; do
  [ -d "$ckpt" ] || continue
  step="$(basename "$ckpt")"
  export MODEL_PATH="$ckpt"
  export TRAIN_FILES="$QUERY_VAL_FILES"
  export VAL_FILES="$QUERY_VAL_FILES"
  export EXPERIMENT_NAME="variance_${step}"
  export VAL_ONLY=True
  export VAL_BEFORE_TRAIN=True
  export TOTAL_EPOCHS=1
  export TOTAL_TRAINING_STEPS=1
  export TEST_FREQ=1
  export ROLLOUT_N="$ROLLOUT_N"
  export VALIDATION_DATA_DIR="$OUT_ROOT/$step"
  export ROLLOUT_DATA_DIR="$OUT_ROOT/$step/train_unused"

  bash src/rl/run_grpo_qwen3_1_7b_query_verl.sh "$ckpt"
  python scripts/reward_variance_report.py "$VALIDATION_DATA_DIR" --output "$OUT_ROOT/$step/report.json"
done

python scripts/reward_variance_report.py "$OUT_ROOT"/global_step_* --output "$OUT_ROOT/summary.json"
