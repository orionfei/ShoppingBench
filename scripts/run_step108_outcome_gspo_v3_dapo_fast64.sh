#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Full eight-engine scheduling capacity: 8 engines * max_num_seqs=8.
# Batch64 preserves approximately the same accepted-token load per GPU as the
# previous four-GPU batch32 run while halving optimizer-step overhead.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-112}"
export MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-12}"
export SAVE_STEPS="${SAVE_STEPS:-[6,12]}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-[6,12]}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-64}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-step108_outcome_gspo_v3_dapo_fast64_b64_pilot12}"

exec bash scripts/run_step108_outcome_gspo_v3_dapo_pilot.sh "$@"
