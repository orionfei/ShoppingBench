#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
WORKERS="${1:?usage: run_rl_v3_worker_smoke.sh WORKERS}"
RUN_ID="rl_v3_worker${WORKERS}_smoke8"
RUN_ROOT="rollouts/rl_v3_worker_smoke/${RUN_ID}"
REPORT_ROOT="reports/rl_v3_worker_smoke/${RUN_ID}"
mkdir -p "$RUN_ROOT" "$REPORT_ROOT"
start=$(date +%s)

MODEL_PATH=checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108 \
PROJECT_NAME=shoppingbench-rl-v3-smoke EXPERIMENT_NAME="$RUN_ID" \
TRAIN_FILES=dataset/probe/rl_v3_worker_smoke8/probe.parquet \
VAL_FILES=dataset/probe/rl_v3_worker_smoke8/probe.parquet \
SHOPPINGBENCH_PRODUCT_CACHE=dataset/shoppingbench_query_rl_v3/product_cache.json \
NGPUS_PER_NODE=4 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=10240 PPO_MAX_TOKEN_LEN_PER_GPU=12288 \
ROLLOUT_N=8 TRAIN_ROLLOUT_TEMPERATURE=0.4 TRAIN_ROLLOUT_TOP_P=0.95 \
VAL_ROLLOUT_TEMPERATURE=0.4 VAL_ROLLOUT_TOP_P=0.95 \
MAX_ASSISTANT_TURNS=15 MAX_USER_TURNS=15 \
ROLLOUT_MAX_MODEL_LEN=12288 ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288 \
ROLLOUT_MAX_NUM_SEQS=8 ROLLOUT_APPLY_MAX_NUM_SEQS=True \
ROLLOUT_ENABLE_PREFIX_CACHING=True ROLLOUT_ENFORCE_EAGER=False ROLLOUT_FREE_CACHE_ENGINE=False \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.55 ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 \
ROLLOUT_AGENT_NUM_WORKERS="$WORKERS" \
STABLE_ROLLOUT_SAMPLING=True STABLE_ROLLOUT_SEED_REQUESTS=True \
STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True STABLE_ROLLOUT_STABLE_SERVER_ORDER=True \
STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True STABLE_ROLLOUT_SEED_BASE=108 \
STABLE_ROLLOUT_OFFSET_BASE=0 STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True \
SHOPPINGBENCH_REWARD_MODE=asr_terminal \
VAL_ONLY=True VAL_BEFORE_TRAIN=True TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=1 SAVE_FREQ=100000 LOGGER=console REQUIRE_SEARCH_SERVER=1 \
VALIDATION_DATA_DIR="$RUN_ROOT" ROLLOUT_DATA_DIR="$RUN_ROOT/train_unused" \
bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh 2>&1 | tee "$REPORT_ROOT/run.log"

elapsed=$(($(date +%s)-start))
python scripts/analyze_outcome_sampling_sweep.py "$RUN_ROOT" \
  --output "$REPORT_ROOT/analysis.json" --group-size 8 --temperature 0.4 --top-p 0.95 --seed 108
python - "$REPORT_ROOT/analysis.json" "$REPORT_ROOT/summary.json" "$WORKERS" "$elapsed" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
summary={"workers":int(sys.argv[3]),"elapsed_seconds":int(sys.argv[4]),"analysis":report}
open(sys.argv[2],"w").write(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"workers":summary["workers"],"elapsed_seconds":summary["elapsed_seconds"]}))
PY
