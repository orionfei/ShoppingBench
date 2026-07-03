#!/bin/bash
set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
ROLLOUT_N_VALUE="${ROLLOUT_N_VALUE:-4}"
MAX_RESPONSE_LENGTH_VALUE="${MAX_RESPONSE_LENGTH_VALUE:-8192}"
RUN_ID="${RUN_ID:-qwen3_4b_hybrid_findbatch_latestprompt_stage1_ckpt_t06_n4_probe16_8192_2gpu_20260702}"
SOURCE_ROOT="${SOURCE_ROOT:-checkpoints/shoppingbench-sft/qwen3-4b_hybrid_findbatch_latestprompt_4epoch_2xa800_sft_lr7e-6_micro2_20260701}"
QUERY_VAL_FILES="${QUERY_VAL_FILES:-dataset/probe/qwen3_4b_voucher_probe_latestprompt_20260702.parquet}"
SELECTED_STEPS="${SELECTED_STEPS:-416 624 728 832}"
OUT_ROOT="${OUT_ROOT:-rollouts/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-reports/${RUN_ID}}"
SELECTED_ROOT="${SELECTED_ROOT:-${REPORT_ROOT}/selected_checkpoints}"

mkdir -p "$OUT_ROOT" "$REPORT_ROOT" "$SELECTED_ROOT" logs

if ! curl --noproxy "127.0.0.1,localhost" -fsS "${SEARCH_SERVER_URL:-http://127.0.0.1:5631/}" >/dev/null; then
  echo "[ERROR] ShoppingBench search server is not reachable at ${SEARCH_SERVER_URL:-http://127.0.0.1:5631/}" >&2
  exit 1
fi

for step in $SELECTED_STEPS; do
  ckpt="${SOURCE_ROOT}/global_step_${step}"
  if [ ! -d "$ckpt" ]; then
    echo "[ERROR] Missing checkpoint: $ckpt" >&2
    exit 1
  fi
  ln -sfn "$(realpath "$ckpt")" "${SELECTED_ROOT}/global_step_${step}"
done

cat > "${REPORT_ROOT}/meta.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "source_root": "${SOURCE_ROOT}",
  "selected_root": "${SELECTED_ROOT}",
  "selected_steps": "${SELECTED_STEPS}",
  "query_val_files": "${QUERY_VAL_FILES}",
  "temperature": ${TEMPERATURE},
  "top_p": ${TOP_P},
  "rollout_n": ${ROLLOUT_N_VALUE},
  "max_response_length": ${MAX_RESPONSE_LENGTH_VALUE},
  "cuda_visible_devices": "0,1",
  "ngpus_per_node": 2,
  "rollout_name": "vllm",
  "rollout_mode": "async",
  "attn_implementation": "flash_attention_2",
  "rollout_tensor_model_parallel_size": 1,
  "rollout_max_num_seqs": 16,
  "rollout_gpu_memory_utilization": 0.60
}
EOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export QUERY_VAL_FILES
export OUT_ROOT
export REPORT_ROOT
export ROLLOUT_N="$ROLLOUT_N_VALUE"
export ROLLOUT_TEMPERATURE="$TEMPERATURE"
export ROLLOUT_TOP_P="$TOP_P"
export MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH_VALUE"
export MAX_PROMPT_LENGTH=2048
export PPO_MAX_TOKEN_LEN_PER_GPU=32768
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export EXPERIMENT_PREFIX="${RUN_ID}"
export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=1
export TEST_FREQ=1
export SAVE_FREQ=100000
export LOGGER=console
export REQUIRE_SEARCH_SERVER=1

bash src/rl/evaluate_sft_checkpoints_query_variance_qwen3_4b_a800.sh "$SELECTED_ROOT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

/root/miniconda3/envs/shoppingbench-verl/bin/python scripts/plot_sft_stage1_probe.py \
  "${REPORT_ROOT}/summary.json" \
  --output-dir "${REPORT_ROOT}/figures"
