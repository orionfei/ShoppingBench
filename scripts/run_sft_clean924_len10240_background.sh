#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
RUN_ID="${RUN_ID:-sft_clean924_prefix_len10240_full_3ep_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/logs"
REPORT_DIR="$ROOT_DIR/reports/$RUN_ID"
CKPT_DIR="$ROOT_DIR/checkpoints/shoppingbench-sft/$RUN_ID"
PID_FILE="$ROOT_DIR/data/tmp/${RUN_ID}.pid"
ENV_FILE="$ROOT_DIR/data/tmp/${RUN_ID}.env"
LAUNCHER_FILE="$ROOT_DIR/data/tmp/${RUN_ID}.launch.sh"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_sft_state_local_step_clean924_prefix_masked_len10240/train.parquet}"
VAL_FILES="${VAL_FILES:-dataset/shoppingbench_sft_state_local_step_clean924_prefix_masked_len10240/test.parquet}"
MODEL_PATH="${MODEL_PATH:-model/Qwen3-4B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_LENGTH="${MAX_LENGTH:-10240}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-27}"
TEST_FREQ="${TEST_FREQ:-27}"
LEARNING_RATE="${LEARNING_RATE:-7e-6}"

status() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if ps -p "$pid" >/dev/null 2>&1; then
      echo "running pid=$pid"
    else
      echo "not running pid=$pid"
    fi
  else
    echo "pid file not found: $PID_FILE"
  fi
  echo "run_id=$RUN_ID"
  echo "log=$LOG_FILE"
  echo "checkpoint_dir=$CKPT_DIR"
  echo "report_dir=$REPORT_DIR"
  if [[ -f "$LOG_FILE" ]]; then
    grep -E "\\[step [0-9]+\\]|Final validation|OutOfMemory|Traceback|Error executing" "$LOG_FILE" | tail -40 || true
  fi
}

if [[ "$ACTION" == "status" ]]; then
  status
  exit 0
fi

if [[ "$ACTION" != "start" ]]; then
  echo "usage: RUN_ID=<id> $0 [start|status]" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$CKPT_DIR" "$(dirname "$PID_FILE")"

cat >"$ENV_FILE" <<EOF
RUN_ID=$RUN_ID
TRAIN_FILES=$TRAIN_FILES
VAL_FILES=$VAL_FILES
MODEL_PATH=$MODEL_PATH
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
MICRO_BATCH_SIZE_PER_GPU=$MICRO_BATCH_SIZE_PER_GPU
MAX_LENGTH=$MAX_LENGTH
TOTAL_EPOCHS=$TOTAL_EPOCHS
SAVE_FREQ=$SAVE_FREQ
TEST_FREQ=$TEST_FREQ
LEARNING_RATE=$LEARNING_RATE
LOG_FILE=$LOG_FILE
REPORT_DIR=$REPORT_DIR
CKPT_DIR=$CKPT_DIR
EOF

cat >"$LAUNCHER_FILE" <<EOF
#!/bin/bash
set -euo pipefail
cd "$ROOT_DIR"
export TRAIN_FILES="$TRAIN_FILES"
export VAL_FILES="$VAL_FILES"
export MODEL_PATH="$MODEL_PATH"
export PROJECT_NAME="shoppingbench-sft"
export EXPERIMENT_NAME="$RUN_ID"
export DEFAULT_LOCAL_DIR="checkpoints/shoppingbench-sft/$RUN_ID"
export NGPUS_PER_NODE=4
export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
export MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH_SIZE_PER_GPU"
export MAX_LENGTH="$MAX_LENGTH"
export TOTAL_EPOCHS="$TOTAL_EPOCHS"
export TOTAL_TRAINING_STEPS=null
export SAVE_FREQ="$SAVE_FREQ"
export TEST_FREQ="$TEST_FREQ"
export LEARNING_RATE="$LEARNING_RATE"
export LOGGER=console
export MODEL_DTYPE=bf16
export FSDP_STRATEGY=fsdp2
export ATTN_IMPLEMENTATION=flash_attention_2
export USE_REMOVE_PADDING=True
export DATA_NUM_WORKERS=4
export PIN_MEMORY=True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
bash src/rl/run_sft_qwen3_1_7b_verl.sh
/root/miniconda3/envs/shoppingbench-verl/bin/python scripts/plot_sft_training_log.py \\
  --log "$LOG_FILE" \\
  --output-dir "$REPORT_DIR"
EOF
chmod +x "$LAUNCHER_FILE"

setsid bash "$LAUNCHER_FILE" >"$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" >"$PID_FILE"
echo "started run_id=$RUN_ID pid=$pid"
echo "log=$LOG_FILE"
echo "checkpoint_dir=$CKPT_DIR"
echo "report_dir=$REPORT_DIR"
