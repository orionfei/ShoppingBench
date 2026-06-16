#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-${MIMO_MODEL:-bailian/deepseek-v4-flash}}"
MODEL_SLUG="${MODEL_SLUG:-$(printf '%s' "$MODEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g')}"
if [[ "$MODEL" == mimo* ]]; then
  TOKEN_LIMIT_PARAM="${TOKEN_LIMIT_PARAM:-max_completion_tokens}"
else
  TOKEN_LIMIT_PARAM="${TOKEN_LIMIT_PARAM:-max_tokens}"
fi
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
THREADS="${THREADS:-8}"
SEED="${SEED:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-shoppingbench}"
PYTHON_BIN="${PYTHON_BIN:-/data1/yfl_data/miniconda3/envs/${CONDA_ENV}/bin/python}"
JAVA_HOME="${JAVA_HOME:-/data1/yfl_data/miniconda3/envs/${CONDA_ENV}}"
PORT="${PORT:-5631}"
BACKGROUND="${BACKGROUND:-1}"
FIXED_QUERIES="${FIXED_QUERIES:-1}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-7200}"
MAX_ROLLOUT_ATTEMPTS="${MAX_ROLLOUT_ATTEMPTS:-3}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"

LOG_DIR="$ROOT_DIR/logs/run_${RUN_ID}"
PID_FILE="$LOG_DIR/pid"
MAIN_LOG="$LOG_DIR/main.log"

run_main() {
  mkdir -p "$LOG_DIR" data config/rollout

  export JAVA_HOME
  export PATH="$JAVA_HOME/bin:$PATH"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export PYTHONUNBUFFERED=1
  if [[ "$MODEL" == mimo* ]]; then
    export OPENAI_API_KEY="${MIMO_API_KEY:-${OPENAI_API_KEY:-}}"
    export OPENAI_BASE_URL="${MIMO_BASE_URL:-${OPENAI_BASE_URL:-}}"
  else
    export OPENAI_API_KEY="${OPENAI_API_KEY:-${MIMO_API_KEY:-}}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${MIMO_BASE_URL:-}}"
  fi

  echo "[INFO] run_id=${RUN_ID}"
  echo "[INFO] model=${MODEL}"
  echo "[INFO] token_limit_param=${TOKEN_LIMIT_PARAM}"
  echo "[INFO] sample_size=${SAMPLE_SIZE}"
  echo "[INFO] threads=${THREADS}"
  echo "[INFO] fixed_queries=${FIXED_QUERIES}"
  echo "[INFO] rollout_timeout=${ROLLOUT_TIMEOUT}"
  echo "[INFO] max_rollout_attempts=${MAX_ROLLOUT_ATTEMPTS}"
  echo "[INFO] allow_partial=${ALLOW_PARTIAL}"
  echo "[INFO] openai_api_key_set=$([ -n "${OPENAI_API_KEY}" ] && echo 1 || echo 0)"
  echo "[INFO] openai_base_url_set=$([ -n "${OPENAI_BASE_URL}" ] && echo 1 || echo 0)"
  echo "[INFO] log_dir=${LOG_DIR}"

  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Python not found: $PYTHON_BIN" >&2
    exit 1
  fi

  if ! curl --noproxy "127.0.0.1,localhost" -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "[ERROR] search server is not reachable at http://127.0.0.1:${PORT}/" >&2
    echo "[HINT] start it first, for example:" >&2
    echo "  OPENAI_API_KEY=sk-dummy JAVA_HOME=${JAVA_HOME} PATH=${JAVA_HOME}/bin:\$PATH nohup ${PYTHON_BIN} src/search_engine/server.py > server.log 2>&1 &" >&2
    exit 1
  fi

  make_sample "product" "data/synthesize_product_test.jsonl"
  make_sample "shop" "data/synthesize_shop_test.jsonl"
  make_sample "voucher" "data/synthesize_voucher_test.jsonl"

  make_config "product"
  make_config "shop"
  make_config "voucher"

  run_task "product"
  run_task "shop"
  run_task "voucher"

  echo "[INFO] all tasks finished"
  echo "[INFO] rollout files:"
  wc -l \
    "data/${RUN_ID}_rollout_product_${MODEL_SLUG}.jsonl" \
    "data/${RUN_ID}_rollout_shop_${MODEL_SLUG}.jsonl" \
    "data/${RUN_ID}_rollout_voucher_${MODEL_SLUG}.jsonl" || true
  echo "[INFO] eval summaries:"
  grep -H -E 'Model `|gt rate|success rate|format score|recommend product score|rule match score|shop match score|budget match score' "$LOG_DIR"/*_eval.log || true
}

make_sample() {
  local task="$1"
  local source_file="$2"
  local sample_file="data/${RUN_ID}_${task}_test.jsonl"
  local fixed_file="data/fixed60_${task}_test.jsonl"

  if [ ! -f "$source_file" ]; then
    echo "[ERROR] missing source test file: $source_file" >&2
    exit 1
  fi

  if [ "$FIXED_QUERIES" = "1" ]; then
    if [ ! -f "$fixed_file" ]; then
      echo "[ERROR] fixed query file missing: $fixed_file" >&2
      echo "[HINT] set FIXED_QUERIES=0 to sample from $source_file" >&2
      exit 1
    fi
    cp "$fixed_file" "$sample_file"
  elif [ -n "$SEED" ]; then
    "$PYTHON_BIN" - "$source_file" "$sample_file" "$SAMPLE_SIZE" "$SEED" <<'PY'
import random
import sys

source, target, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lines = open(source, encoding="utf-8").read().splitlines()
rng = random.Random(f"{seed}:{target}")
sample = rng.sample(lines, min(n, len(lines)))
with open(target, "w", encoding="utf-8") as fout:
    fout.write("\n".join(sample) + "\n")
PY
  else
    shuf -n "$SAMPLE_SIZE" "$source_file" > "$sample_file"
  fi

  echo "[INFO] sampled $(wc -l < "$sample_file") rows -> $sample_file"
}

make_config() {
  local task="$1"
  local config_file="config/rollout/${RUN_ID}_${task}_${MODEL_SLUG}.json"
  local sample_file="data/${RUN_ID}_${task}_test.jsonl"
  local rollout_file="data/${RUN_ID}_rollout_${task}_${MODEL_SLUG}.jsonl"

  "$PYTHON_BIN" - "$task" "$sample_file" "$rollout_file" "$THREADS" "$MODEL" "$TOKEN_LIMIT_PARAM" "$config_file" <<'PY'
import json
import sys

task, sample_file, rollout_file, threads, model, token_limit_param, config_file = sys.argv[1:8]
config = {
    "task": task,
    "system_prompt_file": "src/agent/prompt/rollout.md",
    "synthesize_file": sample_file,
    "rollout_file": rollout_file,
    "threads": int(threads),
    "model_config": {
        "model": model,
        "temperature": 0,
        token_limit_param: 8192,
    },
}
with open(config_file, "w", encoding="utf-8") as fout:
    json.dump(config, fout, indent=4)
    fout.write("\n")
print(config_file)
PY
}

run_task() {
  local task="$1"
  local config_file="config/rollout/${RUN_ID}_${task}_${MODEL_SLUG}.json"
  local rollout_log="$LOG_DIR/${task}_rollout.log"
  local eval_log="$LOG_DIR/${task}_eval.log"
  local sample_file="data/${RUN_ID}_${task}_test.jsonl"
  local rollout_file="data/${RUN_ID}_rollout_${task}_${MODEL_SLUG}.jsonl"
  local target_rows
  local rows
  local attempt

  target_rows="$(wc -l < "$sample_file")"
  echo "[INFO] ${task}: rollout start, target_rows=${target_rows}"
  for attempt in $(seq 1 "$MAX_ROLLOUT_ATTEMPTS"); do
    rows="$(count_rows "$rollout_file")"
    if [ "$rows" -ge "$target_rows" ]; then
      break
    fi

    echo "[INFO] ${task}: rollout attempt ${attempt}/${MAX_ROLLOUT_ATTEMPTS}, current_rows=${rows}/${target_rows}"
    if timeout "$ROLLOUT_TIMEOUT" "$PYTHON_BIN" src/agent/run_rollout.py "$config_file" >> "$rollout_log" 2>&1; then
      echo "[INFO] ${task}: rollout attempt ${attempt} exited normally"
    else
      status="$?"
      echo "[WARN] ${task}: rollout attempt ${attempt} exited with status ${status}; see ${rollout_log}"
    fi
  done

  rows="$(count_rows "$rollout_file")"
  echo "[INFO] ${task}: rollout done, rows=${rows}/${target_rows}"
  if [ "$rows" -lt "$target_rows" ]; then
    echo "[WARN] ${task}: rollout incomplete after ${MAX_ROLLOUT_ATTEMPTS} attempt(s)"
    if [ "$ALLOW_PARTIAL" != "1" ]; then
      echo "[ERROR] ${task}: ALLOW_PARTIAL=${ALLOW_PARTIAL}, stopping before eval" >&2
      exit 1
    fi
    if [ "$rows" -eq 0 ]; then
      echo "[ERROR] ${task}: no rollout rows available for eval" >&2
      exit 1
    fi
    echo "[WARN] ${task}: evaluating partial rollout ${rows}/${target_rows}"
  fi

  echo "[INFO] ${task}: eval start"
  "$PYTHON_BIN" src/agent/run_evaluate.py "$config_file" > "$eval_log" 2>&1
  echo "[INFO] ${task}: eval done"
  tail -n 20 "$eval_log"
}

count_rows() {
  local file="$1"
  if [ -f "$file" ]; then
    wc -l < "$file"
  else
    echo 0
  fi
}

if [ "$BACKGROUND" = "1" ] && [ "${1:-}" != "--foreground" ]; then
  mkdir -p "$LOG_DIR"
  (
    BACKGROUND=0 "$0" --foreground
  ) > "$MAIN_LOG" 2>&1 &
  echo "$!" > "$PID_FILE"
  echo "Started background run."
  echo "PID: $(cat "$PID_FILE")"
  echo "Main log: $MAIN_LOG"
  echo "Follow: tail -f '$MAIN_LOG'"
  exit 0
fi

run_main
