#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/shoppingbench/bin/python}"
MODE="${1:-start}"

MODEL="${MODEL:-gpt-5.5-medium}"
WORKERS="${WORKERS:-32}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_CANDIDATES="${MAX_CANDIDATES:-10}"
MAX_FAILED_SEARCHES="${MAX_FAILED_SEARCHES:-5}"
MAX_VIEWED_PRODUCTS="${MAX_VIEWED_PRODUCTS:-20}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"

SOURCE_FILE="${SOURCE_FILE:-data/synthesize_voucher_train.jsonl}"
SAMPLE_FILE="${SAMPLE_FILE:-data/synthesize_voucher_train.jsonl}"
RUN_NAME="${RUN_NAME:-teacher_gpt55medium_train1500_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-data/tmp/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-data/tmp/${RUN_NAME}.pid}"
ENV_FILE="${ENV_FILE:-data/tmp/teacher_gpt55medium_train1500.env}"

ROLLOUT_FILE="${ROLLOUT_FILE:-${RUN_DIR}/rollout.jsonl}"
REPORT_FILE="${REPORT_FILE:-${RUN_DIR}/runner_report.json}"
STAGE_REPORT_FILE="${STAGE_REPORT_FILE:-${RUN_DIR}/stage_reward_report.json}"
PRODUCT_CACHE_FILE="${PRODUCT_CACHE_FILE:-${RUN_DIR}/stage_product_cache.json}"
META_FILE="${META_FILE:-${RUN_DIR}/meta.json}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:5631}"
SEARCH_LOG_FILE="${SEARCH_LOG_FILE:-logs/${RUN_NAME}_search_server.log}"
SEARCH_PID_FILE="${SEARCH_PID_FILE:-data/tmp/${RUN_NAME}_search_server.pid}"

cd "$ROOT"

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
  export REMOTE_BASE_URL="${REMOTE_BASE_URL:-${OPENAI_BASE_URL:-}}"
  export REMOTE_API_KEY="${REMOTE_API_KEY:-${OPENAI_API_KEY:-}}"
  if [[ -z "${REMOTE_BASE_URL}" || -z "${REMOTE_API_KEY}" ]]; then
    echo "Missing REMOTE_BASE_URL or REMOTE_API_KEY. Put them in ${ENV_FILE} or export them before running." >&2
    exit 2
  fi
  export OPENAI_BASE_URL="$REMOTE_BASE_URL"
  export OPENAI_API_KEY="$REMOTE_API_KEY"
  export NO_PROXY="35.220.164.252,127.0.0.1,localhost,${NO_PROXY:-}"
  export no_proxy="$NO_PROXY"
}

probe_search() {
  curl --noproxy "127.0.0.1,localhost" -fsS "${SEARCH_URL}/find_product?q=test&page=1" >/dev/null
}

ensure_search_server() {
  if probe_search; then
    echo "Search server is already reachable at ${SEARCH_URL}."
    return
  fi

  mkdir -p "$(dirname "$SEARCH_LOG_FILE")" "$(dirname "$SEARCH_PID_FILE")"
  echo "Starting local search server at ${SEARCH_URL}."
  HOST=127.0.0.1 PORT=5631 nohup "$PYTHON" src/search_engine/server.py >"$SEARCH_LOG_FILE" 2>&1 &
  echo "$!" >"$SEARCH_PID_FILE"

  for _ in $(seq 1 90); do
    if probe_search; then
      echo "Search server started; pid=$(cat "$SEARCH_PID_FILE")."
      return
    fi
    sleep 2
  done

  echo "Search server did not become ready. See ${SEARCH_LOG_FILE}." >&2
  exit 3
}

write_meta() {
  mkdir -p "$RUN_DIR"
  "$PYTHON" - <<PY
import json
from pathlib import Path

meta = {
    "run_name": "$RUN_NAME",
    "source_file": "$SOURCE_FILE",
    "sample_file": "$SAMPLE_FILE",
    "rollout_file": "$ROLLOUT_FILE",
    "runner_report_file": "$REPORT_FILE",
    "stage_report_file": "$STAGE_REPORT_FILE",
    "product_cache_file": "$PRODUCT_CACHE_FILE",
    "model": "$MODEL",
    "workers": int("$WORKERS"),
    "max_steps": int("$MAX_STEPS"),
    "max_candidates": int("$MAX_CANDIDATES"),
    "max_failed_searches": int("$MAX_FAILED_SEARCHES"),
    "max_viewed_products": int("$MAX_VIEWED_PRODUCTS"),
    "harness": "state_local_retry2final_known_best",
}
Path("$META_FILE").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY
}

run_rollout() {
  load_env
  ensure_search_server
  write_meta

  echo "Starting teacher rollout ${RUN_NAME} at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  "$PYTHON" scripts/run_state_local_api_rollout.py \
    --source "$SOURCE_FILE" \
    --sample-file "$SAMPLE_FILE" \
    --rollout-file "$ROLLOUT_FILE" \
    --report-file "$REPORT_FILE" \
    --model "$MODEL" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-steps "$MAX_STEPS" \
    --max-candidates "$MAX_CANDIDATES" \
    --max-failed-searches "$MAX_FAILED_SEARCHES" \
    --max-viewed-products "$MAX_VIEWED_PRODUCTS" \
    --workers "$WORKERS"

  echo "Rollout complete. Running stage reward audit."
  "$PYTHON" scripts/analyze_state_local_stage_rewards.py \
    --sample-file "$SAMPLE_FILE" \
    --rollout-file "$ROLLOUT_FILE" \
    --output-json "$STAGE_REPORT_FILE" \
    --product-cache-out "$PRODUCT_CACHE_FILE"
  echo "Teacher rollout ${RUN_NAME} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
}

start_background() {
  mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")" "$RUN_DIR"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running: pid=$(cat "$PID_FILE"), log=${LOG_FILE}"
    exit 0
  fi

  setsid env \
    PYTHON="$PYTHON" \
    MODEL="$MODEL" \
    WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" \
    MAX_CANDIDATES="$MAX_CANDIDATES" \
    MAX_FAILED_SEARCHES="$MAX_FAILED_SEARCHES" \
    MAX_VIEWED_PRODUCTS="$MAX_VIEWED_PRODUCTS" \
    MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    SOURCE_FILE="$SOURCE_FILE" \
    SAMPLE_FILE="$SAMPLE_FILE" \
    RUN_NAME="$RUN_NAME" \
    RUN_DIR="$RUN_DIR" \
    LOG_FILE="$LOG_FILE" \
    PID_FILE="$PID_FILE" \
    ENV_FILE="$ENV_FILE" \
    ROLLOUT_FILE="$ROLLOUT_FILE" \
    REPORT_FILE="$REPORT_FILE" \
    STAGE_REPORT_FILE="$STAGE_REPORT_FILE" \
    PRODUCT_CACHE_FILE="$PRODUCT_CACHE_FILE" \
    META_FILE="$META_FILE" \
    SEARCH_URL="$SEARCH_URL" \
    SEARCH_LOG_FILE="$SEARCH_LOG_FILE" \
    SEARCH_PID_FILE="$SEARCH_PID_FILE" \
    "$SCRIPT_PATH" run >"$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" >"$PID_FILE"
  echo "Started ${RUN_NAME}"
  echo "PID: $(cat "$PID_FILE")"
  echo "Log: ${LOG_FILE}"
  echo "Run dir: ${RUN_DIR}"
}

show_status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "RUNNING pid=$(cat "$PID_FILE")"
  elif [[ -f "$PID_FILE" ]]; then
    echo "NOT RUNNING stale_pid=$(cat "$PID_FILE")"
  else
    echo "NO PID FILE ${PID_FILE}"
  fi
  [[ -f "$META_FILE" ]] && echo "Meta: ${META_FILE}"
  [[ -f "$REPORT_FILE" ]] && echo "Runner report: ${REPORT_FILE}"
  [[ -f "$STAGE_REPORT_FILE" ]] && echo "Stage report: ${STAGE_REPORT_FILE}"
  if [[ -f "$ROLLOUT_FILE" ]]; then
    echo "Rollout lines: $(wc -l < "$ROLLOUT_FILE") / $(wc -l < "$SAMPLE_FILE")"
  fi
  [[ -f "$LOG_FILE" ]] && tail -n 40 "$LOG_FILE"
}

case "$MODE" in
  start)
    start_background
    ;;
  run)
    run_rollout
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 [start|run|status]" >&2
    exit 1
    ;;
esac
