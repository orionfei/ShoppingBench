#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/formal_eval_${RUN_ID}}"
CONFIG_DIR="$LOG_DIR/configs"
REPORT_DIR="$ROOT_DIR/reports/formal_eval_${RUN_ID}"
ROLLOUT_DIR="$ROOT_DIR/data/formal_eval_${RUN_ID}"

SHOPPINGBENCH_PYTHON="${SHOPPINGBENCH_PYTHON:-/root/miniconda3/envs/shoppingbench/bin/python}"
VERL_PYTHON="${VERL_PYTHON:-/root/miniconda3/envs/shoppingbench-verl/bin/python}"
JAVA_HOME="${JAVA_HOME:-/root/.local/jdks/temurin-21}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/shoppingbench-sft/qwen3-4b_state_folded_4xa800_full_sft_lr1e-5_micro2_20260628_1139/global_step_256}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-sft-step256}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-30000}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8704}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"

INDEX_DIR="${INDEX_DIR:-indexes}"
DOCUMENTS_FILE="${DOCUMENTS_FILE:-resources/documents.jsonl}"
DOCUMENTS_GZIP_FILE="${DOCUMENTS_GZIP_FILE:-resources/documents.jsonl.gz}"
SEARCH_PORT="${SEARCH_PORT:-5631}"
SEARCH_CAPACITY="${SEARCH_CAPACITY:-500}"

THREADS="${THREADS:-8}"
WEB_THREADS="${WEB_THREADS:-4}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-21600s}"
MAX_ROLLOUT_ATTEMPTS="${MAX_ROLLOUT_ATTEMPTS:-2}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
INCLUDE_WEB="${INCLUDE_WEB:-auto}"
BUILD_INDEX_IF_MISSING="${BUILD_INDEX_IF_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"
TEST_DATA_DIR="${TEST_DATA_DIR:-}"

TASKS="${TASKS:-voucher}"
PRODUCT_TEMPLATE="${PRODUCT_TEMPLATE:-config/rollout/sft_qwen3-4b_step256_product_test_state_folded.json}"
SHOP_TEMPLATE="${SHOP_TEMPLATE:-config/rollout/sft_qwen3-4b_step256_shop_test_state_folded.json}"
VOUCHER_TEMPLATE="${VOUCHER_TEMPLATE:-config/rollout/sft_qwen3-4b_step256_voucher_test_state_folded.json}"
WEB_TEMPLATE="${WEB_TEMPLATE:-config/simpleqa_rollout/sft_qwen3-4b_step256_web_simpleqa_test_state_folded.json}"

SEARCH_PID=""
VLLM_PID=""

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

cleanup() {
  if [ "$KEEP_SERVERS" = "1" ]; then
    return
  fi
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    log "stopping vLLM pid=${VLLM_PID}"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$SEARCH_PID" ] && kill -0 "$SEARCH_PID" >/dev/null 2>&1; then
    log "stopping search server pid=${SEARCH_PID}"
    kill "$SEARCH_PID" >/dev/null 2>&1 || true
    wait "$SEARCH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_file() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
}

wait_http() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-600}"
  local start
  start="$(date +%s)"
  while true; do
    if curl --noproxy "127.0.0.1,localhost" -fsS "$url" >/dev/null 2>&1; then
      log "${name} is ready: ${url}"
      return
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout_s" ]; then
      echo "[ERROR] timed out waiting for ${name}: ${url}" >&2
      exit 1
    fi
    sleep 5
  done
}

count_rows() {
  local file="$1"
  if [ -f "$file" ]; then
    wc -l < "$file"
  else
    echo 0
  fi
}

target_rows() {
  local config="$1"
  "$SHOPPINGBENCH_PYTHON" - "$config" <<'PY'
import json
import os
import sys
from pathlib import Path

config = json.load(open(sys.argv[1], encoding="utf-8"))
path = Path(config["synthesize_file"])
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
}

make_runtime_config() {
  local template="$1"
  local task="$2"
  local threads="$3"
  local output="$4"
  "$SHOPPINGBENCH_PYTHON" - "$template" "$task" "$threads" "$output" "$SERVED_MODEL_NAME" "$VLLM_PORT" "$ROLLOUT_DIR" "$TEST_DATA_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

template, task, threads, output, model, port, rollout_dir, test_data_dir = sys.argv[1:9]
config = json.load(open(template, encoding="utf-8"))
config["threads"] = int(threads)
config["base_url"] = f"http://127.0.0.1:{port}/v1"
config["api_key"] = "EMPTY"
config["history_compression"] = "state_folded"
config["stop_after_recommend"] = True
config.setdefault("state_max_candidates_per_search", 10)
config.setdefault("state_max_searches", 12)
config.setdefault("state_max_budget_candidates", 120)
config.setdefault("state_max_viewed_products", 40)
config["state_never_expand"] = False
config["state_min_char_saving"] = 0.0
config["model_config"]["model"] = model
config["model_config"]["temperature"] = 0
config["model_config"]["max_tokens"] = int(os.environ.get("EVAL_MAX_TOKENS", "384"))
config["model_config"]["stream"] = True
config["model_config"]["extra_body"] = {"enable_thinking": False}
if test_data_dir:
    config["synthesize_file"] = str(Path(test_data_dir) / Path(config["synthesize_file"]).name)
rollout_name = f"rollout_{task}_{model}.jsonl".replace("/", "_")
config["rollout_file"] = str(Path(rollout_dir) / rollout_name)
Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, "w", encoding="utf-8") as fout:
    json.dump(config, fout, ensure_ascii=False, indent=4)
    fout.write("\n")
print(config["rollout_file"])
PY
}

build_index_if_needed() {
  if [ -d "$INDEX_DIR" ] && [ -n "$(find "$INDEX_DIR" -maxdepth 1 -type f | head -n 1)" ]; then
    log "using existing index: ${INDEX_DIR}"
    return
  fi
  if [ "$BUILD_INDEX_IF_MISSING" != "1" ]; then
    echo "[ERROR] index is missing and BUILD_INDEX_IF_MISSING=${BUILD_INDEX_IF_MISSING}: ${INDEX_DIR}" >&2
    exit 1
  fi
  prepare_documents_file
  require_file "$DOCUMENTS_FILE"
  log "building Lucene index: documents=${DOCUMENTS_FILE}, index=${INDEX_DIR}"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" \
    DOCUMENTS_FILE="$DOCUMENTS_FILE" INDEX_DIR="$INDEX_DIR" \
    bash build_index.sh 2>&1 | tee "$LOG_DIR/build_index.log"
}

prepare_documents_file() {
  if [ -f "$DOCUMENTS_FILE" ]; then
    log "using existing documents file: ${DOCUMENTS_FILE}"
    return
  fi
  if [ ! -f "$DOCUMENTS_GZIP_FILE" ]; then
    echo "[ERROR] missing documents file: ${DOCUMENTS_FILE}" >&2
    echo "[ERROR] also missing compressed official corpus: ${DOCUMENTS_GZIP_FILE}" >&2
    exit 1
  fi
  log "validating official compressed corpus: ${DOCUMENTS_GZIP_FILE}"
  "$SHOPPINGBENCH_PYTHON" scripts/check_official_corpus.py \
    --target "$DOCUMENTS_GZIP_FILE" \
    --output-report "$REPORT_DIR/corpus_status.json" \
    > "$LOG_DIR/corpus_status.log"
  log "decompressing ${DOCUMENTS_GZIP_FILE} -> ${DOCUMENTS_FILE}"
  mkdir -p "$(dirname "$DOCUMENTS_FILE")"
  gzip -dc "$DOCUMENTS_GZIP_FILE" > "${DOCUMENTS_FILE}.tmp"
  mv "${DOCUMENTS_FILE}.tmp" "$DOCUMENTS_FILE"
}

start_search_if_needed() {
  if curl --noproxy "127.0.0.1,localhost" -fsS "http://127.0.0.1:${SEARCH_PORT}/" >/dev/null 2>&1; then
    log "search server already reachable on port ${SEARCH_PORT}"
    return
  fi
  log "starting search server on port ${SEARCH_PORT}, index=${INDEX_DIR}"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" INDEX_DIR="$INDEX_DIR" \
    SHOPPINGBENCH_SEARCH_CAPACITY="$SEARCH_CAPACITY" PORT="$SEARCH_PORT" \
    "$SHOPPINGBENCH_PYTHON" src/search_engine/server.py \
    > "$LOG_DIR/search_server.log" 2>&1 &
  SEARCH_PID="$!"
  wait_http "http://127.0.0.1:${SEARCH_PORT}/" "search server" 300
}

start_vllm_if_needed() {
  if curl --noproxy "127.0.0.1,localhost" -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    log "vLLM already reachable on port ${VLLM_PORT}"
    return
  fi
  require_file "$CHECKPOINT_DIR"
  log "starting vLLM: checkpoint=${CHECKPOINT_DIR}, served_model=${SERVED_MODEL_NAME}"
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" VLLM_USE_V1="${VLLM_USE_V1:-0}" \
    "$VERL_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$CHECKPOINT_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host "$VLLM_HOST" --port "$VLLM_PORT" \
      --dtype bfloat16 --trust-remote-code \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --disable-log-requests \
      > "$LOG_DIR/vllm.log" 2>&1 &
  VLLM_PID="$!"
  wait_http "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" "vLLM" 900
}

run_config() {
  local task="$1"
  local config="$2"
  local rollout_file
  local rows
  local total
  local attempt

  rollout_file="$("$SHOPPINGBENCH_PYTHON" - "$config" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["rollout_file"])
PY
)"
  total="$(target_rows "$config")"
  if [ "$FORCE_RERUN" = "1" ]; then
    rm -f "$rollout_file"
  fi

  log "${task}: rollout target_rows=${total}, file=${rollout_file}"
  for attempt in $(seq 1 "$MAX_ROLLOUT_ATTEMPTS"); do
    rows="$(count_rows "$rollout_file")"
    if [ "$rows" -ge "$total" ]; then
      break
    fi
    log "${task}: rollout attempt ${attempt}/${MAX_ROLLOUT_ATTEMPTS}, rows=${rows}/${total}"
    if timeout "$ROLLOUT_TIMEOUT" "$SHOPPINGBENCH_PYTHON" src/agent/run_rollout.py "$config" \
      >> "$LOG_DIR/${task}_rollout.log" 2>&1; then
      log "${task}: rollout attempt ${attempt} exited normally"
    else
      log "${task}: rollout attempt ${attempt} exited nonzero; see $LOG_DIR/${task}_rollout.log"
    fi
  done

  rows="$(count_rows "$rollout_file")"
  if [ "$rows" -lt "$total" ] && [ "$ALLOW_PARTIAL" != "1" ]; then
    echo "[ERROR] ${task}: rollout incomplete rows=${rows}/${total}; set ALLOW_PARTIAL=1 to evaluate partial output" >&2
    exit 1
  fi
  if [ "$rows" -eq 0 ]; then
    echo "[ERROR] ${task}: no rollout rows to evaluate" >&2
    exit 1
  fi

  log "${task}: evaluate rows=${rows}/${total}"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" INDEX_DIR="$INDEX_DIR" OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
    "$SHOPPINGBENCH_PYTHON" src/agent/run_evaluate.py "$config" \
    2>&1 | tee "$LOG_DIR/${task}_eval.log"

  log "${task}: reward variance"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" \
    "$SHOPPINGBENCH_PYTHON" scripts/analyze_rollout_rewards.py "$config" \
      --index-dir "$INDEX_DIR" \
      --output-json "$REPORT_DIR/${task}_reward_report.json" \
    2>&1 | tee "$LOG_DIR/${task}_reward_report.log"
}

main() {
  mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$REPORT_DIR" "$ROLLOUT_DIR"
  export JAVA_HOME
  export PATH="$JAVA_HOME/bin:$PATH"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export PYTHONUNBUFFERED=1

  log "run_id=${RUN_ID}"
  log "checkpoint=${CHECKPOINT_DIR}"
  log "served_model=${SERVED_MODEL_NAME}"
  log "index_dir=${INDEX_DIR}"
  log "documents_file=${DOCUMENTS_FILE}"
  log "documents_gzip_file=${DOCUMENTS_GZIP_FILE}"
  log "test_data_dir=${TEST_DATA_DIR:-<default full test>}"
  log "log_dir=${LOG_DIR}"
  log "report_dir=${REPORT_DIR}"

  for task in $TASKS; do
    case "$task" in
      product) require_file "$PRODUCT_TEMPLATE" ;;
      shop) require_file "$SHOP_TEMPLATE" ;;
      voucher) require_file "$VOUCHER_TEMPLATE" ;;
      web) require_file "$WEB_TEMPLATE" ;;
      *) echo "[ERROR] unsupported TASKS item: $task" >&2; exit 1 ;;
    esac
  done

  product_config="$CONFIG_DIR/product.json"
  shop_config="$CONFIG_DIR/shop.json"
  voucher_config="$CONFIG_DIR/voucher.json"
  web_config="$CONFIG_DIR/web.json"
  for task in $TASKS; do
    case "$task" in
      product) make_runtime_config "$PRODUCT_TEMPLATE" product "$THREADS" "$product_config" >/dev/null ;;
      shop) make_runtime_config "$SHOP_TEMPLATE" shop "$THREADS" "$shop_config" >/dev/null ;;
      voucher) make_runtime_config "$VOUCHER_TEMPLATE" voucher "$THREADS" "$voucher_config" >/dev/null ;;
      web) make_runtime_config "$WEB_TEMPLATE" web "$WEB_THREADS" "$web_config" >/dev/null ;;
    esac
  done

  if [ "$DRY_RUN" = "1" ]; then
    log "dry run only; generated runtime configs:"
    for task in $TASKS; do
      case "$task" in
        product) printf '  %s\n' "$product_config" ;;
        shop) printf '  %s\n' "$shop_config" ;;
        voucher) printf '  %s\n' "$voucher_config" ;;
        web) printf '  %s\n' "$web_config" ;;
      esac
    done
    log "dry run finished without starting search/vLLM"
    return
  fi

  build_index_if_needed
  start_search_if_needed
  start_vllm_if_needed

  for task in $TASKS; do
    case "$task" in
      product) run_config product "$product_config" ;;
      shop) run_config shop "$shop_config" ;;
      voucher) run_config voucher "$voucher_config" ;;
      web)
        if [ "$INCLUDE_WEB" = "1" ] || { [ "$INCLUDE_WEB" = "auto" ] && [ -n "${SERPER_KEY:-}" ]; }; then
          run_config web "$web_config"
        else
          log "skipping web/simpleqa; set INCLUDE_WEB=1 and SERPER_KEY to run it"
        fi
        ;;
    esac
  done

  "$SHOPPINGBENCH_PYTHON" scripts/aggregate_rollout_reward_reports.py "$REPORT_DIR" \
    --output-json "$REPORT_DIR/summary.json" \
    2>&1 | tee "$LOG_DIR/summary.log"
  log "finished"
}

main "$@"
