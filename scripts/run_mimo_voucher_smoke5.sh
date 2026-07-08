#!/usr/bin/env bash
set -Euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"

if [[ -n "${RUN_ID:-}" ]]; then
  ACTIVE_RUN_ID="$RUN_ID"
elif [[ "$ACTION" == "start" || "$ACTION" == "run" ]]; then
  ACTIVE_RUN_ID="mimo_voucher_first50_x4_$(date -u +%Y%m%d_%H%M%S)"
else
  latest_log_dir="$(ls -td logs/mimo_voucher_first50_x4_* logs/mimo_voucher_smoke5_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_log_dir" ]]; then
    echo "[ERROR] no run found under logs/mimo_voucher_first50_x4_*" >&2
    exit 1
  fi
  ACTIVE_RUN_ID="$(basename "$latest_log_dir")"
fi

LOG_DIR="$ROOT_DIR/logs/$ACTIVE_RUN_ID"
PID_FILE="$LOG_DIR/pid"
SEARCH_PID_FILE="$LOG_DIR/search_server.pid"
MAIN_LOG="$LOG_DIR/main.log"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/shoppingbench/bin/python}"
JAVA_HOME="${JAVA_HOME:-/root/miniconda3/envs/shoppingbench}"
PORT="${PORT:-5631}"
SHOPPINGBENCH_SEARCH_CAPACITY="${SHOPPINGBENCH_SEARCH_CAPACITY:-30000}"
NUM_QUERIES="${NUM_QUERIES:-50}"
ROLLOUTS="${ROLLOUTS:-4}"
THREADS="${THREADS:-2}"
MODEL="${MODEL:-}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
DISABLE_ENV_PROXY="${DISABLE_ENV_PROXY:-1}"
LOAD_BASHRC_EXPORTS="${LOAD_BASHRC_EXPORTS:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-600}"
SHOPPINGBENCH_LLM_TIMEOUT="${SHOPPINGBENCH_LLM_TIMEOUT:-180}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-7200}"

SAMPLE_FILE="$ROOT_DIR/data/tmp/${ACTIVE_RUN_ID}.jsonl"
SFT_FILE="$ROOT_DIR/data/tmp/${ACTIVE_RUN_ID}_sft.json"
RS_CONFIG_FILE="$ROOT_DIR/config/tmp/${ACTIVE_RUN_ID}_rs.json"
CHECK_SUMMARY_FILE="$LOG_DIR/strict_check.json"
SUMMARY_FILE="$LOG_DIR/summary.txt"

rollout_file_for_round() {
  printf '%s/data/tmp/%s_rollout_%02d.jsonl' "$ROOT_DIR" "$ACTIVE_RUN_ID" "$1"
}

config_file_for_round() {
  printf '%s/config/tmp/%s_rollout_%02d.json' "$ROOT_DIR" "$ACTIVE_RUN_ID" "$1"
}

load_runtime_env() {
  set +e
  if [[ "$LOAD_BASHRC_EXPORTS" == "1" && -f "$HOME/.bashrc" ]]; then
    while IFS= read -r line; do
      case "$line" in
        "export MIMO_API_KEY="*)
          eval "$line"
          ;;
        "export MIMO_BASE_URL="*)
          eval "$line"
          ;;
        "export MIMO_MODEL="*)
          eval "$line"
          ;;
        "export OPENAI_API_KEY="*)
          eval "$line"
          ;;
        "export OPENAI_BASE_URL="*)
          eval "$line"
          ;;
      esac
    done < "$HOME/.bashrc"
  fi
  set +e

  export JAVA_HOME
  if [[ -z "${JVM_PATH:-}" ]]; then
    if [[ -f "$JAVA_HOME/lib/jvm/lib/server/libjvm.so" ]]; then
      export JVM_PATH="$JAVA_HOME/lib/jvm/lib/server/libjvm.so"
    elif [[ -f "$JAVA_HOME/lib/server/libjvm.so" ]]; then
      export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
    fi
  fi
  export PATH="$JAVA_HOME/bin:$PATH"
  export PYTHONUNBUFFERED=1
  export SHOPPINGBENCH_LLM_TIMEOUT
  if [[ -z "$MODEL" ]]; then
    MODEL="${MIMO_MODEL:-mimo-v2.5-pro}"
  fi
  export MIMO_MODEL="$MODEL"

  if [[ "$DISABLE_ENV_PROXY" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  fi
  echo "[INFO] runtime env loaded: model=${MODEL}, mimo_api_key_set=$([ -n "${MIMO_API_KEY:-}" ] && echo 1 || echo 0), mimo_base_url_set=$([ -n "${MIMO_BASE_URL:-}" ] && echo 1 || echo 0), openai_api_key_set=$([ -n "${OPENAI_API_KEY:-}" ] && echo 1 || echo 0), openai_base_url_set=$([ -n "${OPENAI_BASE_URL:-}" ] && echo 1 || echo 0)"
}

probe_search_server() {
  curl --noproxy "127.0.0.1,localhost" -fsS \
    --max-time 5 \
    "http://127.0.0.1:${PORT}/find_product?q=test&page=1" >/dev/null 2>&1
  SEARCH_PROBE_RC=$?
  return 0
}

ensure_search_server() {
  set +e
  probe_search_server
  if [[ "$SEARCH_PROBE_RC" -eq 0 ]]; then
    echo "[INFO] search server already reachable on port ${PORT}"
    return
  fi

  echo "[INFO] starting search server on port ${PORT}"
  mkdir -p "$LOG_DIR"
  OPENAI_API_KEY=sk-dummy \
    HOST=0.0.0.0 \
    PORT="$PORT" \
    SHOPPINGBENCH_SEARCH_CAPACITY="$SHOPPINGBENCH_SEARCH_CAPACITY" \
    nohup "$PYTHON_BIN" src/search_engine/server.py \
      > "$LOG_DIR/search_server.log" 2>&1 &
  echo "$!" > "$SEARCH_PID_FILE"

  for _ in $(seq 1 "$WAIT_TIMEOUT_SECONDS"); do
    probe_search_server
    if [[ "$SEARCH_PROBE_RC" -eq 0 ]]; then
      echo "[INFO] search server is ready"
      return
    fi
    sleep 1
  done

  echo "[ERROR] search server did not become ready within ${WAIT_TIMEOUT_SECONDS}s" >&2
  tail -n 80 "$LOG_DIR/search_server.log" >&2 || true
  exit 1
}

make_sample() {
  mkdir -p data/tmp config/tmp "$LOG_DIR"
  if [[ -f "$SAMPLE_FILE" ]]; then
    echo "[INFO] reusing existing sample file -> $SAMPLE_FILE ($(wc -l < "$SAMPLE_FILE") rows)"
    return
  fi
  "$PYTHON_BIN" - "$ROOT_DIR/data/synthesize_voucher_train.jsonl" "$SAMPLE_FILE" "$NUM_QUERIES" <<'PY'
import sys

source, target, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source, encoding="utf-8") as fin:
    sample = [line for _, line in zip(range(n), (line for line in fin if line.strip()))]
with open(target, "w", encoding="utf-8") as fout:
    fout.writelines(sample)
print(f"[INFO] copied first {len(sample)} rows -> {target}")
PY
}

make_configs() {
  for round in $(seq 1 "$ROLLOUTS"); do
    local config_file
    local rollout_file
    config_file="$(config_file_for_round "$round")"
    rollout_file="$(rollout_file_for_round "$round")"
    "$PYTHON_BIN" - "$config_file" "$SAMPLE_FILE" "$rollout_file" "$THREADS" "$MODEL" "$TEMPERATURE" "$TOP_P" "$MAX_COMPLETION_TOKENS" "$round" <<'PY'
import json
import sys

config_file, sample_file, rollout_file, threads, model, temperature, top_p, max_tokens, round_id = sys.argv[1:10]
config = {
    "task": "voucher",
    "system_prompt_file": "src/agent/prompt/rollout.md",
    "synthesize_file": sample_file,
    "rollout_file": rollout_file,
    "threads": int(threads),
    "exclude_tools": ["web_search"],
    "history_compression": "state_folded",
    "model_config": {
        "model": model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_completion_tokens": int(max_tokens),
    },
}
with open(config_file, "w", encoding="utf-8") as fout:
    json.dump(config, fout, indent=4)
    fout.write("\n")
print(f"[INFO] wrote rollout config round {round_id} -> {config_file}")
PY
  done

  "$PYTHON_BIN" - "$RS_CONFIG_FILE" "$SAMPLE_FILE" "$SFT_FILE" "$ROLLOUTS" "$ACTIVE_RUN_ID" "$ROOT_DIR" <<'PY'
import json
import sys

config_file, sample_file, sft_file, rollouts, run_id, root_dir = sys.argv[1:7]
rollout_files = [
    f"{root_dir}/data/tmp/{run_id}_rollout_{idx:02d}.jsonl"
    for idx in range(1, int(rollouts) + 1)
]
config = {
    "task": "voucher",
    "synthesize_file": sample_file,
    "rollout_files": rollout_files,
    "rs_file": sft_file,
}
with open(config_file, "w", encoding="utf-8") as fout:
    json.dump(config, fout, indent=4)
    fout.write("\n")
print(f"[INFO] wrote RS config -> {config_file}")
PY
}

strict_check() {
  PYTHONPATH=src/agent "$PYTHON_BIN" - "$SAMPLE_FILE" "$SFT_FILE" "$CHECK_SUMMARY_FILE" "$ROLLOUTS" "$ACTIVE_RUN_ID" "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

from rewards.orm import length_reward
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES

sample_file = Path(sys.argv[1])
sft_file = Path(sys.argv[2])
summary_file = Path(sys.argv[3])
rollouts = int(sys.argv[4])
run_id = sys.argv[5]
root_dir = Path(sys.argv[6])
rollout_files = [
    root_dir / "data" / "tmp" / f"{run_id}_rollout_{idx:02d}.jsonl"
    for idx in range(1, rollouts + 1)
]

expected_queries = []
with sample_file.open(encoding="utf-8") as fin:
    for line in fin:
        if line.strip():
            expected_queries.append(json.loads(line)["query"])

errors = []
format_checked_steps = 0
tool_names = set()
harness_states = set()
harness_state_steps = 0
terminated = 0
rollout_rows = 0
format_sft_samples = []
per_rollout = []

for rollout_idx, rollout_file in enumerate(rollout_files, 1):
    rows = []
    if not rollout_file.exists():
        errors.append(f"rollout {rollout_idx}: missing file {rollout_file}")
    else:
        with rollout_file.open(encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    rows.append(json.loads(line))
    rollout_rows += len(rows)
    queries = []
    rollout_terminated = 0

    if len(rows) != len(expected_queries):
        errors.append(
            f"rollout {rollout_idx}: line count mismatch: expected {len(expected_queries)}, got {len(rows)}"
        )

    for row_idx, row in enumerate(rows, 1):
        if not isinstance(row, list) or not row:
            errors.append(f"rollout {rollout_idx} row {row_idx}: rollout row is not a non-empty list")
            continue
        query = row[0].get("extra_info", {}).get("query", "")
        queries.append(query)
        if query not in expected_queries:
            errors.append(f"rollout {rollout_idx} row {row_idx}: query is not from sample file")
        if length_reward(row) > 0:
            terminated += 1
            rollout_terminated += 1

        for step_idx, step in enumerate(row, 1):
            prompt = step.get("prompt")
            completion_obj = step.get("completion")
            extra_info = step.get("extra_info") or {}
            if extra_info.get("history_compression") != "state_folded":
                errors.append(
                    f"rollout {rollout_idx} row {row_idx} step {step_idx}: history_compression is not state_folded"
                )
            harness_state = extra_info.get("harness_state")
            if harness_state not in {"CANDIDATE_SEARCH", "CANDIDATE_SELECT", "DECISION"}:
                errors.append(
                    f"rollout {rollout_idx} row {row_idx} step {step_idx}: invalid harness_state={harness_state!r}"
                )
            else:
                harness_states.add(harness_state)
                harness_state_steps += 1
            if not isinstance(prompt, list) or len(prompt) != 2:
                errors.append(
                    f"rollout {rollout_idx} row {row_idx} step {step_idx}: prompt must contain system and user messages"
                )
                continue
            if not isinstance(completion_obj, dict):
                errors.append(f"rollout {rollout_idx} row {row_idx} step {step_idx}: completion must be a dict")
                continue
            message = Message.from_string(
                completion_obj.get("reasoning_content") or "",
                completion_obj.get("content") or "",
            )
            if not message.to_dict():
                message_dict = completion_obj.get("message")
                if isinstance(message_dict, dict):
                    message = Message.from_dict(message_dict)

            for call in message.tool_call:
                name = call.get("name")
                tool_names.add(name)
                if name == "web_search":
                    errors.append(f"rollout {rollout_idx} row {row_idx} step {step_idx}: web_search should be excluded")
                if "tool_call_id" in call:
                    del call["tool_call_id"]

            output = message.to_string(OUTPUT_ROLES)
            if not output:
                errors.append(f"rollout {rollout_idx} row {row_idx} step {step_idx}: empty normalized SFT output")
                continue
            format_checked_steps += 1
            if format_reward(output) < 1:
                errors.append(f"rollout {rollout_idx} row {row_idx} step {step_idx}: normalized output failed format_reward")
            else:
                system_prompt = next(item["content"] for item in prompt if item["role"] == "system")
                user_prompt = next(item["content"] for item in prompt if item["role"] == "user")
                format_sft_samples.append(
                    {
                        "instruction": system_prompt,
                        "input": user_prompt,
                        "output": output,
                    }
                )

    if sorted(queries) != sorted(expected_queries):
        errors.append(f"rollout {rollout_idx}: completed query set does not match sampled query set")
    per_rollout.append(
        {
            "rollout": rollout_idx,
            "file": str(rollout_file),
            "rows": len(rows),
            "terminated_rows": rollout_terminated,
        }
    )

with sft_file.open(encoding="utf-8") as fin:
    raw = fin.read().strip()
    rs_samples = json.loads(raw) if raw else []
if not isinstance(rs_samples, list):
    errors.append("RS SFT file is not a JSON list")
else:
    for idx, sample in enumerate(rs_samples):
        if set(sample) != {"instruction", "input", "output"}:
            errors.append(f"RS sample {idx}: fields must be instruction/input/output")
        elif format_reward(sample["output"]) < 1:
            errors.append(f"RS sample {idx}: output failed format_reward")

summary = {
    "expected_queries": len(expected_queries),
    "num_rollouts": rollouts,
    "expected_total_rollout_rows": len(expected_queries) * rollouts,
    "rollout_rows": rollout_rows,
    "terminated_rows": terminated,
    "per_rollout": per_rollout,
    "format_checked_steps": format_checked_steps,
    "format_all_ok": not errors,
    "tool_names": sorted(name for name in tool_names if name),
    "harness_states": sorted(harness_states),
    "harness_state_steps": harness_state_steps,
    "rs_sft_samples": len(rs_samples) if isinstance(rs_samples, list) else None,
    "format_sft_samples": len(format_sft_samples),
    "errors": errors,
}
summary_file.parent.mkdir(parents=True, exist_ok=True)
with summary_file.open("w", encoding="utf-8") as fout:
    json.dump(summary, fout, indent=2, ensure_ascii=False)
    fout.write("\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if errors:
    raise SystemExit(1)
PY
}

run_main() {
  mkdir -p "$LOG_DIR"
  echo "[INFO] run_id=${ACTIVE_RUN_ID}"
  echo "[INFO] model=${MODEL:-<resolved from MIMO_MODEL>}"
  echo "[INFO] threads=${THREADS}"
  echo "[INFO] num_queries=${NUM_QUERIES}"
  echo "[INFO] rollouts=${ROLLOUTS}"
  echo "[INFO] disable_env_proxy=${DISABLE_ENV_PROXY}"
  echo "[INFO] rollout_timeout_seconds=${ROLLOUT_TIMEOUT_SECONDS}"
  echo "[INFO] log_dir=${LOG_DIR}"

  load_runtime_env
  if [[ "$MODEL" == mimo* ]]; then
    if [[ -z "${MIMO_API_KEY:-}" || -z "${MIMO_BASE_URL:-}" ]]; then
      echo "[ERROR] MIMO_API_KEY/MIMO_BASE_URL must be available in the background environment" >&2
      exit 1
    fi
  else
    if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
      echo "[ERROR] OPENAI_API_KEY/OPENAI_BASE_URL must be available in the background environment for non-mimo models" >&2
      exit 1
    fi
  fi

  total_start="$(date +%s)"
  echo "[INFO] ensuring search server"
  ensure_search_server || exit 1
  echo "[INFO] preparing sample"
  make_sample || exit 1
  echo "[INFO] preparing configs"
  make_configs || exit 1

  rollout_start="$(date +%s)"
  for round in $(seq 1 "$ROLLOUTS"); do
    config_file="$(config_file_for_round "$round")"
    rollout_file="$(rollout_file_for_round "$round")"
    rollout_log="$LOG_DIR/rollout_${round}.log"
    before_rows=0
    if [[ -f "$rollout_file" ]]; then
      before_rows="$(wc -l < "$rollout_file")"
    fi
    echo "[INFO] starting rollout round ${round}/${ROLLOUTS}: existing_rows=${before_rows}, file=${rollout_file}"
    if command -v timeout >/dev/null 2>&1; then
      timeout "$ROLLOUT_TIMEOUT_SECONDS" "$PYTHON_BIN" src/agent/run_rollout.py "$config_file" > "$rollout_log" 2>&1
      rc=$?
    else
      "$PYTHON_BIN" src/agent/run_rollout.py "$config_file" > "$rollout_log" 2>&1
      rc=$?
    fi
    after_rows=0
    if [[ -f "$rollout_file" ]]; then
      after_rows="$(wc -l < "$rollout_file")"
    fi
    echo "[INFO] finished rollout round ${round}/${ROLLOUTS}: rows=${after_rows}"
    if [[ "$rc" -ne 0 ]]; then
      echo "[ERROR] rollout round ${round} failed or timed out with rc=${rc}; see ${rollout_log}" >&2
      exit 1
    fi
  done
  rollout_end="$(date +%s)"
  echo "[INFO] all rollout rounds finished in $((rollout_end - rollout_start))s"

  echo "[INFO] running evaluation"
  for round in $(seq 1 "$ROLLOUTS"); do
    config_file="$(config_file_for_round "$round")"
    "$PYTHON_BIN" src/agent/run_evaluate.py "$config_file" > "$LOG_DIR/evaluate_${round}.log" 2>&1 || {
      echo "[ERROR] evaluation round ${round} failed; see $LOG_DIR/evaluate_${round}.log" >&2
      exit 1
    }
  done

  echo "[INFO] running rejection sampling to SFT JSON across ${ROLLOUTS} rollout files"
  "$PYTHON_BIN" src/agent/run_rs.py "$RS_CONFIG_FILE" > "$LOG_DIR/run_rs.log" 2>&1 || {
    echo "[ERROR] run_rs failed; see $LOG_DIR/run_rs.log" >&2
    exit 1
  }

  echo "[INFO] running strict format check"
  strict_check > "$LOG_DIR/strict_check.log" 2>&1 || {
    echo "[ERROR] strict format check failed; see $LOG_DIR/strict_check.log" >&2
    exit 1
  }

  total_end="$(date +%s)"
  {
    echo "RUN_ID=${ACTIVE_RUN_ID}"
    echo "STATUS=success"
    echo "MODEL=${MODEL}"
    echo "THREADS=${THREADS}"
    echo "NUM_QUERIES=${NUM_QUERIES}"
    echo "ROLLOUTS=${ROLLOUTS}"
    echo "ROLLOUT_SECONDS=$((rollout_end - rollout_start))"
    echo "TOTAL_SECONDS=$((total_end - total_start))"
    echo "SAMPLE_FILE=${SAMPLE_FILE}"
    for round in $(seq 1 "$ROLLOUTS"); do
      echo "ROLLOUT_FILE_${round}=$(rollout_file_for_round "$round")"
    done
    echo "SFT_FILE=${SFT_FILE}"
    echo "RS_CONFIG_FILE=${RS_CONFIG_FILE}"
    echo "STRICT_CHECK=${CHECK_SUMMARY_FILE}"
  } > "$SUMMARY_FILE"
  cat "$SUMMARY_FILE"
}

show_status() {
  echo "[INFO] run_id=${ACTIVE_RUN_ID}"
  echo "[INFO] log_dir=${LOG_DIR}"
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[INFO] status=running pid=${pid}"
    else
      echo "[INFO] status=not-running pid=${pid}"
    fi
  else
    echo "[INFO] status=no-pid-file"
  fi
  for round in $(seq 1 "$ROLLOUTS"); do
    rollout_file="$(rollout_file_for_round "$round")"
    if [[ -f "$rollout_file" ]]; then
      echo "[INFO] rollout_${round}_rows=$(wc -l < "$rollout_file")"
    else
      echo "[INFO] rollout_${round}_rows=0"
    fi
  done
  if [[ -f "$SUMMARY_FILE" ]]; then
    cat "$SUMMARY_FILE"
  fi
}

start_background() {
  mkdir -p "$LOG_DIR"
  runner=(
    env RUN_ID="$ACTIVE_RUN_ID"
    PYTHON_BIN="$PYTHON_BIN"
    JAVA_HOME="$JAVA_HOME"
    PORT="$PORT"
    NUM_QUERIES="$NUM_QUERIES"
    ROLLOUTS="$ROLLOUTS"
    SHOPPINGBENCH_SEARCH_CAPACITY="$SHOPPINGBENCH_SEARCH_CAPACITY"
    THREADS="$THREADS"
    MODEL="$MODEL"
    MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS"
    TEMPERATURE="$TEMPERATURE"
    TOP_P="$TOP_P"
    DISABLE_ENV_PROXY="$DISABLE_ENV_PROXY"
    LOAD_BASHRC_EXPORTS="$LOAD_BASHRC_EXPORTS"
    WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS"
    SHOPPINGBENCH_LLM_TIMEOUT="$SHOPPINGBENCH_LLM_TIMEOUT"
    ROLLOUT_TIMEOUT_SECONDS="$ROLLOUT_TIMEOUT_SECONDS"
    "$ROOT_DIR/scripts/run_mimo_voucher_smoke5.sh" run
  )
  if command -v setsid >/dev/null 2>&1; then
    setsid "${runner[@]}" > "$MAIN_LOG" 2>&1 < /dev/null &
  else
    nohup "${runner[@]}" > "$MAIN_LOG" 2>&1 < /dev/null &
  fi
  echo "$!" > "$PID_FILE"
  echo "[INFO] started background smoke run"
  show_status
  echo "[INFO] main log: ${MAIN_LOG}"
}

wait_for_run() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "[ERROR] pid file not found: ${PID_FILE}" >&2
    exit 1
  fi
  pid="$(cat "$PID_FILE")"
  while kill -0 "$pid" >/dev/null 2>&1; do
    show_status
    sleep 30
  done
  show_status
  if [[ -f "$SUMMARY_FILE" ]] && grep -q '^STATUS=success$' "$SUMMARY_FILE"; then
    exit 0
  fi
  echo "[ERROR] run did not finish successfully; see ${MAIN_LOG}" >&2
  exit 1
}

case "$ACTION" in
  start)
    start_background
    ;;
  run)
    run_main
    ;;
  status)
    show_status
    ;;
  wait)
    wait_for_run
    ;;
  tail)
    tail -f "$MAIN_LOG"
    ;;
  *)
    echo "Usage: RUN_ID=<optional> $0 {start|run|status|wait|tail}" >&2
    exit 2
    ;;
esac
