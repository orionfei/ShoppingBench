#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
ENV_FILE="${ENV_FILE:-/root/project/ResearchHarness/.env}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/shoppingbench-verl/bin/python}"
OUT_DIR="${OUT_DIR:-data/rl_v3}"
TOTAL="${TOTAL:-1500}"
MODEL="qwen3.6-flash"
PLAN="$OUT_DIR/candidates_${TOTAL}.plan.jsonl"
EXCLUDED="$OUT_DIR/excluded_test75_product_ids.txt"
MODEL_CONFIG="config/rl/rl_v3_query_generator_qwen36_flash.json"

[[ -f "$ENV_FILE" ]] || { echo "[ERROR] env file not found: $ENV_FILE" >&2; exit 2; }
set -a
# The trusted env file is loaded only at runtime; its secrets are never printed or copied.
source "$ENV_FILE"
set +a
[[ -n "${API_KEY:-}" && -n "${API_BASE:-}" ]] || {
  echo "[ERROR] API_KEY/API_BASE are not set in $ENV_FILE" >&2
  exit 2
}
export RL_QUERY_API_KEY="${RL_QUERY_API_KEY:-$API_KEY}"
api_host="$($PYTHON_BIN - "$API_BASE" <<'PY'
import sys, urllib.parse
print(urllib.parse.urlparse(sys.argv[1]).hostname or "unknown")
PY
)"
echo "[rl-v3] mode=$MODE model=$MODEL api_key_set=1 api_host=$api_host total=$TOTAL"

prepare() {
  "$PYTHON_BIN" scripts/prepare_rl_v3_candidates.py \
    --output-dir "$OUT_DIR" --total "$TOTAL" --seed 20260711 --max-docs 100000
}

generate() {
  local output="$1" metadata="$2" concurrency="$3" max_generate="${4:-}"
  local extra=()
  [[ -n "$max_generate" ]] && extra+=(--max-generate "$max_generate")
  "$PYTHON_BIN" scripts/sample_coupon_budget.py \
    --stage generate --profile rl-v3-candidate --plan-output "$PLAN" \
    --output "$output" --metadata-output "$metadata" \
    --model-config "$MODEL_CONFIG" --model "$MODEL" --base-url "$API_BASE" \
    --api-key-env RL_QUERY_API_KEY --concurrency "$concurrency" \
    --request-timeout 120 --llm-retries 3 --max-completion-tokens 512 \
    "${extra[@]}"
}

case "$MODE" in
  prepare)
    prepare
    ;;
  smoke)
    [[ -f "$PLAN" ]] || prepare
    generate "$OUT_DIR/smoke3.query.jsonl" "$OUT_DIR/smoke3.query.meta.jsonl" 1 3
    "$PYTHON_BIN" scripts/sample_coupon_budget.py --stage summarize-final \
      --output "$OUT_DIR/smoke3.query.jsonl"
    ;;
  full)
    [[ -f "$PLAN" ]] || prepare
    generate "$OUT_DIR/candidates_${TOTAL}.query.jsonl" "$OUT_DIR/candidates_${TOTAL}.query.meta.jsonl" 8
    "$PYTHON_BIN" scripts/audit_rl_v3_candidates.py \
      --plan "$PLAN" --queries "$OUT_DIR/candidates_${TOTAL}.query.jsonl" \
      --metadata "$OUT_DIR/candidates_${TOTAL}.query.meta.jsonl" \
      --excluded-product-ids "$EXCLUDED" --output-dir "$OUT_DIR" \
      --manifest "$OUT_DIR/manifest.json"
    ;;
  *)
    echo "usage: $0 {prepare|smoke|full}" >&2
    exit 2
    ;;
esac
