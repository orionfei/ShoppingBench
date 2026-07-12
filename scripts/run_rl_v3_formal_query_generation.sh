#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_FILE="${ENV_FILE:-/root/project/ResearchHarness/.env}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/shoppingbench-verl/bin/python}"
set -a
source "$ENV_FILE"
set +a
export RL_QUERY_API_KEY="${RL_QUERY_API_KEY:-$API_KEY}"
MODEL_CONFIG=config/rl/rl_v3_query_generator_qwen36_flash.json

generate() {
  local stem="$1" concurrency="$2"
  "$PYTHON_BIN" scripts/sample_coupon_budget.py \
    --stage generate --profile rl-v3-candidate \
    --plan-output "data/rl_v3/formal/${stem}.plan.jsonl" \
    --output "data/rl_v3/formal/${stem}.query.jsonl" \
    --metadata-output "data/rl_v3/formal/${stem}.query.meta.jsonl" \
    --model-config "$MODEL_CONFIG" --model qwen3.6-flash --base-url "$API_BASE" \
    --api-key-env RL_QUERY_API_KEY --concurrency "$concurrency" \
    --request-timeout 120 --llm-retries 3 --max-completion-tokens 512
}

generate train_repair 2
generate val64 8
"$PYTHON_BIN" scripts/prepare_rl_v3_formal_data.py finalize
