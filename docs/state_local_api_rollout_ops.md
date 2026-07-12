# State-Local API Rollout Operations

This document records the operational procedure for running real model rollouts
against the current ShoppingBench state-local harness.

The formal rollout format is strict:

```text
<think>...</think><tool_call>[{"name":"tool_name","parameters":{}}]</tool_call>
```

Do not use test-only adapters for formal teacher/RL data unless explicitly
requested.

## Safety Rules

- Do not write real API keys into tracked config files.
- Prefer environment variables for keys.
- Keep `--accept-structured-reasoning`, `--no-think-output`, and
  `--loose-tool-call-extraction` for probes only. They are not formal strict
  trajectory modes.
- For the formal harness, visible content must include a non-empty literal
  `<think>...</think>` block followed by one literal `<tool_call>...</tool_call>`
  block.
- `CANDIDATE_SELECT` must not allow `find_product`; use the current FSM rather
  than the original raw-history harness.

## Prerequisites

Work from the ShoppingBench repo root:

```bash
cd /root/project/ShoppingBench
```

Use the ShoppingBench conda environment:

```bash
PY=/root/miniconda3/envs/shoppingbench/bin/python
```

The search server must be reachable at `http://127.0.0.1:5631/`:

```bash
curl --noproxy '127.0.0.1,localhost' -fsS http://127.0.0.1:5631/
```

If it is down, start it:

```bash
JAVA_HOME=/root/.local/jdks/temurin-21 \
PATH=/root/.local/jdks/temurin-21/bin:$PATH \
OPENAI_API_KEY=EMPTY INDEX_DIR=indexes PORT=5631 \
$PY src/search_engine/server.py
```

Leave that process running while the rollout runs.

Before running, verify the harness code:

```bash
$PY -m py_compile \
  scripts/run_state_local_api_rollout.py \
  src/agent/util/harness_fsm.py \
  src/rl/verl/experimental/agent_loop/tool_agent_loop.py \
  scripts/test_harness_fsm_transitions.py

$PY scripts/test_harness_fsm_transitions.py
```

## Sample Files

The current broad 16-query sample used in experiments:

```text
data/tmp/state_local_broad16_20260707.jsonl
```

Create an 8-query subset from the first 8 rows:

```bash
$PY - <<'PY'
from pathlib import Path
src = Path("data/tmp/state_local_broad16_20260707.jsonl")
out = Path("data/tmp/state_local_broad8_from16_20260708.jsonl")
lines = [line for line in src.open(encoding="utf-8") if line.strip()][:8]
out.write_text("".join(lines), encoding="utf-8")
print(out, len(lines))
PY
```

## Endpoint A: Local Codex Proxy

Use this when the local proxy is prepared:

```text
Base URL: http://127.0.0.1:18080/v1
API Key: pwd
Model: gpt-5.5
```

Check available models:

```bash
curl --noproxy '127.0.0.1,localhost' \
  -fsS http://127.0.0.1:18080/v1/models
```

For the harness, use non-streaming chat completions. Streaming worked in a
minimal probe, but returned an empty `<think></think>` in one case, which fails
the formal non-empty think requirement.

Minimal strict-format probe:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
$PY - <<'PY'
from openai import OpenAI
import json, re

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="pwd", timeout=120)
messages = [
    {
        "role": "system",
        "content": (
            "Output exactly one <think>...</think> block followed by exactly one "
            "<tool_call>...</tool_call> block and nothing else. This is "
            "ShoppingBench action schema, not native function calling. Inside "
            "<tool_call> use a JSON array. Each item has exactly keys "
            '"name" and "parameters".'
        ),
    },
    {"role": "user", "content": 'Allowed tool: find_product. Return q="red shoes", page=1.'},
]
resp = client.chat.completions.create(
    model="gpt-5.5",
    messages=messages,
    max_tokens=1024,
    temperature=0.6,
)
content = resp.choices[0].message.content or ""
print(content)
assert content.count("<think>") == 1 and content.count("</think>") == 1
assert content.count("<tool_call>") == 1 and content.count("</tool_call>") == 1
payload = re.search(r"<tool_call>(.*?)</tool_call>", content, re.S).group(1).strip()
parsed = json.loads(payload)
assert isinstance(parsed, list)
assert all(set(item) == {"name", "parameters"} for item in parsed)
client.close()
PY
```

One-query smoke can be done by creating a one-row sample:

```bash
$PY - <<'PY'
from pathlib import Path
src = Path("data/tmp/state_local_broad8_from16_20260708.jsonl")
out = Path("data/tmp/state_local_smoke1_from_broad8_20260708.jsonl")
out.write_text(next(line for line in src.open(encoding="utf-8") if line.strip()), encoding="utf-8")
print(out)
PY
```

Then run:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
$PY scripts/run_state_local_api_rollout.py \
  --source data/synthesize_voucher_train.jsonl \
  --sample-file data/tmp/state_local_smoke1_from_broad8_20260708.jsonl \
  --rollout-file data/tmp/state_local_think_gpt55proxy_smoke1_YYYYMMDD_rollout.jsonl \
  --report-file data/tmp/state_local_think_gpt55proxy_smoke1_YYYYMMDD_report.json \
  --model gpt-5.5 \
  --base-url http://127.0.0.1:18080/v1 \
  --api-key pwd \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-completion-tokens 4096 \
  --max-steps 15 \
  --workers 1
```

Run 8 queries with concurrency 8:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
$PY scripts/run_state_local_api_rollout.py \
  --source data/synthesize_voucher_train.jsonl \
  --sample-file data/tmp/state_local_broad8_from16_20260708.jsonl \
  --rollout-file data/tmp/state_local_think_gpt55proxy_broad8_w8_s15_YYYYMMDD_rollout.jsonl \
  --report-file data/tmp/state_local_think_gpt55proxy_broad8_w8_s15_YYYYMMDD_report.json \
  --model gpt-5.5 \
  --base-url http://127.0.0.1:18080/v1 \
  --api-key pwd \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-completion-tokens 4096 \
  --max-steps 15 \
  --workers 8
```

## Endpoint B: Remote OpenAI-Compatible API

Use this pattern for the remote API endpoint used in earlier experiments:

```bash
export REMOTE_BASE_URL="http://35.220.164.252:3888/v1"
export REMOTE_API_KEY="<YOUR_API_KEY>"
```

Important: bypass the environment proxy for this host, otherwise Python/OpenAI
requests may return nginx `404`.

```bash
export NO_PROXY="35.220.164.252,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
```

### Recommended Strict Model: qwen3.5-flash

`qwen3.5-flash` passed a minimal strict-format probe. For best observed strict
behavior, use:

```text
model: qwen3.5-flash
thinking.type: disabled
temperature: 0.6
top_p: 0.95
```

Probe:

```bash
NO_PROXY=35.220.164.252,127.0.0.1,localhost \
no_proxy=35.220.164.252,127.0.0.1,localhost \
$PY - <<'PY'
from openai import OpenAI
import httpx, json, os, re

client = OpenAI(
    base_url=os.environ["REMOTE_BASE_URL"],
    api_key=os.environ["REMOTE_API_KEY"],
    timeout=120,
    http_client=httpx.Client(trust_env=False),
)
messages = [
    {
        "role": "system",
        "content": (
            "Output exactly one <think>...</think> block followed by exactly one "
            "<tool_call>...</tool_call> block and nothing else. This is "
            "ShoppingBench action schema, not Qwen/OpenAI native function calling. "
            'Inside <tool_call> use a JSON array. Each item has exactly keys "name" and "parameters".'
        ),
    },
    {"role": "user", "content": 'Allowed tool: find_product. Return q="red shoes", page=1.'},
]
resp = client.chat.completions.create(
    model="qwen3.5-flash",
    messages=messages,
    max_completion_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    extra_body={"thinking": {"type": "disabled"}},
)
content = resp.choices[0].message.content or ""
print(content)
assert content.count("<think>") == 1 and content.count("</think>") == 1
assert content.count("<tool_call>") == 1 and content.count("</tool_call>") == 1
parsed = json.loads(re.search(r"<tool_call>(.*?)</tool_call>", content, re.S).group(1).strip())
assert isinstance(parsed, list)
assert all(set(item) == {"name", "parameters"} for item in parsed)
client.close()
PY
```

Run 8 queries:

```bash
NO_PROXY=35.220.164.252,127.0.0.1,localhost \
no_proxy=35.220.164.252,127.0.0.1,localhost \
$PY scripts/run_state_local_api_rollout.py \
  --source data/synthesize_voucher_train.jsonl \
  --sample-file data/tmp/state_local_broad8_from16_20260708.jsonl \
  --rollout-file data/tmp/state_local_think_qwen35flash_broad8_w8_s15_YYYYMMDD_rollout.jsonl \
  --report-file data/tmp/state_local_think_qwen35flash_broad8_w8_s15_YYYYMMDD_report.json \
  --model qwen3.5-flash \
  --base-url "$REMOTE_BASE_URL" \
  --api-key "$REMOTE_API_KEY" \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-completion-tokens 4096 \
  --max-steps 15 \
  --workers 8 \
  --thinking-type disabled
```

Run 16 queries:

```bash
NO_PROXY=35.220.164.252,127.0.0.1,localhost \
no_proxy=35.220.164.252,127.0.0.1,localhost \
$PY scripts/run_state_local_api_rollout.py \
  --source data/synthesize_voucher_train.jsonl \
  --sample-file data/tmp/state_local_broad16_20260707.jsonl \
  --rollout-file data/tmp/state_local_think_qwen35flash_broad16_w8_s15_YYYYMMDD_rollout.jsonl \
  --report-file data/tmp/state_local_think_qwen35flash_broad16_w8_s15_YYYYMMDD_report.json \
  --model qwen3.5-flash \
  --base-url "$REMOTE_BASE_URL" \
  --api-key "$REMOTE_API_KEY" \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-completion-tokens 4096 \
  --max-steps 15 \
  --workers 8 \
  --thinking-type disabled
```

### Models That Need Caution

These observations are from real probes:

- `Qwen/Qwen3.5-35B-A3B`: callable on the remote endpoint, but default mode
  uses `reasoning_content` and visible content was not strict. With thinking
  disabled, visible content was clean `<tool_call>` only but lacked literal
  `<think>`. Not recommended for formal strict trajectories.
- `Qwen/Qwen3-32B`: tends to use provider reasoning or raw JSON/tool-like
  content rather than formal literal `<think>...</think><tool_call>...</tool_call>`.
  Not recommended for formal strict trajectories.
- `qwen3.5-flash`: strict one-step probe passed, but multi-turn rollout can
  still produce `<ththink>`, `<then>`, raw JSON, or repeated tags. Use report
  error counts to judge stability.

## Endpoint C: Local Qwen3-4B via vLLM

Use this when testing the local base model:

```text
model path: model/Qwen3-4B
served model: qwen3-4b-local
base url: http://127.0.0.1:30000/v1
api key: EMPTY
```

Start vLLM from the VERL environment:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 \
/root/miniconda3/envs/shoppingbench-verl/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-4B \
  --served-model-name qwen3-4b-local \
  --host 127.0.0.1 \
  --port 30000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.78 \
  --tensor-parallel-size 1 \
  --disable-log-requests
```

Do not add `--enable-reasoning` for strict literal trajectory tests. Without
reasoning parser, literal `<think>` remains in visible `content`.

Use Qwen3 recommended thinking sampling:

```text
temperature: 0.6
top_p: 0.95
top_k: 20
```

The rollout script supports `--top-k`.

Example:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
$PY scripts/run_state_local_api_rollout.py \
  --source data/synthesize_voucher_train.jsonl \
  --sample-file data/tmp/state_local_broad8_from16_20260708.jsonl \
  --rollout-file data/tmp/state_local_think_qwen3_4b_local_broad8_w8_s15_YYYYMMDD_rollout.jsonl \
  --report-file data/tmp/state_local_think_qwen3_4b_local_broad8_w8_s15_YYYYMMDD_report.json \
  --model qwen3-4b-local \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key EMPTY \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-completion-tokens 4096 \
  --max-steps 15 \
  --workers 8
```

Stop vLLM with Ctrl-C when done.

## Evaluation

Create an evaluation config for any rollout:

```bash
cat > data/tmp/EVAL_CONFIG.json <<'JSON'
{
  "task": "voucher",
  "synthesize_file": "data/tmp/state_local_broad8_from16_20260708.jsonl",
  "rollout_file": "data/tmp/ROLLOUT_FILE.jsonl",
  "model_config": {"model": "MODEL_NAME"}
}
JSON
```

Run official evaluation:

```bash
JAVA_HOME=/root/.local/jdks/temurin-21 \
PATH=/root/.local/jdks/temurin-21/bin:$PATH \
INDEX_DIR=indexes OPENAI_API_KEY=EMPTY \
$PY src/agent/run_evaluate.py data/tmp/EVAL_CONFIG.json \
  2>&1 | tee data/tmp/EVAL_LOG.log
```

The important metrics are:

- `success rate`
- `gt rate`
- `format score`
- `sku & attrs match`
- `rule match`
- `budget match`

## Inspecting Harness Errors

Read the generated report:

```bash
cat data/tmp/REPORT_FILE.json
```

Useful fields:

- `audit.terminated`
- `audit.steps`
- `audit.state_counts`
- `audit.tool_counts`
- `audit.error_counts`

Quick per-row summary:

```bash
$PY - <<'PY'
import json
p = "data/tmp/ROLLOUT_FILE.jsonl"
rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
for i, row in enumerate(rows, 1):
    last = row[-1]["completion"]["message"] if row else {}
    errors = []
    for step in row:
        for obs in step["completion"]["message"].get("obs") or []:
            result = obs.get("results")
            if isinstance(result, dict) and result.get("error"):
                errors.append(result["error"])
    print(
        i,
        "steps", len(row),
        "last_state", row[-1]["extra_info"]["harness_state"],
        "last_tools", [call.get("name") for call in last.get("tool_call") or []],
        "errors", {e: errors.count(e) for e in sorted(set(errors))},
    )
PY
```

Check format-error shapes:

```bash
$PY - <<'PY'
import json, re
from collections import Counter
p = "data/tmp/ROLLOUT_FILE.jsonl"
rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
patterns = Counter()
for row in rows:
    for step in row:
        content = step["completion"]["content"] or ""
        has_format_error = False
        for obs in step["completion"]["message"].get("obs") or []:
            result = obs.get("results")
            if isinstance(result, dict) and result.get("error") == "exactly_one_think_block_required":
                has_format_error = True
        if has_format_error:
            match = re.match(r"\s*<([^>]+)>", content)
            patterns[match.group(1) if match else (content.strip()[:30] or "empty")] += 1
print(dict(patterns))
PY
```

## Common Issues

### nginx 404 from remote endpoint

Set `NO_PROXY`/`no_proxy` for the endpoint host. For direct Python probes, use:

```python
http_client=httpx.Client(trust_env=False)
```

### Empty `<think>` with Codex Proxy

Use non-streaming calls for the formal harness. Streaming can produce an empty
literal think block in some probes.

### Qwen native `arguments` schema

The formal ShoppingBench action schema uses `parameters`, not `arguments`, and
the tool call payload must be a JSON array. The harness should reject the wrong
shape and ask the model to correct it; it must not convert it automatically.

### `tool_not_allowed_in_current_state`

Most often caused by the model trying to `find_product` in `CANDIDATE_SELECT`.
The current FSM delays the first transition to SELECT after the first non-empty
search to reduce this failure mode, but once in SELECT the allowed tools remain
`view_product_information` and `budget_check`.

### Repeated searches after delayed SELECT

Delayed SELECT reduces illegal search-in-SELECT errors but can increase
`repeated_search_not_allowed`. Inspect `candidate_pool`, `previous_searches`,
and the model's next search terms to decide whether the prompt needs stronger
"search only missing product needs" guidance.
