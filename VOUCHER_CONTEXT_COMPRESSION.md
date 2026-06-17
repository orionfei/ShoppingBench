# Voucher/Budget Context Compression Notes

This document records the context compression strategy implemented for the
ShoppingBench Voucher/Budget task.

## Goal

Voucher/Budget rollouts are multi-turn tool-use trajectories. In the original
history format, each step appends full assistant thoughts, tool calls, and tool
observations into the next prompt:

```text
<user>...</user>

<think>...</think>
<tool_call>...</tool_call>
<obs>...</obs>

<think>...</think>
<tool_call>...</tool_call>
<obs>...</obs>
```

This is expensive because `find_product` can return many products and
`view_product_information` can return long descriptions. For Voucher/Budget,
the decision state is much smaller than the raw history. The compression goal is
to preserve the information needed for later tool calls while reducing prompt
tokens during rollout and RL training.

## Implemented Modes

### 1. Raw history

This is the original behavior in `src/agent/run_rollout.py`.

Each new prompt includes all previous user/assistant messages verbatim:

```text
# Dialogue Records History
<user>...</user>

<think>...</think>
<tool_call>...</tool_call>
<obs>...</obs>
```

This is the most faithful format, but it is context-heavy.

### 2. Field-pruned observation history

This was used as an earlier diagnostic compression. It keeps the same history
shape but removes fields that are not useful for Voucher/Budget decisions.

Safe removals for Voucher/Budget:

- `find_product.sold_count`
- `view_product_information.description`
- `view_product_information.short_description`
- verbose `recommend_product` observation text
- verbose `terminate` observation text

Fields kept:

- `find_product`: `product_id`, `shop_id`, `title`, `price`, `service`
- `view_product_information`: `product_id`, `sku_options`, `attributes`
- `python_execute`: execution result and success flag

This preserves almost the original interaction format, so it is conservative.

### 3. State-folded history

This is the implemented online compression path.

Instead of replaying every previous `<think>`, `<tool_call>`, and `<obs>`, the
rollout prompt becomes:

```text
# Dialogue Records History
<user>...</user>

<state>{...}</state>
```

The current model output is not compressed. The policy still has to emit the
normal format:

```text
<think>...</think>
<tool_call>[{"name":"...", "parameters":{...}}]</tool_call>
```

Only the prompt history is folded.

## Online Implementation

The online state folding implementation is in:

```text
src/agent/util/history_compression.py
```

It is connected to rollout in:

```text
src/agent/run_rollout.py
```

Enable it with config:

```json
{
  "history_compression": "state_folded",
  "state_max_candidates_per_search": 5
}
```

The `state_max_candidates_per_search` parameter controls how many candidates
from each `find_product` observation are retained in the state. The current
tested aggressive setting is `5`.

## Important Safety Property

The online state builder does not read:

- gold `reward`
- final answer product ids
- future tool calls
- future observations

It only uses the already accumulated rollout history:

- original user query
- previous tool calls
- previous tool observations
- previous recommendations/termination, if any

This matters because RL rollout must not leak target products into the policy
prompt.

## State Schema

The generated state is JSON inside `<state>...</state>`.

Main fields:

```json
{
  "task_type": "voucher_budget",
  "voucher": {
    "scope": "platform | shop",
    "threshold": 102,
    "budget": 153,
    "parse_ok": true,
    "discount": {
      "type": "fixed",
      "value": 15
    }
  },
  "searches": [
    {
      "parameters": {
        "q": "...",
        "page": 1,
        "shop_id": "..."
      },
      "candidates": [
        {
          "product_id": "...",
          "shop_id": "...",
          "title": "...",
          "price": 166.0,
          "service": ["flashsale"]
        }
      ]
    }
  ],
  "budget_candidates": [
    {
      "product_id": "...",
      "shop_id": "...",
      "price": 166.0
    }
  ],
  "requested_view_product_ids": ["..."],
  "selected_product_ids": ["..."],
  "selected_total_before_voucher": 166.0,
  "shop_anchor": "1222660",
  "voucher_applicable_if_now": true,
  "payable_total_if_now": 151.0,
  "within_budget_if_now": true,
  "viewed_products": [
    {
      "product_id": "...",
      "sku_options": {},
      "attributes": {}
    }
  ],
  "latest_budget_calculation": {},
  "budget_calculation_trusted": true,
  "recommendations": [],
  "terminations": [],
  "pending": ["..."]
}
```

`pending` is not just a progress marker. It must inspect the result of the
previous step. In particular:

- if the latest budget calculation is `within_budget: false`, pending becomes
  `revise_selection_or_fail`, not `recommend_products`
- if the latest budget calculation is `within_budget: true` but product details
  are missing, pending remains `verify_product_information`
- only when the selected products have usable details and the latest budget
  calculation passes should pending become `recommend_products`

This avoids a training/inference mismatch where SFT teaches the model to revise
over-budget selections, but online rollout state tells the model to recommend
any selection that merely has a budget calculation.

## How Each Tool Is Folded

### `find_product`

Raw observation can contain up to 10 products per search. The folded state keeps
two compressed views:

1. `searches[].candidates`: a bounded top-k readable list for product selection.
   This keeps title/service information and is controlled by
   `state_max_candidates_per_search`.
2. `budget_candidates`: all observed search products reduced to
   `product_id/shop_id/price`. This is intentionally not top-k truncated because
   it is the evidence used to verify voucher and budget calculations.

Example readable candidate state:

```json
{
  "parameters": {
    "q": "tank tops jersey sando",
    "page": 1,
    "shop_id": "1222660"
  },
  "candidates": [
    {
      "product_id": "4641993818",
      "shop_id": "1222660",
      "title": "...",
      "price": 149.0,
      "service": []
    }
  ]
}
```

For shop vouchers, this preserves the `shop_id` needed to keep later searches
inside the same shop. For budget verification, `budget_candidates` is the
source of truth for product prices and shop ids.

### `view_product_information`

The folded state keeps only:

- `product_id`
- `sku_options`
- `attributes`

It drops long product text fields:

- `description`
- `short_description`

These descriptions are not used by the Voucher/Budget evaluator and are usually
redundant with title, SKU options, and attributes for this task.

### `python_execute`

The folded state keeps only the parsed calculation result when possible:

```json
{
  "product_ids": ["5101399590"],
  "shop_ids": ["1670015"],
  "same_shop": true,
  "total_before_voucher": 166.0,
  "meets_threshold": true,
  "eligible_scope": true,
  "voucher_used": true,
  "payable_total": 151.0,
  "budget": 153,
  "within_budget": true
}
```

It does not need to preserve the full Python code in the historical prompt.
The parsed result is not trusted just because it is valid JSON. The online
state builder cross-checks it against observed `budget_candidates` and the
parsed voucher rules:

- every `product_id` must exist in observed search candidates
- every calculated `shop_id` must match the observed product `shop_id`
- `total_before_voucher`, `voucher_used`, `payable_total`, `budget`, and
  `within_budget` must match deterministic recomputation within one cent

If any of those checks fail, `budget_calculation_trusted` is false and the
state does not advance to `recommend_products`.

### `recommend_product`

The folded state records recommended product ids:

```json
{
  "recommendations": [
    {
      "product_ids": ["5101399590"]
    }
  ]
}
```

The verbose tool observation is not useful for future decisions.

### `terminate`

The folded state records only the status:

```json
{
  "terminations": [
    {
      "status": "success"
    }
  ]
}
```

## Example

Original history after one search:

```text
<user>Show me a black logo parking button for my Honda PCX...</user>

<think>...</think>
<tool_call>[{"name":"find_product","parameters":{"q":"...","page":1}}]</tool_call>
<obs>[{"results":[{"product_id":"5101399590", ...}, ...]}]</obs>
```

Folded online prompt for the next step:

```text
# Dialogue Records History
<user>Show me a black logo parking button for my Honda PCX...</user>

<state>{
  "task_type": "voucher_budget",
  "voucher": {
    "scope": "platform",
    "threshold": 102,
    "budget": 153,
    "discount": {"type": "fixed", "value": 15}
  },
  "searches": [
    {
      "parameters": {"q": "...", "page": 1},
      "candidates": [
        {
          "product_id": "5101399590",
          "shop_id": "1670015",
          "title": "Universal Motorcycle Parking Switch ...",
          "price": 166.0,
          "service": ["flashsale"]
        }
      ]
    }
  ],
  "pending": ["select_candidates_from_search_results"]
}</state>
```

The next model action should still be a normal tool call, for example:

```text
<think>...</think>
<tool_call>[
  {"name":"view_product_information","parameters":{"product_ids":"5101399590"}},
  {"name":"python_execute","parameters":{"code":"..."}}
]</tool_call>
```

## Offline Gold-State Script

There is also an offline diagnostic script:

```text
scripts/build_voucher_state_folded_history.py
```

This script rewrites existing trajectories into folded SFT samples and reports
compression statistics.

Important caveat: the first version of this script can use final trajectory
product ids to build a very clean state. That is useful for measuring an upper
bound and producing high-quality demonstrations, but it must not be used as the
online RL rollout state builder because it can leak the final answer.

For online RL, use the `history_compression: state_folded` path in
`run_rollout.py`.

## Replay Validation

Replay script:

```text
scripts/replay_state_folded_rollout.py
```

This script tests the actual online prompt construction path without calling an
external model API:

1. It loads existing gold trajectories.
2. It builds prompts through `get_user_prompt(..., history_compression="state_folded")`.
3. It replays the gold action as the policy output.
4. It executes local tools again through `act()`.
5. It checks that the next action's required ids or shop constraints are present
   in the folded state.

This isolates the question:

```text
Does the folded history contain enough information for the next tool call?
```

It does not prove that a trained policy will always choose the same action, but
it verifies that the compressed prompt path does not remove required state for
the tested trajectories.

## Experiment Results

Tested on 10 Voucher/Budget trajectories covering:

- platform vouchers
- shop vouchers
- fixed discounts
- percentage discounts
- one to four products
- same-shop constraints

### Earlier online state folding, top 10 candidates

```text
full prompt chars:          189,533
compact prompt chars:       164,052
online folded prompt chars: 155,808

relative to full:    17.79% char saving
relative to compact:  5.03% char saving
```

This was an earlier conservative diagnostic. The current recommended setting is
the stricter top-5 state folding below, which keeps full `budget_candidates` for
budget verification while truncating only the readable candidate list.

### Current safe online state folding, top 5 candidates

These numbers count only the user/history prompt, because that is the part
changed by history compression:

```text
full history chars:              189,533
field-pruned history chars:      164,052
safe state-folded top5 chars:    135,541

relative to full history:         28.49% char saving
relative to field-pruned history: 17.38% char saving
```

If the fixed system prompt and tool descriptions are included, the same files
measure:

```text
full total prompt chars:          400,433
field-pruned total prompt chars:  374,952
safe state-folded top5 total:     346,441

relative to full total prompt:    13.49% char saving
relative to field-pruned total:    7.60% char saving
```

Replay support checks:

```text
support_all_ok: true
```

Native Voucher/Budget evaluation on replayed trajectories:

```text
gt rate: 1.000
success rate: 1.000
format score: 1.000
recommend product score: 1.000
title match score: 1.000
price match score: 1.000
service match score: 1.000
sku & attrs match score: 1.000
rule match score: 1.000
budget match score: 1.000
```

### Post-fix adversarial validation

An adversarial subagent found two P1 issues in the first state-folded version:

- a structurally valid but numerically wrong `python_execute` JSON could become
  trusted
- a shop voucher calculation could forge `shop_ids` and falsely make a mixed
  shop selection look same-shop

The online compressor now rejects both cases by recomputing the budget from
observed product prices and shop ids. Synthetic checks confirm:

```text
wrong budget JSON trusted: false
forged shop_id JSON trusted: false
```

The parser was also tightened to parse voucher terms from the voucher section,
not from arbitrary product text. Current template coverage:

```text
synthesize_voucher_train.jsonl:      750/750 parse_ok
synthesize_voucher_test.jsonl:       250/250 parse_ok
synthesize_voucher_train_plan.jsonl: 750/750 parse_ok
```

The stricter trust logic was replayed on the six real-code gold trajectories:

```text
support_all_ok: true
trusted_state_steps: 6
gt rate: 1.000
success rate: 1.000
budget match score: 1.000
```

## Recommended RL Setting

For initial RL experiments:

```json
{
  "history_compression": "state_folded",
  "state_max_candidates_per_search": 5
}
```

This is the best current tradeoff:

- no final-answer leakage
- enough replay support on the tested 10 trajectories
- meaningful context reduction
- same output/action format as before

If failures appear in broader rollouts, increase:

```json
"state_max_candidates_per_search": 10
```

That is more conservative but saves less context.

## What This Does Not Yet Prove

The replay test does not call a policy model. It verifies that a known good next
action is still supported by the compressed state.

The next required ablation is real policy rollout:

1. raw history
2. field-pruned history
3. online state-folded history with top 5 candidates
4. online state-folded history with top 10 candidates

Compare:

- success rate
- budget match
- rule match
- tool-call format score
- average prompt tokens
- max prompt tokens
- rollout failure modes

## Current Files

Core implementation:

```text
src/agent/util/history_compression.py
src/agent/run_rollout.py
```

Diagnostic scripts:

```text
scripts/build_voucher_state_folded_history.py
scripts/replay_state_folded_rollout.py
```

Generated validation files:

```text
data/voucher_state_folded_replay_10.jsonl
data/voucher_state_folded_replay_10_top5.jsonl
data/voucher_state_folded_10.jsonl
data/voucher_state_folded_10_sft.json
data/voucher_state_folded_10_report.json
```

## Design Principle

The key rule is:

```text
Compress historical evidence into typed state; do not compress the current action.
```

The model should still learn and produce the normal tool-call protocol. Only
the old context is folded.

## Review Notes

Several correctness risks were reviewed after the first online implementation:

- `view_product_information` is not treated as selection anymore. Viewed
  products are only verified candidates. `selected_product_ids` now comes from
  `python_execute` budget observations or `recommend_product`.
- Price and shop recovery no longer depends only on retained top-k search
  candidates. If `python_execute` returns `product_ids`, `shop_ids`,
  `total_before_voucher`, or `payable_total`, those values are accepted only
  after deterministic recomputation against observed `budget_candidates`.
- Shop voucher applicability requires complete shop information when it is
  inferred locally. This avoids falsely marking a mixed-shop selection as
  same-shop when some shop ids were missing due to candidate truncation.
- The online state now records `voucher.parse_ok` so downstream prompts can
  distinguish a parsed voucher from a failed regex parse.
- Duplicate `tool_call_id` observations are handled defensively by keeping the
  first observation for a call id instead of silently overwriting it.

Not adopted for now:

- Adding full price/shop fields to `view_product_information`. The current
  local search server does not return those fields, and changing the tool schema
  would expand every view observation. Budget state should instead come from
  search candidates or `python_execute`.
- Keeping a full history of budget calculations. This would increase context;
  the folded state keeps only the latest calculation plus `pending`, which is
  enough for the current decision.
- Keeping multiple versions of repeated `view_product_information` results for
  the same product. Later observations overwrite earlier ones; this is acceptable
  for static ShoppingBench product data and saves context.
