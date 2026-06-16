# Coupon & Budget Sampling Plan

This document defines the final plan for generating the ShoppingBench
Coupon & Budget training queries. The goal is to preserve the original
ShoppingBench task definition and output schema while adding small,
controlled improvements in sampling balance, difficulty coverage, quality
validation, and generation speed.

## Target Output

Final output file:

```text
data/synthesize_voucher_train.jsonl
```

Each line must match the existing `synthesize_voucher_test.jsonl` schema:

```json
{"query": "...", "reward": [...], "voucher": {...}}
```

No debug fields, stage fields, sample IDs, prompts, or metadata should be
written into the final JSONL. Auxiliary information should be written to
separate plan, log, or metadata files.

## Dataset Size

Use:

```text
750 Coupon & Budget training queries
```

Rationale: the ShoppingBench paper reports 1000 Coupon & Budget instructions
with 250 used as test data, implying 750 training instructions. Keeping 750
preserves comparability with the original benchmark setting.

## Three-Stage Pipeline

The generation process follows the ShoppingBench paper's three stages.

### Stage I: Product and Voucher Sampling

Sample real products from the shopping sandbox, then synthesize voucher rules
that are satisfiable by the sampled product set.

Keep the original task semantics:

- `platform` voucher applies to all products.
- `shop` voucher requires all selected products to come from the same shop.
- Product IDs must be globally deduplicated across the generated train set.
- Coupon & Budget tasks should not expose product prices in the user query.

### Stage II: Product Field Sampling

For each sampled product:

- Always include `title`.
- Sample additional constraints from `sku_options`, `attributes`, and `service`.
- Exclude `price`.
- Keep most products at 2-4 total fields.
- Allow a small number of harder examples with 5 or more fields.

### Stage III: User Query Simulation

Use the LLM to convert sampled product fields into a natural user query, then
append budget and voucher rules in the same style as the official
ShoppingBench data:

```text
My budget is only `...`, but I have a voucher with the following rules:
1. ...
2. ...
3. ...
```

## Sampling Hyperparameters

### Product Count Distribution

Match the paper's Coupon & Budget distribution as closely as possible:

```text
1 product :  8.8% ->  66 examples
2 products: 30.5% -> 229 examples
3 products: 31.7% -> 238 examples
4 products: 29.0% -> 217 examples
total     :        -> 750 examples
```

Rationale: pure random sampling can drift noticeably at 750 examples. Fixed
quotas keep the generated train set close to the paper's reported distribution.

### Voucher Type Distribution

Use:

```text
platform: 40% -> 300 examples
shop    : 60% -> 450 examples
```

Rationale: `shop` vouchers add a same-shop constraint, which is one of the
main reasoning challenges in Coupon & Budget tasks. Raising `shop` to 60%
adds more training signal while still preserving substantial platform-voucher
coverage.

### Discount Type Distribution

Use:

```text
fixed     : 40% -> 300 examples
percentage: 60% -> 450 examples
```

Rationale: percentage discounts with caps require more careful arithmetic than
fixed discounts:

```text
discounted_total = max(total * (1 - discount), total - cap)
```

Increasing percentage discounts to 60% makes the dataset slightly more
reasoning-heavy without changing the core task.

### Budget Slack Distribution

Define budget slack relative to `price_after_voucher`:

```text
hard  : 35% -> budget = ceil(price_after_voucher * 1.00-1.02)
medium: 45% -> budget = ceil(price_after_voucher * 1.02-1.06)
easy  : 20% -> budget = ceil(price_after_voucher * 1.06-1.10)
```

Rationale: boundary-budget examples force the agent to actually check totals
instead of relying on approximate price intuition.

### Voucher Threshold Distribution

Sample threshold as a ratio of pre-discount total price:

```text
low : 25% -> threshold = total_price * 0.20-0.50
mid : 50% -> threshold = total_price * 0.50-0.80
high: 25% -> threshold = total_price * 0.80-0.95
```

Rationale: higher thresholds better test whether the selected product
combination satisfies the voucher's minimum-spend requirement.

### LLM Query Generation Parameters

Use Xiaomi MiMo through the OpenAI-compatible API:

```text
model: mimo-v2.5
base_url: https://token-plan-cn.xiaomimimo.com/v1
temperature: 0.35
top_p: 0.8
max_completion_tokens: 512
thinking: disabled
request_timeout: 60 seconds
retries: 2
```

Rationale: `temperature=0.35` is high enough to improve natural-language
diversity, but low enough to reduce missing constraints or malformed JSON.

## Quality Validation

Every generated item must pass validation before it is accepted into
`synthesize_voucher_train.jsonl`.

Required checks:

- Top-level keys are exactly `query`, `reward`, and `voucher`.
- `query` is non-empty and contains the appended budget/voucher rules.
- `reward` is a non-empty list.
- `voucher` contains all required fields:
  - `voucher_type`
  - `threshold`
  - `discount_type`
  - `face_value`
  - `discount`
  - `cap`
  - `price_after_voucher`
  - `budget`
- `price_after_voucher <= budget`.
- Product IDs are globally unique across the final train set.
- For `shop` vouchers, sampled products come from one shop.
- LLM output must parse as JSON and contain a non-empty `query`.
- Reject outputs where the query is only the voucher/budget suffix.

## Speed and Reliability Improvements

### 1. Separate Sampling from LLM Generation

First build a deterministic plan file:

```text
data/synthesize_voucher_train_plan.jsonl
```

The plan should contain sample ID, selected products, sampled fields, prompt,
reward, voucher, and sampling buckets. This file is not used as final training
data; it exists for reproducibility and resume support.

### 2. Parallelize Stage III

Generate LLM queries from the plan with controlled concurrency.

Recommended default:

```text
concurrency: 4
```

If the endpoint rate-limits or becomes unstable, reduce to:

```text
concurrency: 2
```

Expected impact: if serial generation takes about 2-3 seconds per query,
750 examples would take roughly 25-40 minutes. With 4-way concurrency, expected
runtime should drop to roughly 8-12 minutes, depending on API latency and
retry rate.

### 3. Resume Support

The generation script should:

- Read existing output if present.
- Skip already completed sample IDs.
- Continue from failed or missing samples.
- Flush each accepted JSONL line immediately.

### 4. Metadata and Logs

Write a separate metadata file:

```text
data/synthesize_voucher_train.meta.json
```

Include:

- Seed
- Total accepted examples
- Product count distribution
- Voucher type distribution
- Discount type distribution
- Budget slack distribution
- Threshold bucket distribution
- LLM failure count
- Retry count
- Validation rejection count

Logs should go to:

```text
logs/
```

## Recommended Execution Flow

1. Generate a small probe set, for example 2-5 examples.
2. Validate schema and inspect query quality manually.
3. Generate the full 750-example plan.
4. Run concurrent LLM query generation.
5. Validate the final `synthesize_voucher_train.jsonl`.
6. Save metadata and logs.

## Final Position

The final setup keeps the original ShoppingBench Coupon & Budget semantics and
output format, while adding a small amount of deliberate dataset design:

- 750 examples for paper comparability.
- Product-count quotas matching the paper.
- More `shop` voucher examples for same-shop reasoning.
- More `percentage` discount examples for arithmetic reasoning.
- Boundary-budget and high-threshold examples for harder budget checks.
- Plan/generate separation, concurrency, resume support, and validation for
  speed and reliability.
