# Teacher Voucher Hard20 Notes

Built with online `history_compression=state_folded`; no model API was called.

## Current Setting

- `state_max_candidates_per_search=10`.
- `workers=2` case-level parallelism.
- Hard samples are selected from `synthesize_voucher_train.jsonl` by budget tightness: `budget <= price_after_voucher * 1.02`.

## Trajectory Pattern

- Search user-request terms until target candidates are visible in retained search state.
- For shop vouchers, use an observed same-shop candidate as an anchor and search remaining items with `shop_id`.
- Use explicit service filters such as `flashsale,freeShipping` only when the user requested them.
- Verify product details and compute voucher budget from observed prices/shop ids.
- Recommend verified products in request order and terminate.

## Compression Observations

- Step 1 has no `<state>` because no assistant/tool history exists yet.
- Top10 is safer than top5 because it preserves a full original search page as readable state.
- `budget_candidates` is enough for budget verification but not enough for product selection teaching; ids used by `view_product_information` should appear in `searches[].candidates`.
- `python_execute` must print strict JSON with product ids, shop ids, totals, `voucher_used`, and `within_budget`.

## Validation Results

- Official voucher eval on 20 trajectories: all metrics 1.000.
- Step count: 16 trajectories use 3 steps; 4 trajectories use 4 steps.
- State-support check: 54 view ids, 54 recommend ids, and 5 same-shop search ids were all supported by compressed state.
- Every recommendation step had `budget_calculation_trusted=true`.
- All 64 `<think>` strings are unique after the concrete-thought rewrite.
- Product ids in `<think>` appear only after they are visible in the current prompt/state.
- Qwen3-4B tokenizer maxima after the rewrite: step prompt 7845, step assistant output with EOS 398.

## Parallelism Notes

- Case-level parallelism is safe because trajectories are independent.
- The parent process writes JSONL after workers finish, ordered by original case index.
- `workers=8` worked against the local search server for this hard20 batch.

## Teacher Style Constraints

- `<think>` is short but concrete: search terms, visible candidate ids, shop anchor, observed prices, and trusted budget totals appear when available from state.
- Gold answers are used only by the builder to choose actions; prompts and thoughts do not mention reward metadata.
- Product ids appear only after environment search observations make them visible.
