# ShoppingBench Reward Design

This note records the current reward design for SFT checkpoint probing and
query-level GRPO training.

The central distinction is:

- `protocol_reward` measures whether the model can follow the output and tool
  protocol.
- `task_reward` measures whether the model is making correct task progress and
  eventually reaches the correct outcome.

SFT checkpoint selection uses both. Formal GRPO may use a small amount of
protocol reward at the beginning, but it should anneal that weight to zero so
the final optimization target is task correctness.

## 1. Protocol Reward

Protocol reward checks whether the model is able to play the environment
properly.

```text
protocol_reward =
  0.5 * format_valid
+ 0.5 * tool_valid
```

`format_valid` checks whether assistant output follows the required XML-style
format, for example:

```text
<think>...</think>
<tool_call>[...]</tool_call>
```

or:

```text
<think>...</think>
<response>...</response>
```

`tool_valid` checks whether tool calls are structurally legal and executable:

- tool names are in the allowed ShoppingBench tool set
- `parameters` is a JSON object
- tool observations are present and usable
- `python_execute` does not fail

Protocol reward should not be treated as final task success. It only says the
model can produce legal actions in the environment.

## 2. Correctness-Aware Progress

Progress reward should not merely reward whether a stage was attempted. It must
verify that each stage moves toward the correct answer.

The proposed progress reward is:

```text
progress =
  0.10 * search_gold_recall
+ 0.20 * select_gold_overlap
+ 0.10 * same_shop_correct
+ 0.15 * verify_selected_gold
+ 0.15 * budget_recomputed_correct
+ 0.10 * within_budget_correct
+ 0.10 * recommend_gold_overlap
+ 0.10 * terminate_after_valid_recommend
```

The weights sum to `1.0`.

### search_gold_recall

Checks whether search observations contain the gold product ids.

```text
search_gold_recall =
  |observed_candidate_ids ∩ gold_ids| / |gold_ids|
```

This is different from rewarding any `find_product` call. A search that never
surfaces the correct products should not receive high progress.

### select_gold_overlap

Checks whether selected product ids overlap with gold product ids.

```text
select_gold_overlap =
  |selected_ids ∩ gold_ids| / |gold_ids|
```

Selecting the correct number of products is not enough. Wrong selected products
should receive little or no credit here.

### same_shop_correct

Checks the shop constraint.

For platform vouchers, this component is `1.0` because same-shop is not needed.

For shop vouchers, selected or recommended products must come from the same
shop:

```text
same_shop_correct = 1.0 if len(selected_shop_ids) == 1 else 0.0
```

Shop ids must come from observed candidates or the product cache. Do not trust a
model-written shop id without verification.

### verify_selected_gold

Checks whether the model verified selected products that are also gold products.

```text
verify_selected_gold =
  |viewed_ids ∩ selected_ids ∩ gold_ids| / |gold_ids|
```

Viewing arbitrary products should not be rewarded as much as verifying products
that matter for the correct answer.

### budget_recomputed_correct

Checks whether budget calculation can be independently verified.

The reward should not trust the model's `python_execute` result by itself. It
should recompute totals from observed prices/shop ids or from the product cache:

- total before voucher
- voucher applicability
- discount amount
- payable total
- budget comparison

This component should be high only when the budget calculation is supported by
environment evidence.

### within_budget_correct

Checks whether the selected or recommended products are actually within budget
after recomputing voucher application.

```text
within_budget_correct = 1.0 if recomputed_payable_total <= budget else 0.0
```

This differs from `budget_recomputed_correct`: one checks whether the
calculation is trustworthy, the other checks whether the result satisfies the
budget constraint.

### recommend_gold_overlap

Checks whether final recommended product ids overlap with gold ids.

```text
recommend_gold_overlap =
  |recommended_ids ∩ gold_ids| / |gold_ids|
```

For shaping, overlap is useful because exact success may be sparse. Exact
matching is handled separately in `outcome`.

### terminate_after_valid_recommend

Checks whether the model terminates only after a meaningful recommendation.

A permissive version:

```text
terminate_after_valid_recommend =
  1.0 if terminate_success and recommend_gold_overlap > 0 else 0.0
```

A stricter version can require exact final success before giving terminate
credit. The permissive version is more useful for dense shaping, while the
strict version is safer against premature `terminate(success)`.

## 3. Outcome Reward

Outcome reward is the sparse final-success signal.

Recommended first version:

```text
outcome =
  2.0 * exact_success
+ 1.0 * budget_success
```

`exact_success`:

```text
recommended_ids == gold_ids
```

The comparison should preserve order when the task requires products in request
order.

`budget_success`:

```text
recomputed_payable_total <= budget
```

The payable total must be recomputed from environment evidence or product cache,
not accepted from model text alone.

An even stricter binary outcome can also be defined:

```text
final_success =
  exact_success
  and budget_success
  and terminate_success
```

Then:

```text
outcome = 3.0 * final_success
```

The weighted exact-plus-budget version gives more gradient signal when product
selection is right but termination or minor protocol details are imperfect.

## 4. Task Reward

The task reward combines correctness-aware progress, final outcome, and a small
step penalty.

```text
task_reward =
  progress
+ outcome
- step_penalty
```

Recommended step penalty:

```text
step_penalty = 0.02 * steps
```

Expanded recommended version:

```text
task_reward =
  progress
+ 2.0 * exact_success
+ 1.0 * budget_success
- 0.02 * steps
```

The step penalty prevents the model from endlessly searching, viewing, or
recomputing to collect shaping reward.

## 5. SFT Checkpoint Probe

For each fixed SFT checkpoint, run a fixed probe query set. For example:

```text
8 probe queries
G = 4 rollouts per query
32 total rollouts per checkpoint
```

For every rollout, compute:

```text
protocol_reward
task_reward
```

Then group rollouts by query and compute:

```text
protocol_mean
protocol_group_var_mean
task_group_var_mean
```

Selection criterion:

```text
high protocol_mean
low protocol_group_var_mean
high task_group_var_mean
```

Interpretation:

- high `protocol_mean`: the checkpoint can produce legal format and tool calls
- low `protocol_group_var_mean`: protocol behavior is stable
- high `task_group_var_mean`: task decisions still branch enough for GRPO to
  learn from within-query reward differences

The probe query set must be fixed across checkpoints. Otherwise differences may
come from query sampling noise rather than checkpoint quality.

## 6. GRPO Reward Schedule

During formal GRPO, the reward can include a small protocol component early in
training:

```text
grpo_reward =
  alpha(t) * protocol_reward
+ task_reward
```

`alpha(t)` should anneal to zero:

```text
alpha(t) = max(0, alpha0 * (1 - step / warmup_steps))
```

Example:

```text
alpha0 = 0.3
warmup_steps = first 10% of GRPO steps
```

Early GRPO:

```text
grpo_reward = 0.3 * protocol_reward + task_reward
```

Late GRPO:

```text
grpo_reward = task_reward
```

The reason for annealing is that protocol shaping is useful at the beginning to
avoid format and tool-call collapse, but the final objective should be task
correctness, not formatting.

## 7. Design Principle

Probe reward may inspect both protocol and task variance. Formal GRPO reward
should prioritize correctness.

Do not reward a stage merely because it happened. Reward it only when it is
supported by evidence and moves toward the correct ShoppingBench outcome.
