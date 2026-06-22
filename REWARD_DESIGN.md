# ShoppingBench Reward Design

This document records the reward that is currently used by the ShoppingBench
query-level RL pipeline.

The reward is split into two parts:

- `protocol_reward`: whether the model follows the output format and tool
  protocol.
- `task_reward`: how far the trajectory correctly advances the shopping task.

The final RL objective may include a small protocol term early in training, but
the long-term optimization target should be task correctness.

## 1. Protocol Reward

```text
protocol_reward =
  0.5 * format_valid
+ 0.5 * tool_valid
```

`format_valid` checks whether each assistant message follows the required
structure:

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
- required parameters are present
- tool observations are present and usable
- `python_execute` does not fail

Protocol reward is not task success. It only measures whether the model can play
the environment correctly.

## 2. Task Reward

```text
task_reward =
  progress
+ outcome
- penalties
```

```text
penalties =
  step_penalty
+ wrong_recommend_penalty
+ count_penalty
+ premature_terminate_penalty
+ invalid_tool_penalty
```

```text
step_penalty = 0.005 * steps
```

`steps` is the number of assistant turns. The penalty is intentionally small so
early RL exploration is not killed before the model learns to recommend and
terminate correctly.

Wrong or opportunistic recommendations are penalized:

- recommending products with zero gold overlap
- recommending too many or too few products
- terminating successfully before any recommendation
- making structurally invalid tool calls

## 3. Progress Reward

Progress reward is correctness-aware. It rewards a stage only when that stage is
supported by environment evidence and moves toward the gold ShoppingBench answer.

```text
progress =
+ 0.18 * search_gold_recall
+ 0.18 * select_gold_f1
+ 0.12 * verify_gold_f1
+ 0.08 * shop_constraint_correct
+ 0.10 * budget_attempt_quality
+ 0.12 * budget_recomputed_correct
+ 0.12 * budget_numeric_alignment
+ 0.10 * within_budget_correct
+ 0.35 * recommend_gold_f1
+ 0.12 * recommend_count_match
+ 0.25 * set_exact
+ 0.10 * terminate_quality
```

The weights no longer sum to `1.0`. The task reward is deliberately denser and
higher-variance, because GRPO needs stable within-query preference differences.
Search gets useful but limited credit; recommendation, exact set matching,
budget evidence, and valid termination carry the stronger learning signal.

### search_gold_recall

```text
search_gold_recall =
  |observed_candidate_ids intersect gold_ids| / |gold_ids|
```

This gives small credit for surfacing the correct products in search results. A
trajectory that searches but never observes gold products gets no credit here.

### select_gold_f1

```text
select_gold_f1 = F1(selected_ids, gold_ids)
```

F1 is used instead of recall so extra wrong products reduce the score. This
rewards partial correct selection without letting long product lists hack the
reward.

### verify_gold_f1

```text
verify_gold_f1 = F1(viewed_product_ids, gold_ids)
```

Viewing random products is not rewarded. Verification credit is given only for
viewing products that overlap with the gold answer, with extra wrong viewed
products penalized through precision.

### shop_constraint_correct

```text
shop_constraint_correct =
  same_shop_or_platform_ok * F1(selected_or_recommended_ids, gold_ids)
```

For shop vouchers, selected or recommended products must come from the same shop.
For platform vouchers, same-shop is not required, but this component still needs
the selected or recommended products to be relevant. Platform vouchers do not get
a free progress point before product selection.

### budget_recomputed_correct

```text
budget_recomputed_correct =
  budget_attempted
* budget_recomputation_supported
* F1(selected_or_recommended_ids, gold_ids)
```

`budget_attempted` is true only when the trajectory contains an actual
budget-like calculation, such as a `python_execute` call or state calculation
that includes `product_ids` plus budget/voucher/price/total/payable evidence.
Plain diagnostic printing is not a budget attempt.

`budget_recomputation_supported` means the reward function can independently
recompute prices, shop ids, voucher applicability, discount, payable total, and
budget comparison from environment evidence or the product cache.

Wrong product sets receive no budget recomputation credit even if their prices
are known.

### budget_numeric_alignment

```text
budget_numeric_alignment =
  soft_match(model_claimed_total, recomputed_total)
  and/or soft_match(model_claimed_payable, recomputed_payable)
  and/or match(model_claimed_within_budget, recomputed_within_budget)
```

The reward reads budget-like `python_execute` outputs or compressed state. A
numeric value receives full credit when it is exact or within 1%, half credit
when it is within 5%, and zero otherwise. This makes budget calculation
learnable before the model reaches exact final success.

### within_budget_correct

```text
within_budget_correct =
  1.0 if recomputed_payable_total(selected_or_recommended_ids) <= budget else 0.0
```

This value is multiplied by `F1(selected_or_recommended_ids, gold_ids)`. An
affordable but irrelevant product set therefore receives zero credit.

### recommend_gold_f1

```text
recommend_gold_f1 = F1(recommended_ids, gold_ids)
```

This is the main final-product shaping signal before exact success. F1 penalizes
both missing gold products and recommending extra wrong products.

### recommend_count_match

```text
recommend_count_match =
  recommend_gold_f1 if len(recommended_ids) == len(gold_ids) else 0.0
```

This rewards trajectories that recommend the expected number of products. It
helps distinguish a one-correct-item partial answer from a count-correct partial
answer.

### set_exact

```text
set_exact =
  set(recommended_ids) == set(gold_ids)
  and len(recommended_ids) == len(gold_ids)
```

Order is not required for this reward. ShoppingBench recommendations are product
sets; using set equality gives a useful success signal even if the model orders
the correct products differently.

### terminate_quality

```text
terminate_quality =
  recommend_gold_f1 if terminate_success and recommended_ids else 0.0
```

Terminate receives partial credit only when it follows a non-empty
recommendation with gold overlap. Terminating without a recommendation is
penalized.

## 4. Outcome Reward

Outcome reward is still strict, but it is now staged:

```text
exact_success = set_exact
```

```text
budget_success =
  exact_success
  and recomputed_payable_total(recommended_ids) <= budget
```

Budget success is gated by exact product success. This prevents wrong but cheap
recommendations from receiving outcome credit.

```text
final_success =
  budget_success
  and terminate_success
```

```text
outcome =
  0.25 * exact_success
+ 0.50 * budget_success
+ 1.50 * final_success
```

The model receives progressively larger outcome credit for exact product set,
budget-valid set, and fully terminated success.

## 5. State-Folded Rollout Support

The reward parser supports both structured rollout messages and saved
state-folded rollout text. For saved text, it can read:

- assistant `<tool_call>...</tool_call>` blocks
- compressed user `<state>...</state>` snapshots

The state snapshots are used as product evidence for search candidates, viewed
products, selected product ids, recommendations, terminations, voucher state, and
budget candidates.

This matters because eval rollout logs intentionally omit raw `<obs>` while
keeping compressed state.

## 6. GRPO Reward Schedule

During formal GRPO, the scalar reward can include a small protocol component at
the beginning:

```text
grpo_reward =
  alpha(t) * protocol_reward
+ task_reward
```

`alpha(t)` should anneal to zero:

```text
alpha(t) = max(0, alpha0 * (1 - step / warmup_steps))
```

Current RL default:

```text
alpha0 = 0.0
warmup_steps = 0
```

Protocol reward is disabled by default because SFT already learned most format
and tool legality. The next RL run should optimize task reward directly.

Earlier experimental schedule:

```text
grpo_reward = 0.2 * protocol_reward + task_reward
```

Late GRPO:

```text
grpo_reward = task_reward
```

The protocol term is useful to prevent format/tool-call collapse. It should not
be the final objective.

## 7. Validation

This reward was tested on the real state-folded SFT checkpoint rollout:

```text
rollouts/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029
```

The final audit artifacts are:

```text
plots/task_reward_audit_statefolded_20260620_final.json
plots/task_reward_audit_statefolded_20260620_final.csv
```

Audit result:

```text
rows: 256
wrong_final_high_count: 0
long_list_high_count: 0
partial_signal_lost_count: 0
old_wrong_final_high_count: 5
```

The old reward had five trajectories where the final recommendation had zero
gold overlap but still received high task reward. The current reward removes
those false positives while preserving positive shaping for correct partial
search, selection, verification, budget calculation, and recommendation steps.
