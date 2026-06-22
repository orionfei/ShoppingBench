# Task Reward Exploration

Date: 2026-06-20

This note records the task-reward redesign process after auditing the real
state-folded SFT checkpoint rollout:

- Rollout root:
  `rollouts/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029`
- Probe ground truth:
  `dataset/probe/sft_probe_query_8_statefolded_20260620.parquet`
- Final audit outputs:
  `plots/task_reward_audit_statefolded_20260620_final.json`
  and `plots/task_reward_audit_statefolded_20260620_final.csv`

## Goal

The task reward should measure how far a trajectory has correctly advanced the
ShoppingBench voucher task. It should reward correct intermediate work such as
finding gold products, selecting relevant products, checking budget constraints,
and recommending the right products. It should not reward wrong products merely
because the model made tool calls or found an affordable but irrelevant set.

Protocol reward remains separate. Protocol reward measures format and tool
legality; task reward measures task correctness.

## Baseline Problem

The old task reward was:

```text
task = progress + 2.0 * exact_success + 1.0 * budget_success - 0.02 * steps
budget_success = recomputed_recommended_payable_total <= budget
```

The problem was that `budget_success` did not require the recommended products to
match the gold products. Therefore wrong recommendations could receive a large
positive outcome if they were cheap enough.

Concrete false positives from the real rollout:

| trajectory | old task | final recommendation | gold ids | issue |
| --- | ---: | --- | --- | --- |
| `global_step_256 row28` | `1.29` | `2323444515` | `5074851046,4835851287` | completely wrong product, within budget |
| `global_step_32 row29` | `1.39` | `3688591923` | `5074851046,4835851287` | found gold in search but recommended wrong product |
| `global_step_64 row10` | `1.2233` | `4362876449,276273617` | `4876810268,3423715293,4855586021` | wrong final products |
| `global_step_128 row5` | `1.2233` | `4235137419,3342710507` | `4234705587,4485541181,4417572249` | wrong final products |
| `global_step_192 row8` | `1.2233` | `276273617,748050052,747968837` | `4876810268,3423715293,4855586021` | wrong final products |

The old reward also used recall-like `recommend_gold_overlap`, so long
recommendation lists could receive partial credit by including some gold products
among many wrong products.

## Tested Reward Versions

### v1: Gate Budget Outcome

First attempt:

- require product correctness before giving final budget outcome
- keep most old progress components

Result on 256 real rollout trajectories:

- wrong-final high score count: `0`
- long-list high score count: `3`
- max task: `0.49`

This removed the largest budget false positives, but long lists still received
too much reward because the recommendation component was recall-based.

### v2: F1-Based Progress With Budget Relevance Gates

Second attempt:

- use F1 for selected and recommended products
- gate shop, budget recomputation, and within-budget shaping by product
  relevance
- require explicit budget-like `python_execute` or state budget calculation for
  budget recomputation credit
- keep within-budget shaping only for relevant selected/recommended product sets
- make sparse outcome strict final success

Offline result before implementation:

- wrong-final high score count: `0`
- long-list high score count: `0`
- partial-signal-lost count: `0`
- max task: about `0.3733`

This version kept useful dense signal but stopped rewarding wrong final products.

### Final Implementation

The final implementation is v2 with two parser fixes:

- state-folded `<state>...</state>` text is parsed as product evidence, so the
  reward can be replayed from saved rollout text.
- `python_execute` counts as a budget attempt only when its code/output contains
  budget-like evidence such as `product_ids` plus `budget`, `voucher`, `total`,
  `payable`, `price`, `discount`, or similar fields. Plain diagnostic printing
  no longer receives budget credit.

Final full-rollout audit:

```text
rows: 256
new_task_mean: 0.031122364614552114
new_task_min: -0.08
new_task_max: 0.37333333333333335
wrong_final_high_count: 0
long_list_high_count: 0
partial_signal_lost_count: 0
old_wrong_final_high_count: 5
success_count: 0
exact_count: 0
budget_success_count: 0
```

The `success_count`, `exact_count`, and `budget_success_count` are zero because
none of these 256 SFT checkpoint probe rollouts exactly solved a task. This is
expected and is why dense progress shaping is still needed.

I also ran a positive exact-success sanity check by constructing a trajectory
that recommends the ordered gold ids for the first probe query and then calls
`terminate(success)`. The new reward returned:

```text
exact: 1.0
budget: 1.0
success: 1.0
outcome: 3.0
task: 3.6
```

This verifies that the final sparse signal remains strong when the trajectory
actually solves the task.

## Final Task Reward

```text
task_reward =
  progress
+ 3.0 * final_success
- 0.02 * steps
```

Progress:

```text
progress =
  0.12 * search_gold_recall
+ 0.16 * select_gold_f1
+ 0.10 * verify_gold_f1
+ 0.10 * shop_constraint_correct
+ 0.16 * budget_recomputed_correct
+ 0.10 * within_budget_correct
+ 0.20 * recommend_gold_f1
+ 0.06 * recommend_count_match
```

Final success:

```text
exact_success =
  recommended_ids == gold_ids

budget_success =
  exact_success
  and recomputed_payable_total(recommended_ids) <= budget

final_success =
  budget_success
  and terminate_success
```

## Component Semantics

`search_gold_recall`:

```text
|observed_candidate_ids intersect gold_ids| / |gold_ids|
```

Search gets small reward only when the correct products actually appear in
observed candidates.

`select_gold_f1`:

```text
F1(selected_ids, gold_ids)
```

This rewards partial correct selection while penalizing extra wrong selected
products.

`verify_gold_f1`:

```text
F1(viewed_product_ids, gold_ids)
```

Viewing arbitrary products is not enough; viewed products must overlap with gold.

`shop_constraint_correct`:

```text
same_shop_or_platform_ok * F1(selected_or_recommended_ids, gold_ids)
```

The shop constraint only gives credit for relevant product sets. Platform
vouchers do not get a free point before the model selects relevant products.

`budget_recomputed_correct`:

```text
budget_attempted
* budget_recomputation_supported
* F1(selected_or_recommended_ids, gold_ids)
```

The model must actually attempt a budget calculation, and the calculated product
set must be relevant. Wrong products do not receive budget credit just because
their prices are known.

`within_budget_correct`:

```text
recomputed_selected_or_recommended_payable_total <= budget
multiplied by F1(selected_or_recommended_ids, gold_ids)
```

An affordable wrong product set receives zero credit.

`recommend_gold_f1`:

```text
F1(recommended_ids, gold_ids)
```

This replaces recall-only overlap and prevents long-list reward hacking.

`recommend_count_match`:

```text
recommend_gold_f1 if len(recommended_ids) == len(gold_ids) else 0.0
```

This gives extra shaping only when the model recommends the expected number of
products.

## Validation Examples

| trajectory | old task | new task | interpretation |
| --- | ---: | ---: | --- |
| `global_step_256 row28` | `1.29` | `-0.06` | wrong product, no correct progress |
| `global_step_32 row29` | `1.39` | `0.06` | found gold in search, final recommendation wrong |
| `global_step_32 row13` | `1.59` | `0.3733` | one of two products correct, no exact success |
| `global_step_128 row20` | `0.515` | `0.31` | partial correct view/selection/budget attempt, no final recommendation |
| `global_step_32 row4` | `0.4567` | `0.1308` | long list with partial gold overlap, extra wrong products penalized |

## Files Changed

- Training reward implementation:
  `src/rl/verl/utils/reward_score/shoppingbench_query.py`
- Reproducible audit script:
  `scripts/audit_task_reward_on_rollouts.py`
- Final audit artifacts:
  `plots/task_reward_audit_statefolded_20260620_final.json`
  and `plots/task_reward_audit_statefolded_20260620_final.csv`
