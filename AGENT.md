# ShoppingBench Training Flow Notes

This repository uses a unified verl-based training flow for ShoppingBench SFT
and GRPO. The key point is that SFT checkpoint selection is not driven primarily
by eval loss. SFT is treated as preparation for GRPO: the checkpoint should have
stable protocol/tool-use behavior while still preserving enough task-level
decision variance for GRPO to learn from group differences.

## 1. Unified verl Pipeline

SFT and RL are intended to run through the verl stack, instead of training SFT in
a separate external SFT path and then switching to a different RL framework.

The unified data preparation entry is:

```text
scripts/prepare_verl_shoppingbench_data.py
```

It prepares:

- SFT parquet data from state-folded teacher trajectories.
- Query-level GRPO parquet data from RL voucher queries.
- `dataset/shoppingbench_query/product_cache.json`, used by query-level outcome
  reward.

This keeps prompt format, chat template assumptions, tool protocol, and data
shape closer between SFT and GRPO.

## 2. What SFT Is Optimizing For

SFT is not selected by lowest eval loss alone. For this project, the useful SFT
checkpoint is the one that is ready for GRPO.

The SFT stage should teach the model to:

- Produce the required XML-style output format, including `<think>` and
  `<tool_call>` or `<response>`.
- Call tools legally and effectively.
- Follow the ShoppingBench task structure.
- Retain enough decision diversity that multiple rollouts for the same query can
  produce meaningful task-level differences.

The desired state is not overfit teacher imitation. If SFT collapses all
rollouts into nearly identical trajectories, GRPO has less useful group-level
signal.

## 3. SFT Checkpoint Probing

During SFT, checkpoints are saved at fixed intervals, for example every
`0.25` epoch. For each checkpoint, sample a fixed small set of RL training
queries, for example `8` queries, and run GRPO-style rollouts with group size
`G=4`.

The fixed-step probing process is:

1. Train SFT normally with teacher trajectory supervision.
2. Save checkpoints at fixed progress intervals, such as `0.25`, `0.50`,
   `0.75`, `1.00` epoch, instead of waiting only for the final checkpoint.
3. Before comparing checkpoints, choose a fixed probe query set from the RL
   training queries, for example `8` voucher-budget queries. This probe set
   should stay unchanged across all SFT checkpoints.
4. For each SFT checkpoint, load that checkpoint as the rollout policy and run
   `G` rollouts per probe query, for example `G=4`.
5. Score the resulting grouped rollouts with protocol and task metrics.
6. Select the checkpoint whose protocol is stable enough and whose task-level
   rollout variance is still high enough for GRPO.

The probe query set is fixed so checkpoint differences reflect the model state,
not changes in sampled queries. Multiple rollouts per query are needed because
GRPO learns from within-query group differences; a single rollout cannot reveal
whether the checkpoint still has useful decision variance.

The checkpoint probe logic is in:

```text
scripts/sft_grpo_probe.py
```

The probe separates protocol readiness from task-decision variance:

```text
protocol = 0.5 * format + 0.5 * tool_valid
task = progress + 2.0 * exact_success + 1.0 * budget_success - 0.02 * steps
total = protocol + task
```

Important grouped metrics:

- `protocol_mean`: whether the model has learned legal output and tool calling.
- `protocol_group_var_mean`: whether protocol behavior is stable within the same
  query group.
- `task_group_var_mean`: whether different rollouts for the same query still
  diverge in task decisions and outcomes.

The preferred checkpoint has:

```text
high protocol_mean
low protocol_group_var_mean
high task_group_var_mean
```

Practically, the probe chooses among checkpoints with stable protocol behavior
and prefers the one with stronger task-level variance. This is the checkpoint
that should enter GRPO.

## 4. GRPO Uses Query-Level Task Reward

For formal GRPO, use query-level RL:

```bash
QUERY_LEVEL_RL=1 ./src/rl/run_grpo.sh <model_path>
```

With `QUERY_LEVEL_RL=1`, `src/rl/run_grpo.sh` switches to:

```text
dataset/shoppingbench_query/train.parquet
dataset/shoppingbench_query/test.parquet
data_source = shoppingbench_query
```

It also enables async multi-turn rollout and ShoppingBench tool configuration.

The query-level reward is implemented in:

```text
src/rl/verl/utils/reward_score/shoppingbench_query.py
```

Current formal GRPO reward:

```text
score =
  task_reward
+ protocol_weight * protocol_reward

protocol_reward =
  0.5 * format_valid
+ 0.5 * tool_valid

task_reward =
  progress
+ 2.0 * exact_success
+ 1.0 * budget_success
- 0.02 * steps
```

Component meanings:

- `protocol_weight`: starts small and linearly anneals to zero, so final GRPO
  optimizes task correctness rather than protocol shaping.
- `progress`: correctness-aware dense shaping over gold search recall, selected
  gold overlap, same-shop correctness, verified selected gold products,
  recomputed budget support, actual within-budget status, recommended gold
  overlap, and terminate-after-valid-recommend.
- `exact_success`: final recommended product ids exactly match gold ids in
  order.
- `budget_success`: final recommended products are within budget after
  recomputing voucher application from observed product evidence or product
  cache.

This is whole-query task reward over complete multi-turn trajectories. It is
not the old step-level teacher-action imitation reward.

## 5. Old Step-Level Reward Path

The old step-level ToolRL reward still exists:

```text
src/rl/verl/utils/reward_score/shoppingbench_toolrl.py
```

It is used when data has:

```text
data_source = shoppingbench
```

That path scores step-level format and tool-call similarity against teacher
actions:

```text
score = format + tool_call_correctness + optional_length
```

Do not confuse this with formal query-level GRPO. The intended GRPO path for the
current training flow is:

```text
QUERY_LEVEL_RL=1
data_source=shoppingbench_query
reward=query-level task reward with protocol annealing
```

## 6. One-Sentence Summary

SFT should produce a GRPO-ready model with stable protocol/tool-use behavior and
non-collapsed task decisions; GRPO then uses query-level final outcome reward
over multiple rollouts per query to optimize ShoppingBench success.
