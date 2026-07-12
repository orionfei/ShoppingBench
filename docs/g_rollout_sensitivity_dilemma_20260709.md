# G Rollout Sensitivity Dilemma

Date: 2026-07-09

## 2026-07-10 Causal Conclusion (Supersedes The Earlier Hypothesis)

The root cause has now been reproduced and fixed with controlled experiments on
`global_step_108`, using two A800 GPUs, temperature `0.2`, top-p `0.9`, maximum
response length `10240`, and maximum assistant turns `15`.

The failure is not caused by checkpoint initialization, GRPO group math, or vLLM
generating `G` candidates inside each request. `G` changes the number of outer
trajectory requests submitted in the same validation wave. In the async vLLM
server integration, `rollout.max_num_seqs` existed in Hydra and the launch scripts,
but it was never forwarded to `AsyncEngineArgs`. Consequently, the intended
scheduler concurrency limit had no effect. A large `G`/validation batch could put
far more live sequences into each vLLM server than this long, multi-turn,
tool-calling workload handled reliably, producing malformed and runaway outputs.

This also explains why the GPU count matters. With tensor parallel size 1:

- four GPUs create four rollout servers;
- two GPUs create two rollout servers;
- a 64-trajectory wave is therefore approximately 16 initial trajectories per
  server on four GPUs, but 32 per server on two GPUs.

Moving from four cards to two cards did not change model initialization, but it
doubled the request pressure seen by each rollout engine. It can therefore make the
same `G=8` problem more likely or more severe.

### Controlled result matrix

All rows below use the same checkpoint and sampling parameters. `VB` is validation
batch size. `maxseq` means the value was actually forwarded to vLLM, rather than
merely appearing in the config.

| Run | Trajectories in one wave | Engine change | Format | Protocol | Task | Success |
|---|---:|---|---:|---:|---:|---:|
| `G=8, VB=1` | 8 | legacy/unlimited | 1.000 | 0.845 | 0.182 | 20.3% |
| `G=4, VB=2`, offset 0 | 8 | legacy/unlimited | 1.000 | 0.852 | 0.177 | 12.5% |
| `G=4, VB=2`, offset 4 | 8 | legacy/unlimited | 1.000 | 0.819 | 0.085 | 12.5% |
| `G=8, VB=8` | 64 | legacy/no explicit cap | 0.518 | 0.507 | 0.060 | 6.3% |
| `G=8, VB=8` | 64 | unlimited, `enforce_eager=True` | 0.722 | 0.669 | 0.085 | 7.8% |
| `G=8, VB=8` | 64 | unlimited, prefix cache off | 0.984 | 0.825 | 0.074 | 14.1% |
| `G=8, VB=8` | 64 | `maxseq=4` | 1.000 | 0.845 | 0.178 | 15.6% |
| `G=8, VB=8` | 64 | `maxseq=8` | 1.000 | 0.863 | 0.207 | 20.3% |
| `G=4, VB=8` | 32 | `maxseq=8` | 1.000 | 0.851 | 0.153 | 12.5% |

The decisive comparison is `G=8, VB=8` unlimited versus the same 64-request wave
with `maxseq=8`: format recovers from `0.518` to `1.000`, protocol from `0.507` to
`0.863`, task score from `0.060` to `0.207`, and success from `6.3%` to `20.3%`.
No checkpoint, prompt, seed set, temperature, top-p, response limit, `G`, validation
batch, or number of submitted trajectories changed.

The equal-concurrency experiment also resolves the earlier observation that two
`G=4` runs were much better than one `G=8` run. When `G=8, VB=1` and `G=4, VB=2`
both submit only eight trajectories per wave, the `G=8` run no longer collapses.
The two `G=4` runs remain statistically different samples, but their combined
quality distribution is comparable; all 64 outputs are format-valid. Thus the old
comparison accidentally changed both group size and instantaneous engine load.

`enforce_eager=True` partially helps, so CUDA Graph/batch-shape effects amplify the
problem, but they are not sufficient to explain it. Disabling prefix caching almost
restores protocol validity, showing that prefix-cache scheduling under excessive
concurrency is another major amplifier, but task score remains poor. The robust fix
is to bound vLLM scheduler concurrency while leaving prefix caching enabled.

### Implemented, fallback-safe fix

The async server now supports two explicit options:

- `actor_rollout_ref.rollout.apply_max_num_seqs`: when `True`, forward
  `rollout.max_num_seqs` into `AsyncEngineArgs`;
- `actor_rollout_ref.rollout.enable_prefix_caching`: control prefix caching for
  diagnostics.

The implementation retains an explicit legacy fallback. After the G=4 protection
run confirmed no regression, the project launchers were standardized on
`apply_max_num_seqs=True`, `max_num_seqs=8`, and `enable_prefix_caching=True`.
Setting `ROLLOUT_APPLY_MAX_NUM_SEQS=False` remains the exact fallback for this fix.

Recommended two-A800 settings for the next `G=8` run:

```bash
ROLLOUT_APPLY_MAX_NUM_SEQS=True
ROLLOUT_MAX_NUM_SEQS=8
ROLLOUT_ENABLE_PREFIX_CACHING=True
ROLLOUT_ENFORCE_EAGER=False
STABLE_ROLLOUT_SAMPLING=True
STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True
```

`maxseq=8` is preferred over `4` because it had the best measured quality while
retaining more rollout throughput. The protective `G=4, VB=8, maxseq=8` run also
shows that enabling the fix does not sacrifice the currently good `G=4` behavior.
Do not reduce the RL optimizer's train batch solely to fix this generation issue;
the engine can queue excess rollout requests behind the scheduler limit without
changing GRPO grouping or optimizer batch semantics.

The reproducible runner is `scripts/run_g_root_cause_probe_20260710.sh`; all raw
reports are under `reports/g_root_step108_2gpu_*_20260710/`.

## Background

We are evaluating the SFT checkpoint `global_step_108` from:

`checkpoints/shoppingbench-sft/sft_clean924_prefix_len10240_full_3ep_20260709_043231/global_step_108`

The intended evaluation setup is:

- Test split: original 8-query probe or 16-query test probe.
- Rollout backend: verl async multi-turn agent loop with vLLM.
- Max assistant turns: 15.
- Response length: 10240.
- Temperature: 0.2.
- Top-p: originally 0.9 for the comparable runs.
- `ROLLOUT_N` / `G`: expected to only change the number of samples per query.

The current problem is that changing `G` appears to change model behavior much more than expected. In principle, a single `G=8` run over 8 queries should be statistically comparable to two independent `G=4` runs over the same 8 queries. It is not.

## Observed Results

### Original 8-query `G=4`

Report:

`reports/sft_clean924_test8_g4_t02_normal15_20260709_070720/global_step_108.json`

Summary:

- Rows: 32.
- Queries: 8.
- Group size: 4.
- Success: 5 / 32 = 15.6%.
- Protocol mean: 0.873.
- Format mean: 1.0.
- Tool-valid mean: 0.746.
- Progress mean: 0.368.
- Mean steps: 6.156.
- Failure modes:
  - `success`: 5
  - `protocol_invalid`: 16
  - `search_recall_gap`: 6
  - `workflow_invalid`: 4
  - `final_selection_after_full_recall_gap`: 1

### Single 8-query `G=8`

Report:

`reports/sft_clean924_step108_test8_g8_t02_top_p09_normal15_20260709/global_step_108.json`

Rollout:

`rollouts/sft_clean924_step108_test8_g8_t02_top_p09_normal15_20260709/global_step_108/0.jsonl`

Summary:

- Rows: 64.
- Queries: 8.
- Group size: 8.
- Success: 0 / 64.
- Protocol mean: 0.277.
- Format mean: 0.173.
- Tool-valid mean: 0.380.
- Progress mean: 0.035.
- Mean steps: 2.078.
- Max output length in the report reached about 80k characters.
- Failure modes:
  - `protocol_invalid`: 58
  - `search_recall_gap`: 4
  - `workflow_invalid`: 1
  - `no_recommendation`: 1

This is the anomalous run. The model produced many malformed or runaway outputs, including repeated JSON-ish fragments and long repeated numeric strings.

### Control: two independent 8-query `G=4` runs

Combined report:

`reports/sft_clean924_step108_test8_g4x2_t02_top_p09_normal15_20260709/compact_summary.json`

Individual reports:

- `reports/sft_clean924_step108_test8_g4x2_t02_top_p09_normal15_20260709_rep1/global_step_108.json`
- `reports/sft_clean924_step108_test8_g4x2_t02_top_p09_normal15_20260709_rep2/global_step_108.json`

Summary:

- Rows: 64.
- Split: 2 independent runs x `G=4`.
- Success: 8 / 64 = 12.5%.
- Protocol mean: 0.823.
- Format mean: 1.0.
- Tool-valid mean: 0.646.
- Progress mean: 0.243.
- Mean steps: 6.766.
- Failure modes:
  - `success`: 8
  - `protocol_invalid`: 45
  - `search_recall_gap`: 6
  - `workflow_invalid`: 5

This control is close to the original `G=4` behavior and very far from the single `G=8` behavior. That strongly suggests the issue is not simply that the checkpoint is weak. Something about the single `G=8` execution path changes the rollout distribution.

## What `G` Changes In The Current Code

The launch script writes `ROLLOUT_N` into two config fields:

- `actor_rollout_ref.rollout.n`
- `actor_rollout_ref.rollout.val_kwargs.n`

Reference:

`src/rl/run_grpo_qwen3_1_7b_query_verl.sh`

Relevant lines:

- `actor_rollout_ref.rollout.n="$ROLLOUT_N"`
- `actor_rollout_ref.rollout.val_kwargs.n="$ROLLOUT_N"`

In validation, the actual outer repeat uses `val_kwargs.n`:

`src/rl/verl/trainer/ppo/ray_trainer.py`

The validation batch is repeated with:

```python
test_batch = test_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
    interleave=True
)
```

Therefore:

- 8 queries with `G=4` become 32 validation rows.
- 8 queries with `G=8` become 64 validation rows.

The repeat is interleaved by query:

`q1, q1, q1, q1, ..., q2, q2, ...`

This comes from:

`src/rl/verl/protocol.py`

where `repeat_interleave` and `np.repeat` are used when `interleave=True`.

The repeated batch is then padded to the number of async agent workers and chunked:

`src/rl/verl/experimental/agent_loop/agent_loop.py`

```python
chunkes = prompts.chunk(len(self.agent_loop_workers))
outputs = ray.get([
    worker.generate_sequences.remote(chunk)
    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
])
```

With 8 agent workers:

- `G=4`: 32 rows -> 8 chunks -> 4 rows per worker.
- `G=8`: 64 rows -> 8 chunks -> 8 rows per worker.

This means `G` changes not just the sample count, but also the per-worker workload, request arrival order, concurrency shape, and vLLM scheduler interaction.

## What Has Been Ruled Out

### Temperature and top-p mismatch

For the comparable runs, the explicit sampling settings were the same:

- Temperature: 0.2.
- Top-p: 0.9.
- Max response length: 10240.
- Max turns: 15.
- `max_num_seqs`: 16.
- Agent workers: 8.
- `max_parallel_calls`: 4.

The single `G=8` and the `G=4x2` control differ primarily in `ROLLOUT_N`.

### Internal vLLM `n=8` candidate generation

At first glance, `actor_rollout_ref.rollout.n` looks dangerous because `vllm_async_server.py` builds an `override_generation_config` from rollout config keys, and this can include `n`.

However, the actual internal generation path does this per request:

`src/rl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

```python
sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
generator = self.engine.generate(
    prompt=prompt,
    sampling_params=sampling_params,
    request_id=request_id,
)
return final_res.outputs[0].token_ids
```

The `sampling_params` passed from the agent loop only includes:

- `temperature`
- `top_p`
- `repetition_penalty`
- `max_tokens` / remaining budget

It does not include `n`.

In the active environment:

- vLLM version: `0.8.5.post1`.
- `SamplingParams()` default `n` is `1`.

The vLLM V1 internal processor clones the request-level `SamplingParams` and only updates stop/eos-related fields from generation config. It does not appear to overwrite request-level `n` from `override_generation_config` in this internal `engine.generate` path.

So the explanation "single `G=8` makes each request generate 8 candidates internally and then we only take output 0" is probably false for this code path.

## Current Best Explanation

`G` is not behaviorally neutral in the current async rollout implementation.

The likely mechanism is:

1. `G` changes the validation batch size and repeated row ordering.
2. The repeated rows are chunked across async agent workers.
3. Each trajectory creates a random UUID request id.
4. The request id is mapped to a vLLM server through a least-request sticky router.
5. vLLM is initialized with a global engine seed, but no per-request seed is passed.
6. With async scheduling, more simultaneous requests with similar or identical prefixes can change scheduler order, prefix-cache behavior, and random-number consumption.
7. Therefore a single `G=8` run is not equivalent to two independent `G=4` runs.

This is especially plausible because the `G=8` run did not merely show lower task success. It showed a large drop in protocol validity and format validity, plus runaway malformed generations. That points to a sampling/distribution shift during generation, not just normal evaluation variance.

## Earlier Deterministic-Sampling Hypothesis (Useful But Not Sufficient)

The async rollout path had three G-sensitive execution details:

1. `AgentLoopManager.generate_sequences()` repeats validation/train prompts and then chunks the whole repeated batch by `agent.num_workers`. Changing `G` changes each worker's chunk size and the request arrival pattern.
2. Each `AgentLoopWorker` created its own `AsyncLLMServerManager`, shuffled the vLLM server handles independently, and then used a per-worker least-request heap. That means the same logical rollout can route to a different vLLM server when `G` changes.
3. The agent loops used random UUID request ids and did not pass per-request `seed` into vLLM `SamplingParams`. With async scheduling, vLLM sampling RNG consumption was therefore tied to runtime scheduling rather than to the logical `(prompt, rollout_index)` sample.

There was also a trace-only bug: `get_trajectory_info()` recomputed `rollout_n` inside each worker chunk. Chunk boundaries change with `G`, so trace rollout ids were not globally stable. The fix avoids using that chunk-local count for stable sampling.

## Implemented Fallback-Safe Fix

The fix is opt-in and fully fallback-able. By default the legacy behavior remains unchanged.

New config:

`actor_rollout_ref.rollout.agent.stable_sampling`

Fields:

- `enabled`: master switch, default `False`.
- `seed_requests`: pass deterministic per-trajectory seed to vLLM, default `True`.
- `deterministic_request_id`: derive request id from step, prompt fingerprint, and rollout offset, default `True`.
- `stable_server_order`: stop per-worker random server shuffling, default `True` when enabled.
- `stable_server_routing`: route request ids by stable hash instead of per-worker least-request state, default `True` when enabled.
- `seed_base`: base seed for independent evaluation sweeps, default `0`.
- `rollout_offset_base`: offset added to rollout ids, default `0`.

Launch-script env vars:

- `STABLE_ROLLOUT_SAMPLING=True`
- `STABLE_ROLLOUT_SEED_BASE=0`
- `STABLE_ROLLOUT_SEED_REQUESTS=True`
- `STABLE_ROLLOUT_DETERMINISTIC_REQUEST_ID=True`
- `STABLE_ROLLOUT_STABLE_SERVER_ORDER=True`
- `STABLE_ROLLOUT_STABLE_SERVER_ROUTING=True`
- `STABLE_ROLLOUT_OFFSET_BASE=0`

Fallback:

- Set `STABLE_ROLLOUT_SAMPLING=False`, or omit it, to return to the old UUID + shuffled least-request route + no per-request seed behavior.

Recommended comparison:

- Run `G=4` with `STABLE_ROLLOUT_SEED_BASE=0`.
- Run `G=8` with `STABLE_ROLLOUT_SEED_BASE=0`.
- The first four logical rollouts per prompt now receive the same request ids and seeds in both runs; `G=8` only adds rollout ids 4-7.
- To mimic two `G=4` runs that together match one `G=8` seed set exactly, run the first `G=4` with `STABLE_ROLLOUT_OFFSET_BASE=0` and the second `G=4` with `STABLE_ROLLOUT_OFFSET_BASE=4`.
- Change `STABLE_ROLLOUT_SEED_BASE` only when you intentionally want a new independent deterministic sample set.

## Why This Matters

The evaluation result is currently not trustworthy as a clean estimate of checkpoint quality if we compare runs with different `G`.

Right now, `G` changes both:

- The statistical estimator: number of samples per query.
- The execution topology: batch size, chunk size, worker load, async request ordering, vLLM scheduling, and probably RNG consumption.

Because of that, a worse result at `G=8` cannot be interpreted as "the model fails when sampled more." It may instead mean "the rollout system has a G-sensitive sampling/runtime artifact."

## Immediate Risks

1. Checkpoint comparisons can be misleading if different checkpoint sweeps use different `G`.
2. Temperature/top-p sweeps can be confounded if the total query count or `G` changes at the same time.
3. GRPO/RL training may see a different behavior distribution from SFT evaluation if the training `rollout.n` differs from validation `val_kwargs.n`.
4. Teacher/student trajectory conclusions may be biased by the rollout engine rather than the model.

## Historical Verification Plan (Completed Or Superseded)

The next controlled experiments should isolate one variable at a time.

### Experiment A: decouple engine `rollout.n` from validation repeat

Run the same original 8 queries with:

- `actor_rollout_ref.rollout.n=1`
- `actor_rollout_ref.rollout.val_kwargs.n=8`

If this behaves like the bad single `G=8`, then the issue is outer repeat/concurrency/chunking.

If this behaves like `G=4`, then `rollout.n` entering rollout server config still matters somewhere despite the current static reading.

### Experiment B: single `G=8` but reduce async shape

Run:

- `G=8`
- 8 queries
- agent workers reduced from 8 to 4 or 1

If quality recovers, the issue is strongly tied to async worker scheduling or vLLM request interleaving.

### Experiment C: keep `G=8`, but execute as two sequential `G=4` shards automatically

This is already known to recover normal behavior in the manual control. If it remains stable, it is the safest practical evaluation workaround.

### Experiment D: add request-level deterministic seeds

For validation, derive a seed from:

- query index
- sample index within group
- assistant turn index
- global step

Pass that seed into request-level `SamplingParams` if vLLM accepts it cleanly.

This should make sample identity less dependent on async request order.

### Experiment E: shuffle repeated validation rows before chunking

Right now `repeat_interleave` creates contiguous identical-query groups. For `G=8`, each worker receives larger blocks of repeated query replicas. Shuffling after repeat, while preserving query/sample metadata for grouping, may reduce prefix-cache and worker-local correlation.

This is a diagnostic experiment, not necessarily the final training setup.

## Historical Interim Recommendation

Until the mechanism is fixed or proven harmless:

1. Do not compare checkpoint results across different `G` values.
2. For evaluation, prefer multiple independent smaller-`G` runs over one large-`G` run.
3. Treat the current single `G=8` failure as a rollout-system anomaly, not as a final judgment on `global_step_108`.
4. For any reported test metric, record:
   - `G`
   - query count
   - agent worker count
   - `max_num_seqs`
   - `max_num_batched_tokens`
   - whether the run was single large-`G` or sharded smaller-`G`
5. Before starting RL, decide whether `rollout.n` should remain tied to `val_kwargs.n`. For validation-only runs, these probably should be decoupled so we can control sampling count separately from rollout server/trainer internals.

## Status Before The 2026-07-10 Causal Experiments

The strongest evidence so far is:

- Single `G=8`: 0 / 64 success, protocol and format collapse.
- Two independent `G=4` runs: 8 / 64 success, protocol and format remain close to the original `G=4` run.

The most likely root cause is G-sensitive async rollout execution, not the harness reward, not the prompt, and not an explicit temperature/top-p mismatch.

The next best engineering step is to run Experiment A, because it cleanly tests whether `actor_rollout_ref.rollout.n` has hidden side effects beyond validation repeat.
