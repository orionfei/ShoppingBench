# SFT and Eval Rollout Parameters

This file records the actual parameters used in the state-folded SFT training
and the later SFT checkpoint eval rollout/probe flow.

## 1. SFT Training

### Run Identity

| Item | Value |
|---|---|
| Log file | `logs/sft_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453.log` |
| Project | `shoppingbench-sft` |
| Experiment | `qwen3-1.7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32` |
| Output checkpoint dir | `checkpoints/sft/qwen3_1_7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453` |
| Base model | `model/Qwen3-1.7B` |
| SFT data | `dataset/shoppingbench_sft_state_folded/train.parquet` |
| SFT validation data | `dataset/shoppingbench_sft_state_folded/test.parquet` |

### Hardware and Parallelism

| Parameter | Value |
|---|---|
| Nodes | `1` |
| GPUs per node | `2` |
| FSDP strategy | `fsdp2` |
| Data parallel size | `2` |
| Sequence parallel size | `1` |
| Remove padding | `False` |
| Attention implementation | `sdpa` |
| Model dtype | `bf16` |
| Gradient checkpointing | `True` |

### Batch and Sequence Parameters

| Parameter | Value |
|---|---|
| Command-level effective global train batch size | `16` |
| Logged per-DP `data.train_batch_size` after normalization | `8` |
| Micro batch size per GPU | `1` |
| Max sequence length | `8704` |
| Prompt key | `question` |
| Response key | `answer` |
| Multiturn enabled | `True` |
| Messages key | `messages` |
| Tools key | `tools` |
| Truncation | `right` |
| Data loader workers | `2` |
| Pin memory | `True` |

Note: the log prints `Normalize batch size by dp 2`, so the visible
`data.train_batch_size=8` is the per-DP normalized value. The effective global
batch size for the run is `16`.

### Optimizer and Schedule

| Parameter | Value |
|---|---|
| Learning rate | `1e-5` |
| Betas | `[0.9, 0.95]` |
| Weight decay | `0.01` |
| Warmup ratio | `0.05` |
| LR scheduler | `cosine` |
| Clip grad | `1.0` |

### Training Duration and Checkpoints

| Parameter | Value |
|---|---|
| Total epochs | `2` |
| Steps per epoch | `128` |
| Total training steps | `256` |
| Save frequency | `32` steps |
| Validation/test frequency | `32` steps |
| Save final checkpoint | `True` |
| Seed | `1` |
| Logger | `console` |

Saved checkpoints:

```text
global_step_32
global_step_64
global_step_96
global_step_128
global_step_160
global_step_192
global_step_224
global_step_256
```

SFT loss curve:

```text
plots/sft_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453_loss_curve.png
```

## 2. Aligned Probe Data

The eval rollout used a newly aligned state-folded probe parquet, built from the
same fixed 8 RL train queries used earlier, but with the current SFT/RL-aligned
tool schema in the system prompt.

| Item | Value |
|---|---|
| 8-query probe parquet | `dataset/probe/sft_probe_query_8_statefolded_20260620.parquet` |
| 8-query probe jsonl | `dataset/probe/sft_probe_query_8_statefolded_20260620.jsonl` |
| 1-query smoke parquet | `dataset/probe/sft_probe_query_1_statefolded_20260620.parquet` |
| 1-query smoke jsonl | `dataset/probe/sft_probe_query_1_statefolded_20260620.jsonl` |
| Source train parquet | `dataset/shoppingbench_query/train.parquet` |
| Fixed source row indices | `[4, 107, 218, 221, 279, 510, 521, 573]` |
| Data source | `shoppingbench_query` |
| Agent name | `tool_agent` |

The query-level reward product cache used by rollout/reward:

```text
dataset/shoppingbench_query/product_cache.json
```

## 3. Eval Rollout Flow Validation Smoke

Before running all checkpoints, a real 4-GPU smoke rollout was run on
`global_step_128` to verify that RL rollout matches the SFT state-folded view.

### Final Successful Smoke Run

| Parameter | Value |
|---|---|
| Checkpoint | `global_step_128` |
| Model path | `checkpoints/sft/qwen3_1_7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453/global_step_128` |
| Train files | `dataset/probe/sft_probe_query_1_statefolded_20260620.parquet` |
| Val files | `dataset/probe/sft_probe_query_1_statefolded_20260620.parquet` |
| Validation output dir | `rollouts/statefolded_smoke_4gpu_tp4_20260620/global_step_128_g4_fixed` |
| Log file | `logs/statefolded_smoke_global_step_128_4gpu_tp4_g4_fixed_20260620.log` |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| CUDA device order | `PCI_BUS_ID` |
| GPUs per node | `4` |
| Tensor parallel size | `4` |
| Rollout group size `n` | `4` |
| Train batch size | `1` |
| Val batch size | `1` |
| Agent workers | `8` |
| GPU memory utilization | `0.85` |
| Max num seqs | `16` |
| Max batched tokens | `16384` |
| Max model length | `8192` |
| Max prompt length | `4096` |
| Max response length | `4096` |
| Total epochs | `1` |
| Total training steps | `1` |
| Save frequency | `-1` |
| Test frequency | `1` |
| Val only | `True` |
| Validate before train | `True` |
| Logger | `console` |

The earlier attempted smoke with `ROLLOUT_N=2` failed configuration validation:

```text
real_train_batch_size (2) must be divisible by minimal possible batch size (4)
```

So all real 4-GPU eval rollout runs use `ROLLOUT_N=4`.

### Smoke Verification Results

The successful smoke produced 4 rows and verified:

| Check | Result |
|---|---|
| Rows | `4` |
| Tool valid mean | `1.0` |
| Protocol mean | `0.9166666667` |
| Decoded output contains `<state>` | `True` for all rows |
| Decoded output contains raw `<obs>` | `False` for all rows |
| Prompt contains Qwen `<tools>` template | `False` for all rows |
| Structured message counts present | `message_count`, `state_user_message_count`, `user_obs_message_count`, `event_count`, `observed_event_count` |

## 4. Full Eval Rollout: SFT Checkpoint Probe

### Run Identity

| Item | Value |
|---|---|
| Run id | `sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029` |
| Log dir | `logs/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029` |
| Rollout dir | `rollouts/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029` |
| Plot/metric prefix | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_*` |
| Checkpoint root | `checkpoints/sft/qwen3_1_7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453` |

### Shared Rollout Parameters

| Parameter | Value |
|---|---|
| Script | `src/rl/run_grpo_qwen3_1_7b_query_verl.sh` |
| CUDA devices | `0,1,2,3` |
| CUDA device order | `PCI_BUS_ID` |
| `PYTHONPATH` | `/root/autodl-tmp/ShoppingBench/src/rl` |
| `OMP_NUM_THREADS` | `8` |
| `OMP_NUM_THREADS_OVERRIDE` | `8` |
| `RAY_DEDUP_LOGS` | `1` |
| Nodes | `1` |
| GPUs per node | `4` |
| Train files | `dataset/probe/sft_probe_query_8_statefolded_20260620.parquet` |
| Val files | `dataset/probe/sft_probe_query_8_statefolded_20260620.parquet` |
| Train batch size | `8` |
| Val batch size | `8` |
| Rollout group size `n` | `4` |
| Total eval samples | `8 queries * 4 = 32` per checkpoint |
| Rollout mode | `async` |
| Rollout backend | `vllm` |
| Tensor parallel size | `4` |
| Rollout agent workers | `32` |
| Rollout GPU memory utilization | `0.88` |
| Rollout max num seqs | `64` |
| Rollout max num batched tokens | `32768` |
| Rollout max model len | `8192` |
| Max prompt length | `4096` |
| Max response length | `4096` |
| Temperature | `1.0` |
| Top-p | `0.9` |
| Do sample | `True` |
| Dtype | `bfloat16` |
| `VLLM_USE_V1` | `1` |
| Attention implementation | `sdpa` |
| Use remove padding | `False` |
| LoRA rank | `0` |
| LoRA alpha | `16` |
| Target modules | `all-linear` |
| Total epochs | `1` |
| Total training steps | `1` |
| Save frequency | `-1` |
| Test frequency | `1` |
| Val only | `True` |
| Validate before train | `True` |
| Logger | `console` |

### Multi-turn Tool Rollout Parameters

| Parameter | Value |
|---|---|
| Multi-turn enabled | `True` |
| Multi-turn format | `shoppingbench_xml` |
| Tool config | `config/rl/shoppingbench_tools.yaml` |
| Max assistant turns | `6` |
| Max user turns | `6` |
| Max parallel calls | default `1` |
| Max tool response length | `256` |
| Tool response truncate side | `middle` |
| Tokenization sanity check mode | `ignore_strippable` |
| State max candidates per search | default `10` |

Enabled tools:

```text
find_product
view_product_information
recommend_product
terminate
python_execute
```

Search server used by `find_product` and `view_product_information`:

| Parameter | Value |
|---|---|
| Host | `127.0.0.1` |
| Port | `5631` |
| Base URL | `http://127.0.0.1:5631` |
| Command | `HOST=127.0.0.1 PORT=5631 OMP_NUM_THREADS=8 python3 src/search_engine/server.py` |

### Reward and Metrics

| Parameter | Value |
|---|---|
| Data source | `shoppingbench_query` |
| Reward function | `src/rl/verl/utils/reward_score/shoppingbench_query.py` |
| Reward manager | `naive` |
| Protocol weight start | default `0.2` |
| Protocol anneal steps | derived from total training steps unless explicitly set |
| Step penalty | default `0.02` |
| Product cache | `dataset/shoppingbench_query/product_cache.json` |

Core checkpoint selection metrics:

```text
protocol_mean
protocol_group_var_mean
task_group_var_mean
```

Selection rule used in this probe:

```text
protocol_mean >= 0.90
protocol_group_var_mean <= 0.05
choose the highest task_group_var_mean
```

## 5. Checkpoints Evaluated

Each checkpoint was evaluated with the same 8 fixed queries and `G=4`.

```text
global_step_32
global_step_64
global_step_96
global_step_128
global_step_160
global_step_192
global_step_224
global_step_256
```

Each checkpoint produced:

```text
rollouts/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029/global_step_<step>/0.jsonl
```

## 6. Eval Rollout Outputs

| Artifact | Path |
|---|---|
| Summary CSV | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_summary.csv` |
| Summary JSON | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_summary.json` |
| Per-query CSV | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_per_query.csv` |
| Debug validation JSON | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_debug_validation.json` |
| Three-metric plot | `plots/sft_ckpt_probe_statefolded_4gpu_tp4_g4_20260620_0029_three_metrics.png` |

## 7. Eval Summary

| Checkpoint | Rows | Queries | Complete Groups | Protocol Mean | Protocol Group Var Mean | Task Mean | Task Group Var Mean | Format Mean | Tool Valid Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `global_step_32` | 32 | 8 | 8 | 0.757812 | 0.053874 | 0.216250 | 0.168433 | 0.703125 | 0.812500 |
| `global_step_64` | 32 | 8 | 8 | 0.958333 | 0.004015 | 0.076771 | 0.038543 | 0.916667 | 1.000000 |
| `global_step_96` | 32 | 8 | 8 | 0.924479 | 0.008735 | 0.070521 | 0.046531 | 0.848958 | 1.000000 |
| `global_step_128` | 32 | 8 | 8 | 0.953125 | 0.005425 | 0.159427 | 0.069987 | 0.906250 | 1.000000 |
| `global_step_160` | 32 | 8 | 8 | 0.945312 | 0.008626 | 0.036771 | 0.003633 | 0.916667 | 0.973958 |
| `global_step_192` | 32 | 8 | 8 | 0.927083 | 0.007704 | 0.111354 | 0.043100 | 0.869792 | 0.984375 |
| `global_step_224` | 32 | 8 | 8 | 0.940104 | 0.005914 | 0.038542 | 0.001352 | 0.890625 | 0.989583 |
| `global_step_256` | 32 | 8 | 8 | 0.947917 | 0.006619 | 0.175833 | 0.099348 | 0.911458 | 0.984375 |

Selected checkpoint by the rule above:

```text
global_step_256
```

## 8. Flow Consistency Checks

The final eval rollout verified these invariants:

| Check | Result |
|---|---|
| Every checkpoint has 32 rollout rows | yes |
| Every checkpoint has 8 complete query groups | yes |
| Decoded input has no Qwen `<tools>` template | yes |
| Decoded model context has no raw `<obs>` tokens | yes |
| State-folded context is present after tool turns | yes |
| Reward receives structured message/event fields | yes |
| Tool observations are available for valid executable tool calls | yes |

Some checkpoints have `event_count > observed_event_count` for a few rows. This
means the model emitted malformed, incomplete, or otherwise non-executable tool
calls, and the reward correctly penalized those cases through protocol/tool
validity. It is not evidence that the pipeline dropped observations.
