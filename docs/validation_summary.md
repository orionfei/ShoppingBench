# Validation Summary

This records the current lightweight validation state for `model/Qwen3-1.7B` on the local 32 GB RTX 4080 SUPER.

## Environment

- Python: 3.12.3
- torch: 2.6.0+cu124
- vLLM: 0.8.5.post1
- Ray: 2.55.1
- xFormers: 0.0.29.post2
- FlashInfer: 0.2.2.post1+cu124torch2.6
- verl: local editable package
- `python -m pip check`: clean

`logs/vllm_flashinfer_022_smoke.log` confirms vLLM uses its own Flash Attention backend and FlashInfer sampling.

## SFT

The SFT loss mask is assistant-only. In the generated ShoppingBench SFT parquet, assistant messages are fully inside `<think>...</think>` and `<tool_call>...</tool_call>`, so user prompts and tool responses are context but not supervised targets.

GPU smoke overfit on one training row for three optimizer steps passed:

- step 1 train/loss: 2.3223843574523926
- step 2 train/loss: 1.1562037467956543
- step 3 train/loss: 0.630605161190033
- final val/loss: 0.5662730932235718

Log: `logs/sft_overfit_1x3.log`

## Search And RL Validation

The product corpus was decompressed and indexed under `indexes/`. The search service was tested with both `find_product` and `view_product_information`.

GRPO validation-only rollout passed after fixing validation metric aggregation for nullable reward extras:

- `val-core/shoppingbench_query/reward/mean@1=0.0`
- `val-aux/shoppingbench_query/expected_count/mean@1=4.0`
- `val-aux/num_turns/mean=2.0`

The base Qwen3 model did not produce useful ShoppingBench tool calls yet, so reward 0 is expected before SFT/RL adaptation.

Log: `logs/grpo_gpu_val_smoke_fixed_retry.log`

The XML tool parser and tool execution path were tested separately with a synthetic `<tool_call>` and returned live search results.

## Single-GPU GRPO Training Boundary

A full-parameter 1-step GRPO update on the 32 GB 4080 reached `update_actor`, then OOMed inside AdamW:

- attempted allocation: 1.16 GiB
- free GPU memory at failure: about 1.01 GiB
- failure point: `torch.optim.adamw._multi_tensor_adamw`

Log: `logs/grpo_gpu_train_1step_lowmem.log`

Conclusion: one 32 GB 4080 is enough for SFT smoke, vLLM rollout validation, reward validation, and parser/tool validation, but not a safe target for full-parameter GRPO optimizer updates with Qwen3-1.7B plus rollout engine sharing the device.

For single-card RL update smoke, use:

```bash
src/rl/run_grpo_qwen3_1_7b_query_verl_lora_smoke.sh
```

For multi-card or larger-memory full-parameter GRPO, use:

```bash
src/rl/run_grpo_qwen3_1_7b_query_verl.sh
```

Keep `LORA_RANK=0` for full-parameter training, and set `NGPUS_PER_NODE`, `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE`, batch sizes, and rollout lengths to match the available GPU budget.
