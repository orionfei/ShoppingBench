# ShoppingBench verl Hyperparameters

This note records the defaults used by the local verl entry scripts and why they differ from upstream examples.

## SFT

`src/rl/run_sft_qwen3_1_7b_verl.sh` is set for full-parameter SFT of `model/Qwen3-1.7B` on one 32 GB RTX 4080-class GPU:

- `max_length=8704`, matching the local SFT token stats where all rows fit under this cap.
- `train_batch_size=4` and `micro_batch_size_per_gpu=1`, giving gradient accumulation of 4 on one GPU.
- `lr=1e-5`, `weight_decay=0.01`, `warmup_steps_ratio=0.05`, `total_epochs=3`.
- `bf16`, gradient checkpointing, FSDP2, and `attn_implementation=sdpa` by default.

The Qwen MS-SWIFT Qwen3 full-tuning example uses the same broad regime for full SFT: bf16, per-device batch 1, gradient accumulation 4, `learning_rate=1e-5`, `max_length=8192`, `warmup_ratio=0.05`, and FlashAttention when packing is enabled:
https://qwen.readthedocs.io/en/latest/training/ms_swift.html

Our SFT loss mask is role-based: only assistant messages are supervised. In the generated ShoppingBench SFT parquet, assistant content is entirely inside `<think>...</think>` and `<tool_call>...</tool_call>`, so user prompts and tool responses condition the model but do not contribute loss.

## GRPO / RL

`src/rl/run_grpo_qwen3_1_7b_query_verl.sh` follows Qwen's official Qwen3-1.7B verl GRPO example, then scales down for a single 32 GB GPU and longer ShoppingBench tool rollouts:

- `adv_estimator=grpo`.
- `lr=1e-6`, `use_kl_loss=False` by default, with `kl_loss_coef=0.001` and `kl_loss_type=low_var_kl` kept for opt-in KL-loss experiments.
- `ref.fsdp_config.param_offload=True` to reduce GPU pressure from the reference policy.
- `rollout.name=vllm`, `rollout.mode=async`, `rollout.gpu_memory_utilization=0.6`.
- `rollout.n=3` by default for group-relative rewards; use higher values when more GPUs are available.
- `train_batch_size=8`, `ppo_mini_batch_size=8`, `ppo_micro_batch_size_per_gpu=1` for the initial single-card run.
- `max_prompt_length=4096`, `max_response_length=4096`, `max_model_len=8192`.
- `use_remove_padding=False` by default because the current Python 3.12 + torch 2.6.0 environment has no compatible ready flash-attn wheel on the configured mirror. Enable it after flash-attn is installed.
- The single-card smoke path uses LoRA (`src/rl/run_grpo_qwen3_1_7b_query_verl_lora_smoke.sh`) to verify rollout, rewards, backward, and optimizer stepping on a 32 GB 4080. A full-parameter 1-step GRPO update reached the AdamW optimizer step but OOMed on this GPU, so full-parameter GRPO should be treated as a multi-GPU or larger-memory run.

Qwen's official verl example for Qwen3-1.7B uses FSDP + vLLM GRPO with `lr=1e-6`, `use_remove_padding=True`, `ppo_mini_batch_size=80`, `ppo_micro_batch_size_per_gpu=20`, `use_kl_loss=True`, `kl_loss_coef=0.001`, `rollout.gpu_memory_utilization=0.6`, `rollout.n=3`, and `ref.fsdp_config.param_offload=True` on a single 80 GB GPU:
https://qwen.readthedocs.io/en/latest/training/verl.html

verl's performance guide recommends tuning vLLM throughput with `gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens`, commonly using `gpu_memory_utilization` around 0.5-0.7 to avoid OOM when trainer states share the GPU:
https://verl.readthedocs.io/en/latest/perf/perf_tuning.html

verl's GRPO documentation describes the intended algorithmic behavior: sample multiple responses per prompt, score each response, normalize rewards relative to the group, and update the policy without a critic:
https://verl.readthedocs.io/en/latest/algo/grpo.html
