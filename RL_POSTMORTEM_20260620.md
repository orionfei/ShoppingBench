# RL Postmortem - 2026-06-20

## Stopped run

- Run id: `grpo_gs224_taskreward_4gpu_bs4_n4_resp2048_vllm038_entropychunk_20260620_100222`
- Init checkpoint: `checkpoints/sft/qwen3_1_7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453/global_step_224`
- Stopped at: train step 19
- Saved checkpoint: none from this run, because `save_freq=20` and the run was stopped before step 20 completed.
- Validation available: step 10 full validation at `rollouts/grpo_gs224_taskreward_4gpu_bs4_n4_resp2048_vllm038_entropychunk_20260620_100222/validation/10.jsonl`

## Main pitfalls

1. Search server must be persistent before RL starts.
   - A previous rollout attempt failed with connection refused because the search server was not alive.
   - The stable service command used later was `setsid python -u src/search_engine/server.py`.

2. The first 4GPU RL configuration was too memory-heavy.
   - `train_batch_size=8`, `max_response=4096`, and high vLLM memory caused OOM during actor log-prob/entropy computation.
   - The OOM happened in the non-remove-padding entropy path.

3. Entropy computation needed chunking in the non-remove-padding path.
   - Patched `src/rl/verl/workers/actor/dp_actor.py`.
   - New default in the RL script: `actor_rollout_ref.actor.entropy_from_logits_with_chunking=True`.

4. `max_response=1024` is not enough for current RL rollouts.
   - RL train initial prompts: max 992 tokens.
   - RL test initial prompts: max 961 tokens.
   - SFT final assistant responses: max 431 tokens.
   - Current RL generated trajectories with `max_response=2048`: train mean about 1632 tokens, validation mean about 1641 tokens, and more than 95% of validation responses exceeded 1024 tokens.
   - Therefore `max_prompt_length=1024` is fine, but `max_response_length` should stay at 2048 for now.

5. Full validation is expensive.
   - Step 10 validation used 75 prompts with `n=4`, i.e. 300 trajectories.
   - It took about 28 minutes inside the step 10 wall time.
   - For short diagnostics, validate less frequently and save more frequently.

6. Protocol metrics are no longer the main bottleneck.
   - Step 10 validation protocol mean was about 0.932.
   - Format and tool-validity are mostly learned from SFT.
   - The real bottleneck is task completion: recommendation, budget, and terminate correctness.

7. The current reward did not show convincing task improvement in the first 19 steps.
   - Train hard task counts remained zero for `final_success`, `exact`, `success`, `outcome`, and `budget`.
   - Step 10 validation also had all hard completion metrics at zero.
   - Weak signals such as `search_gold_recall` and `recommend_gold_overlap` existed, but did not clearly trend upward.

## Key observed metrics

Step 10 full validation:

- `score/task mean`: 0.014958
- `progress mean`: 0.053358
- `search_gold_recall mean`: 0.339444
- `recommend_gold_overlap mean`: 0.035833
- `recommend_gold_f1 mean`: 0.023039
- `budget mean`: 0.0
- `final_success/exact/success/outcome`: all 0.0
- `protocol mean`: 0.932139

Train step 19:

- `score/task mean`: 0.036170
- `progress mean`: 0.076170
- `search_gold_recall mean`: 0.442708
- `recommend_gold_overlap mean`: 0.083333
- `recommend_gold_f1 mean`: 0.054754
- Hard completion counts: all 0 out of 16.

## Updated default RL parameters

Updated in `src/rl/run_grpo_qwen3_1_7b_query_verl.sh`.

- `TRAIN_BATCH_SIZE=4`
- `VAL_BATCH_SIZE=4`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=2048`
- `PPO_MINI_BATCH_SIZE=4`
- `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
- `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1`
- `PPO_MAX_TOKEN_LEN_PER_GPU=10240`
- `ROLLOUT_N=4`
- `ROLLOUT_TEMPERATURE=0.8`
- `ROLLOUT_TOP_P=0.9`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.38`
- `ROLLOUT_MAX_MODEL_LEN=6144`
- `ROLLOUT_MAX_NUM_BATCHED_TOKENS=6144`
- `ROLLOUT_MAX_NUM_SEQS=8`
- `ROLLOUT_AGENT_NUM_WORKERS=8`
- `MAX_ASSISTANT_TURNS=4`
- `MAX_USER_TURNS=4`
- `MAX_TOOL_RESPONSE_LENGTH=256`
- `ACTOR_OPTIMIZER_OFFLOAD=True`
- `ACTOR_OPTIMIZER_FOREACH=false`
- `ENTROPY_FROM_LOGITS_WITH_CHUNKING=True`
- `ENTROPY_COEFF=0`
- `LEARNING_RATE=1e-6`
- `NGPUS_PER_NODE=4`
- `TOTAL_EPOCHS=1`
- `TOTAL_TRAINING_STEPS=20`
- `SAVE_FREQ=20`
- `TEST_FREQ=20`
- `VAL_BEFORE_TRAIN=False`
- `ACTOR_CHECKPOINT_SAVE_CONTENTS=['model','extra']`
- `ACTOR_CHECKPOINT_LOAD_CONTENTS=['model','extra']`

Reward env defaults for next run:

- `SHOPPINGBENCH_PROTOCOL_WEIGHT_START=0.0`
- `SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS=0`
- `SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION=0.0`
- `SHOPPINGBENCH_STEP_PENALTY=0.005`

The learning rate now follows the Qwen documentation's veRL Qwen3-1.7B GRPO
example (`https://qwen.readthedocs.io/en/latest/training/verl.html`), which uses
`actor_rollout_ref.actor.optim.lr=1e-6`,
`kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`, and `entropy_coeff=0`.
Memory-related parameters remain adapted to this 4x32GB machine rather than the
single 80GB example.

## Next recommendation

Run a 20-step diagnostic from `global_step_224` or from a saved RL checkpoint only after checking step 20 validation. If hard task metrics are still zero and weak signals do not improve, modify reward before running longer. The likely reward direction is to make correct search, valid candidate selection, budget verification, correct recommendation count, and terminate-after-recommend each provide clearer intermediate preference.

## Reward v2 follow-up

Reward v2 was implemented after this stopped run. It increases within-query
variance for GRPO by separately rewarding search recall, selection F1, viewed
gold evidence, budget attempt quality, budget numeric alignment, recommendation
F1, exact product-set match, and terminate quality. It also penalizes wrong
recommendations, count mismatch, premature success termination, invalid tool
calls, and long multi-turn loops.

The recomputed 8-checkpoint SFT probe outputs are:

- `plots/task_reward_v2_8ckpts_20260620_raw.csv`
- `plots/task_reward_v2_8ckpts_20260620_raw.json`
- `plots/task_reward_v2_8ckpts_20260620_summary.csv`
- `plots/task_reward_v2_8ckpts_20260620_summary.json`
- `plots/task_reward_v2_8ckpts_20260620_per_query_variance.csv`
- `plots/task_reward_v2_8ckpts_20260620_task_mean_var.png`
- `plots/task_reward_v2_8ckpts_20260620_components.png`

## Reward v2 20-step diagnostic

Successful run:

- Run id: `grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144`
- Init checkpoint: `checkpoints/sft/qwen3_1_7b_state_folded_2gpu_bs16_micro1_lr1e-5_ep2_save32_20260619_1453/global_step_224`
- Train file: `dataset/shoppingbench_query/train_sft_overlap_300.parquet`
- Validation file: `dataset/shoppingbench_query/test.parquet`
- Actual SFT-overlap train rows: 275. The 25 copied rows that belong to test
  were not used for RL train.
- GPUs: 4
- Learning rate: `1e-6`
- Rollout: `n=4`, `temperature=0.8`, `top_p=0.9`
- Max lengths: `max_prompt_length=1024`, `max_response_length=2048`
- Steps: `total_training_steps=20`
- Save/test frequency: `save_freq=20`, `test_freq=20`
- Checkpoint save contents: `['model','extra']`
- Saved checkpoint:
  `checkpoints/shoppingbench-rl/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144/global_step_20`
- Checkpoint size: about `7.6G`
- Validation rollout:
  `rollouts/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144/validation/20.jsonl`

Disk notes:

- Two failed intermediate RL checkpoint directories were removed because they
  were incomplete and consumed about 34G.
- `checkpoints/smoke` was removed because it was only a smoke-test artifact and
  consumed about 7.6G.
- Final available disk after the successful run was about 34G.

Train reward result:

- First 5 train steps average score: `0.01277`
- Last 5 train steps average score: `0.09154`
- Delta: `+0.07877`
- Step 20 train score mean: `0.11000`
- Step 20 train search recall mean: `0.66667`

Validation result after step 20:

- `score mean@4`: `0.07552`
- `progress mean@4`: `0.08879`
- `penalties mean@4`: `0.01327`
- `search_gold_recall mean@4`: `0.48528`
- `recommend_gold_f1 mean@4`: `0.00067`
- `select_gold_f1 mean@4`: `0.00067`
- `verify_gold_f1 mean@4`: `0.00859`
- `final_success mean@4`: `0.0`
- `set_exact mean@4`: `0.0`
- `budget mean@4`: `0.0`
- `tool_valid mean@4`: `0.98167`
- `format mean@4`: `0.96611`

Summary:

Reward v2 successfully creates a positive train signal and improves the search
recall/progress part of the behavior over 20 steps. It still does not push the
model into correct final recommendation, budget recomputation, or termination.
For the next reward iteration, the main issue is not protocol validity. The
reward needs stronger pressure for selecting a compact candidate set, computing
budget, recommending the correct count, and terminating after a valid
recommendation.

Generated plots and summaries:

- `plots/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144_summary_train_curve.png`
- `plots/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144_summary_val_metrics.png`
- `plots/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144_summary_train_summary.csv`
- `plots/grpo_gs224_rewardv2_sftwarm275_4gpu_lr1e-6_save20_modelonly_20260620_1144_summary_val_overall.json`
