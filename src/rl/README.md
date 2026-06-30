# ShoppingBench RL

Recommended Qwen3-4B/A800 entry:

```bash
src/rl/run_grpo_qwen3_4b_state_folded_a800.sh
```

Default policy seed:

```text
checkpoints/shoppingbench-sft/qwen3-4b_state_folded_4xa800_full_sft_lr1e-5_micro2_20260628_1139/global_step_256
```

The wrapper keeps the state-folded history controls enabled for RL:

```text
STATE_MAX_CANDIDATES_PER_SEARCH=10
STATE_MAX_SEARCHES=12
STATE_MAX_BUDGET_CANDIDATES=120
STATE_MAX_VIEWED_PRODUCTS=40
STATE_NEVER_EXPAND=True
STATE_MIN_CHAR_SAVING=0.10
```

Check parameters without starting Ray/vLLM:

```bash
DRY_RUN=1 src/rl/run_grpo_qwen3_4b_state_folded_a800.sh
```

Reward variance / val-only entry for the selected SFT checkpoint:

```bash
src/rl/evaluate_sft_step256_grpo_reward_variance_a800.sh
```

LoRA smoke entry:

```bash
src/rl/run_grpo_qwen3_4b_state_folded_lora_smoke.sh
```

RL tool rollouts require the ShoppingBench search server at `http://127.0.0.1:5631/`.
Start it against the official full `indexes/` for real runs.
