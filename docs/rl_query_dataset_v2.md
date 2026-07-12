# Step108 GRPO Query Dataset v2

Date: 2026-07-10

## Decision

The first outcome-only GRPO run reuses the existing 750 synthetic ShoppingBench queries.  Query text,
ground truth, voucher semantics, and binary reward are not regenerated.  The dataset is instead repartitioned
to prevent development leakage, repair final-test product overlap, and attach curriculum diagnostics.

The frozen layout is:

| Split | Queries | Role |
|---|---:|---|
| `train.parquet` | 643 | GRPO optimization |
| `validation.parquet` | 16 | fixed online learning-curve monitor |
| `calibration.parquet` | 16 | archived sampling-search panel; never train on it |
| `test.parquet` | 75 | untouched internal final evaluation |

All four splits form a disjoint cover of the original 750 queries.  The old
`dataset/shoppingbench_query/` directory is retained unchanged as the fallback source.

## Why repartitioning was necessary

The sampling calibration16 and validation16 panels were selected from the old 675-query training parquet.
Training on all 675 rows would therefore leak the online validation panel into optimization.  Both panels are
now permanently excluded from the 643-query training split.

The old test75 had no query or reward-tuple overlap with train, but two test queries contained four product IDs
that also occurred in development data.  Those two test rows were exchanged for two train rows with globally
unique products.  Each replacement exactly matches the removed row on:

- voucher type;
- discount type;
- number of products;
- operational budget-difficulty bucket; and
- source corpus.

The repaired final test has zero query overlap and zero product-ID overlap with train, validation, and
calibration combined.  Its aggregate voucher, discount, product-count, difficulty, complexity, and source
distributions are unchanged by the exact-stratum swaps.

## Training distribution

The 643-query train split contains 1,797 unique target products:

| Dimension | Counts |
|---|---|
| Voucher | platform 193, shop 450 |
| Discount | fixed 238, percentage 405 |
| Products/query | 1: 51, 2: 195, 3: 217, 4: 180 |
| Budget difficulty | easy 117, medium 226, hard 300 |
| Threshold difficulty | easy 166, medium 324, hard 153 |
| Constraint complexity | low 105, medium 237, high 301 |
| Historical source | clean300: 252, sample450: 391 |

The class imbalance is intentional for the first run: the data remains multi-product and tight-budget heavy,
matching the current task rather than being artificially balanced after observing Step108 results.  The first
epoch should use uniform query sampling.  Outcome-history prioritization, if enabled later, must retain a
uniform component and must not alter reward values.

## Added metadata

Every row preserves the six VERL columns and adds the following fields under `extra_info`:

```text
query_id
original_split / original_split_index
voucher_type / discount_type / product_count
reward_product_ids
constraint_count / constraint_complexity
budget / price_after_voucher / budget_slack / budget_slack_ratio / budget_difficulty
ground_truth_total_price / voucher_threshold / threshold_ratio / threshold_difficulty
```

These fields are diagnostics and future curriculum inputs only.  They never enter `terminal_asr` and do not
constitute reward shaping.

## Acceptance results

- emitted counts: 643 / 16 / 16 / 75;
- unique query union: 750, with pairwise overlap zero;
- final-test/development product-ID overlap: zero;
- product cache: 2,093 required IDs, zero missing;
- frozen calibration and validation query/prompt/ground-truth payloads: unchanged;
- exact annotated answers through the official-aligned batch scorer: 750/750 `paper_asr=1` and 750/750
  `terminal_asr=1` when followed by `terminate(success)`;
- maximum prompt length: 1532, below the 2048 launcher limit;
- deterministic rebuild: all four parquet SHA-256 values and the manifest SHA-256 are stable across reruns.

Machine-readable details, every split member, source hashes, swap identities, and output hashes are stored in
`dataset/shoppingbench_query_rl_v2/report.json` and `dataset/shoppingbench_query_rl_v2/manifest.json`.

## Runtime wiring and fallback

The Step108 4B launcher now defaults to:

```text
TRAIN_FILES=dataset/shoppingbench_query_rl_v2/train.parquet
VAL_FILES=dataset/shoppingbench_query_rl_v2/validation.parquet
```

The final-test runner defaults to `dataset/shoppingbench_query_rl_v2/test.parquet`.  All paths remain ordinary
environment-variable defaults, so reverting to the old 675/75 split requires no code rollback:

```bash
TRAIN_FILES=dataset/shoppingbench_query/train.parquet \
VAL_FILES=dataset/shoppingbench_query/test.parquet \
bash src/rl/run_grpo_qwen3_4b_state_folded_a800.sh
```

To reproduce the v2 data exactly:

```bash
python scripts/prepare_rl_query_dataset_v2.py --force
```

## First-run volume

With train batch 8 and G=8, one full pass is approximately 80 optimizer steps and 5,120–5,144 trajectories,
depending on remainder handling.  A 200-step ceiling corresponds to 12,800 trajectories and roughly 2.5
query passes.  The recommended operational policy is to evaluate the fixed validation16 during the first pass
and use early stopping; 200 is a ceiling, not a requirement to consume every step.

