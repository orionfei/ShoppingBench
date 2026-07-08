# Harness Optimization Log - 2026-07-08

This file is the tracked experiment log. Runtime copy: `logs/harness_optimization_20260708.md` (`logs/` is gitignored).

## Objective

Optimize the ShoppingBench state-local harness using real `gpt-5.5-medium` rollouts. Harness decisions must rely only on structured tool/state outputs, not parsing the user query or behavior recognition. Keep changes only when rollout evidence shows better success, shorter trajectories, or removal of wasted error turns without regression.

## Runtime

- Model: `gpt-5.5-medium`
- Remote API: user-provided OpenAI-compatible endpoint
- Required proxy bypass: `NO_PROXY=35.220.164.252,127.0.0.1,localhost`
- Search server: local `src/search_engine/server.py` on port `5631`
- Max steps: `15`
- Broad sample: `data/tmp/state_local_broad16_20260707.jsonl`

## Validated Baselines

- Initial broad16 reference:
  - Rollout: `data/tmp/state_local_think_gpt55_broad16_w1_s15_20260708_063531_rollout.jsonl`
  - Reward: `data/tmp/state_local_think_gpt55_broad16_w1_s15_20260708_reward_report.json`
  - Result: `8/16` success, average `10.0` steps.
- Current validated harness:
  - Rollout: `data/tmp/state_local_retry2final_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
  - Runner: `data/tmp/state_local_retry2final_remote_gpt55medium_broad16_w16_s15_20260708_report.json`
  - Stage reward: `data/tmp/state_local_retry2final_broad16_stage_reward_audit_20260708.json`
  - Result: `10/16` success, average `4.1875` steps, stage progress `0.7416`, `format=1.0`, `tool_valid=1.0`, `workflow_valid=1.0`, runner `error_counts={}`.

## Kept Changes

- Balanced candidate truncation:
  - Problem: first-N truncation hid candidates from later searches.
  - Change: internally tag source search and round-robin displayed candidates by source.
  - Evidence: targeted rows 4 and 6 succeeded `2/2`; broad16 improved from `8/16` to `10/16` with shorter average steps.
- Shop voucher structural issue detection:
  - Problem: cross-shop selections could pass toward recommendation under shop vouchers.
  - Change: detect multi-shop selections from structured selected products / `budget_check`, expose `selection_issues`, and force replacement search path.
  - Evidence: broad16 retained `10/16`; shop-voucher failures moved into same-shop retry flow.
- SELECT retry `previous_searches`:
  - Problem: SELECT retry with `find_product` allowed lacked search history and repeated searches.
  - Change: include `previous_searches` when SELECT allows `find_product`.
  - Evidence: repeated-search errors in failed8 dropped from 6 to 0; broad16 retained `10/16`.
- Recoverable format wrapper errors:
  - Problem: valid tool JSON could be discarded due minor outer wrapper mistakes.
  - Change: recover wrapper-level errors only when parsed tool calls and JSON schema are valid; implemented in runner and RL loop.
  - Evidence: broad16 stayed `10/16`, average steps improved `6.75 -> 6.625`, format `0.9833 -> 1.0`, runner errors `5 -> 0`.
- Retry-final DECISION gate:
  - Problem: several failures already had a structured verified selection with `budget_check.within_budget=true`, then spent many DECISION/SELECT cycles searching and re-verifying without success.
  - Change: when the state is DECISION, `budget_calculation.within_budget is true`, no shop-voucher cross-shop issue exists, and the trajectory has already made at least two structured DECISION `find_product` retry calls, restrict allowed tools to `recommend_product` and `terminate`.
  - Rationale: initial DECISION remains free to self-correct. The gate only shortens repeated retry loops after multiple replacement searches.
  - Evidence: broad16 with `max_steps=15` improved from `10/16`, average `6.625` steps, stage progress `0.7159` to `10/16`, average `4.1875` steps, stage progress `0.7416`, with `format=1.0`, `tool_valid=1.0`, `workflow_valid=1.0`, and runner errors `{}`.

## Reverted Experiments

- `same_shop_candidate_pool` overflow:
  - Hypothesis: expose omitted same-shop candidates for shop vouchers.
  - Evidence: row14 targeted still failed `0/1`, 15 steps, no terminate.
  - Decision: reverted.
- Page-2 search rule:
  - Hypothesis: tell model to try page 2 before page-1 paraphrases.
  - Evidence: rows 3/10/14 targeted `0/3`, all 15 steps, introduced `repeated_search_not_allowed:1`.
  - Decision: reverted.
- `candidate_sources`:
  - Hypothesis: group displayed candidates by originating `find_product` call.
  - Rollout: `data/tmp/state_local_sources_remote_gpt55medium_failed6_w6_s15_20260708_rollout.jsonl`
  - Reward: `data/tmp/state_local_sources_remote_gpt55medium_failed6_w6_s15_20260708_reward_report.json`
  - Result: `0/6` success, average `13.667` steps, format `0.9889`, errors `exactly_one_think_block_required:1`, `repeated_search_not_allowed:2`.
  - Baseline failed6 subset: `0/6`, average `11.667` steps, format `1.0`.
  - Decision: reverted.
- Candidate title 120 chars:
  - Hypothesis: 80-char candidate titles hide product/model evidence.
  - Targeted rollout: `data/tmp/state_local_title120_remote_gpt55medium_failed6_w6_s15_20260708_rollout.jsonl`
  - Targeted result: `0/6`, average `11.5` steps, better partial rule/sku but no success.
  - Broad rollout: `data/tmp/state_local_title120_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
  - Broad reward: `data/tmp/state_local_title120_remote_gpt55medium_broad16_w16_s15_20260708_reward_report.json`
  - Broad result: `9/16` success, average `7.3125` steps, format `0.9922`.
  - Baseline: `10/16`, average `6.625`, format `1.0`.
  - Decision: reverted.
- Reject repeated verified selections:
  - Hypothesis: after DECISION replacement searches, forbid exact duplicate `budget_check.product_ids` bundles already verified, reducing repeated SELECT loops without parsing the query.
  - Targeted rollout: `data/tmp/state_local_nodup_remote_gpt55medium_failed6_w6_s15_20260708_rollout.jsonl`
  - Runner report: `data/tmp/state_local_nodup_remote_gpt55medium_failed6_w6_s15_20260708_report.json`
  - Stage reward: `data/tmp/state_local_nodup_failed6_stage_reward_audit_20260708.json`
  - Result: `0/6` success, average `12.167` steps, workflow `0.833`, tool_valid `0.994`, one new `budget_check_repeats_previous_verified_selection` error.
  - Baseline failed6 subset: `0/6`, average `11.667` steps, workflow `1.0`, stage progress about `0.284`.
  - Decision: reverted. Hard-blocking duplicate verified bundles reduced workflow validity and did not improve success or length.
- Initial SELECT `candidate_shop_groups`:
  - Hypothesis: expose compact shop grouping before a shop-voucher error so the model can choose same-shop bundles earlier.
  - Targeted rollout: `data/tmp/state_local_shopgroups_remote_gpt55medium_failed6_w6_s15_20260708_rollout.jsonl`
  - Runner report: `data/tmp/state_local_shopgroups_remote_gpt55medium_failed6_w6_s15_20260708_report.json`
  - Stage reward: `data/tmp/state_local_shopgroups_failed6_stage_reward_audit_20260708.json`
  - Result: `0/6` success, average `14.0` steps, stage progress `0.292`, format `0.989`, tool_valid `0.993`, runner errors `repeated_search_not_allowed:1`, `exactly_one_think_block_required:1`.
  - Baseline failed6 subset: `0/6`, average `11.667` steps, stage progress about `0.284`, format/tool/workflow `1.0`.
  - Decision: reverted. Tiny stage-progress gain did not justify much longer trajectories and new errors.
- Aggressive first-decision final-only gate:
  - Hypothesis: after any `within_budget=true` verified selection, force DECISION to recommend and terminate.
  - Rollout: `data/tmp/state_local_finalonly_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
  - Stage reward: `data/tmp/state_local_finalonly_broad16_stage_reward_audit_20260708.json`
  - Result: average steps improved to `3.6875`, but success fell to `9/16`.
  - Failure mode: row6 selected a wrong third item in SELECT; final-only prevented DECISION from searching a correction.
  - Decision: reverted/refined. Initial DECISION must retain the ability to self-correct.
- One-retry final-only gate:
  - Hypothesis: only force final after at least one DECISION retry search.
  - Rollout: `data/tmp/state_local_retryfinal_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
  - Stage reward: `data/tmp/state_local_retryfinal_broad16_stage_reward_audit_20260708.json`
  - Result: average steps `4.5`, but success still `9/16`; row6 remained the regression after one retry.
  - Decision: reverted/refined to require at least two DECISION retry `find_product` calls.
- Force voucher usage:
  - Hypothesis: if `budget_check` shows a voucher structure but `voucher_used=false`, force DECISION back to search because cheap no-voucher bundles may be semantically wrong.
  - Targeted row: broad16 row8.
  - Rollout: `data/tmp/state_local_voucherused_row8_remote_gpt55medium_w1_s15_20260708_rollout.jsonl`
  - Stage reward: `data/tmp/state_local_voucherused_row8_stage_reward_audit_20260708.json`
  - Result: row8 stayed `0/1`, progress stayed `0.1333`, steps increased from `6` to `9`, and format fell to `0.8889`.
  - Decision: reverted. The structural signal was real, but forcing voucher use did not recover the missing same-shop product and made trajectory longer.
- Retry SELECT history top1 anchors:
  - Hypothesis: retry SELECT loses unselected gold candidates from earlier searches; preserving the top1 product from each historical non-empty search could recover cases like row13 without increasing displayed candidate count.
  - Targeted rollout: `data/tmp/state_local_anchor_remote_gpt55medium_failed6_w6_s15_20260708_rollout.jsonl`
  - Stage reward: `data/tmp/state_local_anchor_failed6_stage_reward_audit_20260708.json`
  - Result: `0/6` success, average steps `9.667`, stage progress `0.159`, workflow `0.5`.
  - Decision: reverted. Reintroducing historical unselected candidates created stale-candidate workflow issues and sharply reduced progress.

## Current State

Keep only evidence-backed harness changes. The latest code should match the retry2-final validated harness result: `10/16` success, average `4.1875` steps on broad16 with the original `max_steps=15` evaluation setting.

Current recommended runtime config for `gpt-5.5-medium` broad16:

- `max_steps=15`
- `max_candidates=10`
- `workers=16`

Evidence: retry2-final with `max_steps=15` preserves hard success at `10/16`, reduces average steps from `6.625` to `4.1875`, improves stage progress from `0.7159` to `0.7416`, and preserves `format=1.0`, `tool_valid=1.0`, and `workflow_valid=1.0`. New `max_steps=8` and `max_steps=10` experiments both dropped to `9/16`, so the rollout cap should not be lowered yet.

## Config Experiment: max_candidates=8

- Hypothesis: reducing displayed candidates from 10 to 8 may reduce selection noise and shorten trajectories while preserving enough balanced candidates.
- Change: no code change; run current validated harness with `--max-candidates 8`.
- Rollout: `data/tmp/state_local_mc8_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
- Runner report: `data/tmp/state_local_mc8_remote_gpt55medium_broad16_w16_s15_20260708_report.json`
- Hard reward: `data/tmp/state_local_mc8_remote_gpt55medium_broad16_w16_s15_20260708_reward_report.json`
- Stage reward: `data/tmp/state_local_mc8_broad16_stage_reward_audit_20260708.json`
- Result:
  - Hard success: `10/16`, same as baseline.
  - Average steps: `7.4375`, worse than baseline `6.625`.
  - Hard format: `0.9844`, worse than baseline `1.0`.
  - Stage progress: `0.7052`, worse than baseline `0.7159`.
  - Runner errors: `repeated_search_not_allowed:1`, `exactly_one_think_block_required:1`.
- Decision: do not use. Keep `max_candidates=10`.

Additional visibility audit on baseline failures:

- Gold products often already appeared in raw `find_product` results and sometimes in visible top10 `candidate_pool`.
- Examples:
  - row3: gold swing arm visible in step2 candidate pool; gold phone case visible in step10 candidate pool.
  - row10: gold cap and dress visible in step2 candidate pool; dress remains visible in later retries.
  - row13: gold toner visible in step2; gold hair mask and lip oil visible in step4.
- Conclusion: remaining failures are not primarily caused by top10 candidate truncation. Tightening to 8 harmed performance; expanding candidates is unlikely to be a high-signal next change without increasing context.

## Config Experiment: max_steps=10

- Hypothesis: current successful broad16 trajectories complete within 5 steps, while most 15-step trajectories still fail. Reducing max steps from 15 to 10 may preserve success while saving RL memory/compute on long failures.
- Change: no code change; run current validated harness with `--max-steps 10`, `--max-candidates 10`.
- Keep criterion: success does not drop from `10/16`, and average steps decreases meaningfully.
- Rollout: `data/tmp/state_local_s10_remote_gpt55medium_broad16_w16_s10_20260708_rollout.jsonl`
- Runner report: `data/tmp/state_local_s10_remote_gpt55medium_broad16_w16_s10_20260708_report.json`
- Hard reward: `data/tmp/state_local_s10_remote_gpt55medium_broad16_w16_s10_20260708_reward_report.json`
- Stage reward: `data/tmp/state_local_s10_broad16_stage_reward_audit_20260708.json`
- Result:
  - Hard success: `10/16`, same as baseline.
  - Average steps: `5.8125`, better than baseline `6.625`.
  - Format/tool/workflow: `1.0`.
  - Stage progress: `0.7150`, essentially tied with baseline `0.7159`.
  - Runner errors: `{}`.
- Decision: positive. Candidate recommended runtime config unless a stricter max-step experiment preserves success with shorter trajectories.

## Config Experiment: max_steps=8

- Hypothesis: successful broad16 cases are still all under 8 steps; reducing from 10 to 8 may save more failed-trajectory compute without success loss.
- Change: no code change; run current validated harness with `--max-steps 8`, `--max-candidates 10`.
- Keep criterion: success remains `10/16`; average steps improves over `5.8125`.
- Rollout: `data/tmp/state_local_s8_remote_gpt55medium_broad16_w16_s8_20260708_rollout.jsonl`
- Runner report: `data/tmp/state_local_s8_remote_gpt55medium_broad16_w16_s8_20260708_report.json`
- Hard reward: `data/tmp/state_local_s8_remote_gpt55medium_broad16_w16_s8_20260708_reward_report.json`
- Stage reward: `data/tmp/state_local_s8_broad16_stage_reward_audit_20260708.json`
- Result:
  - Hard success: `9/16`, worse than baseline and max_steps=10.
  - Average steps: `5.25`, shorter than max_steps=10 but with success loss.
  - Format: `0.9922`, worse than max_steps=10.
  - Stage progress: `0.6858`, worse than max_steps=10 `0.7150`.
  - Runner errors: `exactly_one_think_block_required:1`.
- Decision: do not use. `max_steps=10` is the current best length/success tradeoff.

## Reward Shaping Audit

Added a reusable staged reward audit script:

- Script: `scripts/analyze_state_local_stage_rewards.py`
- Failure audit script: `scripts/audit_state_local_failures.py`
- Input rollout: `data/tmp/state_local_recoverfmt_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
- Output: `data/tmp/state_local_recoverfmt_broad16_stage_reward_audit_script_20260708.json`
- Product cache generated from structured rollout observations plus gold product lookup:
  `data/tmp/state_local_recoverfmt_broad16_stage_product_cache_20260708.json`

Current validated broad16 staged reward summary after retry-final DECISION gate:

- `success`: `0.625`
- `progress`: `0.7416`
- `find_correct`: `0.8490`
- `view_confirmed`: `0.7316`
- `budget_correct`: `0.7316`
- `recommend_correct`: `0.7316`
- `terminate_complete`: `0.5972`
- `workflow_valid`: `1.0`
- `format`: `1.0`
- `tool_valid`: `1.0`
- `steps`: `4.1875`

Interpretation:

- Search is mostly solved on broad16; the main remaining gap is not protocol/tool validity.
- Failed trajectories often have partial find/view/budget progress but fail to form a correct final recommendation, or terminate with a semantically close but reward-wrong bundle.
- Reward shaping should preserve the current stage gates and step penalty; current evidence does not support relaxing success or adding prompt-heavy guidance.

Retry2-final failure audit:

- Output: `data/tmp/state_local_retry2final_broad16_failure_audit_20260708.json`
- Summary: `6` failures; `4` used the final-only gate; `5/6` failures had at least one gold product never observed in structured `find_product` results; only row13 had all gold products observed.
- Structured failure modes:
  - `search_recall_gap`: `5/6` failures, rows 3, 7, 8, 10, 14.
  - `final_selection_after_full_recall_gap`: `1/6` failures, row13.
- Implication: most remaining failures are not caused by DECISION protocol or candidate truncation. They need better search/query generation or sample curation, not more DECISION prompt text.
- Unclear/mismatched samples to deprioritize for harness changes:
  - row7: query asks beige/white girls shoes, gold shoe title says black color.
  - row13: query asks black travel shaver, but the gold item is a professional hair clipper/trimmer; model-selected portable shaver is semantically plausible though reward-wrong.

Filtered harness audit:

- Skip list: `proposal/harness_skip_cases_20260708.json`
- Summary script: `scripts/summarize_stage_report_with_skiplist.py`
- Output: `data/tmp/state_local_retry2final_broad16_stage_reward_filtered_20260708.json`
- Excluding row7 and row13 for harness-optimization decisions:
  - Kept rows: `14`
  - Hard success: `10/14` (`0.7143`)
  - Stage progress: `0.7952`
  - Format/tool/workflow: `1.0 / 1.0 / 1.0`
  - Average steps: `4.2143`
  - Remaining failure modes: `search_recall_gap: 4`, `success: 10`
- Interpretation: after removing semantically unclear cases from harness decision-making, every remaining failure is search-recall dominated. Do not add more DECISION/SELECT control-flow constraints based on these failures unless a future rollout shows a non-search-recall failure mode.

Reward shaping direction:

- Keep protocol, workflow, format, and tool-validity gates as-is; current retry2-final rollout is already `1.0` on all of them.
- Keep the step penalty because retry2-final demonstrated it can reduce actual trajectory length without lowering success.
- `shoppingbench_query.compute_score` now returns a non-reward diagnostic field `structured_failure_mode`; `scripts/analyze_state_local_stage_rewards.py` reports aggregate `failure_modes`.
- `scripts/analyze_verl_query_rollouts.py` now preserves query-level and aggregate `structured_failure_modes` for validation dumps, without changing checkpoint-selection rules.
- Current retry2-final failure modes from the stage audit:
  - `success`: `10`
  - `search_recall_gap`: `5`
  - `final_selection_after_full_recall_gap`: `1`
- For training diagnostics, track `structured_failure_mode` from `scripts/audit_state_local_failures.py` alongside staged reward:
  - `search_recall_gap` should push improvements in search/query generation, not DECISION prompt text.
  - `final_selection_after_full_recall_gap` should be inspected for semantic ambiguity before adding stricter harness rules.
- Do not optimize harness against row7 and row13 unless the dataset/gold mismatch is resolved; they are poor evidence for general harness behavior.

Search recall ablation:

- Script: `scripts/audit_search_filter_ablations.py`
- Output: `data/tmp/state_local_retry2final_search_filter_ablation_20260708.json`
- Method: replay each failed `find_product` call and structured variants without optional filters (`service`, `shop_id`, `sort`, `price`) plus next page, then check whether missing gold product ids appear in returned tool results.
- Result: `0/5` search-recall failures recovered any missing gold; `variant_hit_counts={}`.
- Implication: remaining `search_recall_gap` is not caused by optional filters or page-1 truncation in a way the harness can fix generically. This supports keeping prior page-2 and candidate-expansion experiments reverted.

## Kept Experiment: retry2-final DECISION gate

- Hypothesis: after multiple DECISION retry searches, continuing to search rarely recovers success and mostly lengthens trajectories. Force final recommendation only after at least two structured DECISION `find_product` calls and a later verified `within_budget=true` bundle.
- Counterfactual on current validated rollout:
  - Success preserved at `10/16`.
  - Average steps would drop from `6.625` to `4.25`.
  - Stage progress would improve from `0.7159` to `0.7499`.
- Rollout: `data/tmp/state_local_retry2final_remote_gpt55medium_broad16_w16_s15_20260708_rollout.jsonl`
- Runner report: `data/tmp/state_local_retry2final_remote_gpt55medium_broad16_w16_s15_20260708_report.json`
- Stage reward: `data/tmp/state_local_retry2final_broad16_stage_reward_audit_20260708.json`
- Result:
  - Hard success: `10/16`, same as baseline.
  - Average steps: `4.1875`, better than baseline `6.625`.
  - Stage progress: `0.7416`, better than baseline `0.7159`.
  - Format/tool/workflow: all `1.0`.
  - Runner errors: `{}`.
- Decision: keep.

## Config Experiment: retry2-final max_steps=8

- Change: no code change; run retry2-final harness with `--max-steps 8`.
- Rollout: `data/tmp/state_local_retry2final_s8_remote_gpt55medium_broad16_w16_s8_20260708_rollout.jsonl`
- Stage reward: `data/tmp/state_local_retry2final_s8_broad16_stage_reward_audit_20260708.json`
- Result: success `9/16`, average steps `4.5625`, stage progress `0.6590`.
- Decision: do not use.

## Config Experiment: retry2-final max_steps=10

- Change: no code change; run retry2-final harness with `--max-steps 10`.
- Rollout: `data/tmp/state_local_retry2final_s10_remote_gpt55medium_broad16_w16_s10_20260708_rollout.jsonl`
- Stage reward: `data/tmp/state_local_retry2final_s10_broad16_stage_reward_audit_20260708.json`
- Result: success `9/16`, average steps `4.5625`, stage progress `0.6859`.
- Decision: do not use. Keep `max_steps=15` for now; the kept retry2-final gate lowers actual average trajectory length without reducing the rollout cap.
