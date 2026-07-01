# Teacher Voucher Train Clean691 Final Audit

Status: PASSED

## Final Training Files

- Rollout: `data/teacher_voucher_train_clean691_state_folded.jsonl`
- Synthesize: `data/teacher_voucher_train_clean691_synthesize.jsonl`
- verl SFT dataset: `dataset/shoppingbench_sft_state_folded`
- Deep audit JSON: `data/teacher_voucher_train_clean691_deep_audit.json`

## Counts

- Source training rows: 750
- Clean trajectories included: 691
- Hard clean: 227
- Non-hard clean: 464
- Assistant steps: 2159
- Total assistant steps: 2159
- Step histogram: {'3': 605, '4': 86}
- Excluded rows: 59

## Verification

- Deep audit problem count: 0
- Format/schema/source alignment/recommend ids/budget checks: passed
- `tool_call_id` is not visible in `completion.content`
- Unique `<think>` strings: 2159
- Empty `<think>` strings: 0
- SFT dependency check: True
- SFT local eval check: True
- Official voucher eval shard 1: 400 cases, all metrics 1.000
- Official voucher eval shard 2: 291 cases, all metrics 1.000

## Qwen3-4B Tokenizer

- Prompt max / p99 / p95 / mean: {'max': 3796, 'p99': 3088, 'p95': 2757, 'mean': 1924.5}
- Response max / p99 / p95 / mean: {'max': 431, 'p99': 368, 'p95': 357, 'mean': 196.3}
- Recommended tight verl setting: `data.max_prompt_length=8192`, `data.max_response_length=512`

## Excluded Rows

Hard unconstructable source lines:

`298,408,459,494,526,559,572,654,677,681`

Non-hard unresolved source lines:

`45,58,74,76,120,160,161,166,179,206,210,220,230,237,238,241,253,275,280,286,299,332,362,365,394,430,441,444,445,449,474,491,493,510,547,556,558,567,569,608,609,610,618,627,691,692,704,744,747`
