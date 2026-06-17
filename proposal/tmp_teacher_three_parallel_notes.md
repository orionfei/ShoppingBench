# Teacher Voucher Hard20 Notes

Built with online `history_compression=state_folded`; no model API was called.

Trajectory pattern: search all target candidates from environment, verify details and budget, then recommend and terminate.

Compression observations:
- Step 1 has no `<state>` because no assistant/tool history exists yet.
- Step 2 state contains search candidates and full budget candidates.
- Step 3 state can trust the budget only when python output matches observed prices and shop ids.

Teacher style constraints:
- `<think>` is short and states the next action basis.
- Gold answers are used only by the builder to choose actions; prompts and thoughts do not mention reward metadata.
- Product ids appear only after environment search observations make them visible.
