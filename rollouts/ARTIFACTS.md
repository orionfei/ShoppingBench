# Versioned experiment artifacts

This directory keeps the reproducibility-critical outputs for the final RL v3 DAPO run while model
checkpoints and merged model weights remain excluded from Git.

Included artifacts:

- formal training and test manifests;
- trainer and system metrics;
- console logs;
- every accepted, raw-dynamic, validation, and test250 trajectory as individually gzipped JSONL;
- completion sentinels for the one-shot test250 runs.

Use `gzip -dk <file>.jsonl.gz` to restore an individual JSONL file; gzip's embedded CRC verifies each
stream during decompression. Historical exploratory raw rollouts are not duplicated here when
their reports, plots, and conclusions are already versioned under `reports/` and `docs/`.
