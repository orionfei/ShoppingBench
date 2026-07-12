#!/usr/bin/env python3
"""Create a deterministic, stratified 8-query concurrency smoke panel."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
source = pd.read_parquet(ROOT / "dataset/shoppingbench_query_rl_v3/validation.parquet")
selected = []
for count in (1, 2, 3, 4):
    group = source[source.extra_info.map(lambda value: int(value["product_count"]) == count)]
    for voucher in ("platform", "shop"):
        row = group[group.extra_info.map(lambda value: value["voucher_type"] == voucher)].iloc[0]
        selected.append(row)
output = ROOT / "dataset/probe/rl_v3_worker_smoke8"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(selected).to_parquet(output / "probe.parquet", index=False)
print({"rows": len(selected), "output": str(output / "probe.parquet")})
