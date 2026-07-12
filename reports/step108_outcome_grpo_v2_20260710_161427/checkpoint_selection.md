# Formal GRPO checkpoint selection

Selected `global_step_60`. Registered +5pp success achieved: **False**. Training stopped at two epochs because the last two saved milestones improved by only 0.78pp, below the 2pp extension gate.

| Step | Terminal ASR | Mixed | Pass@8 | Infra failure | Healthy | Saved |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 36.72% | 62.50% | 68.75% | 1.56% | no | no |
| 10 | 42.97% | 43.75% | 56.25% | 0.78% | yes | no |
| 20 | 42.19% | 56.25% | 62.50% | 1.56% | no | yes |
| 30 | 39.06% | 50.00% | 62.50% | 0.78% | yes | no |
| 40 | 39.84% | 50.00% | 62.50% | 0.78% | yes | yes |
| 50 | 41.41% | 50.00% | 62.50% | 0.00% | yes | no |
| 60 | 41.41% | 50.00% | 62.50% | 0.78% | yes | yes |
| 70 | 37.50% | 56.25% | 62.50% | 1.56% | no | no |
| 80 | 42.19% | 56.25% | 68.75% | 2.34% | no | yes |

Pruned non-selected RL checkpoints: `[20, 40, 80]`. The original SFT checkpoint was not modified.
