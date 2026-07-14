# Step108 Outcome-Only GSPO 实验：从 token ratio 到 sequence ratio

## 1. 为什么在 GRPO 和 DAPO 之后尝试 GSPO

上一轮实验已经说明：DAPO 式 mixed-group 动态采样可以避免把 all-fail/all-success group 送入 old-log-prob 和反向传播，但没有解决最终泛化问题。DAPO-GRPO 的最佳 val64 checkpoint 为 step23，terminal ASR 从 33.01% 升到 34.77%；然而同一 checkpoint 在 test250 上为 8.25%，未超过 Step108 的 8.30%。

这一次不再同时改变数据、reward、advantage 或动态采样。实验保留：

- Step108、RL-v3 train1414、val64；
- 二元 `terminal_asr = paper_asr * terminate_success`；
- G=8、train `temperature=0.4/top_p=0.95`、validation `0.2/0.9`；
- DAPO mixed-only buffer、有效batch32、PPO mini-batch16；
- LR 1e-6、无 KL loss、无 reward shaping。

唯一算法变量是把 DAPO-GRPO 的 token-level PPO objective 换成 GSPO objective。因此准确名称是 **GSPO + DAPO dynamic sampling**，不能把动态采样本身说成 GSPO。

## 2. GSPO 的核心原理

GRPO 对每个 token 使用 importance ratio：

```text
r_i,t = exp(log pi_theta(y_i,t) - log pi_old(y_i,t))
```

但 outcome reward 是整条 trajectory 的奖励。GSPO 将 importance sampling、clipping 和 reward 的基本单位统一为完整序列，使用长度归一化的序列几何平均：

```text
s_i = exp(
  sum_t mask_i,t * (log pi_theta(y_i,t) - log pi_old(y_i,t))
  / sum_t mask_i,t
)
```

随后整条序列共享同一个 clipping 决策。长度归一化很重要，否则长response的likelihood ratio会随长度指数放大，无法使用统一clip范围。

Qwen GSPO 论文指出，GRPO的token importance weight基于每个next-token分布中的单个sample，不能真正完成importance-sampling估计，反而会引入随序列长度累积的高方差梯度。GSPO让同一response内所有token获得相同的序列级权重。论文还强调，GSPO对MoE routing变化以及训练/推理engine的精度差异尤其稳健。

本项目是dense Qwen3-4B，不存在MoE routing不稳定，因此合理预期是：GSPO可能让长trajectory更新更稳定，但它不会自动修复训练数据分布偏差、val/test错配或binary reward稀疏性。

资料：

- [GSPO论文（arXiv:2507.18071）](https://arxiv.org/abs/2507.18071)
- [Qwen官方GSPO介绍](https://qwen.ai/blog?id=gspo)
- [VERL官方仓库（当前版本已列出GSPO支持）](https://github.com/verl-project/verl)
- [VERL GSPO实践文档](https://verl.readthedocs.io/en/latest/ascend_tutorial/model_support/examples/gspo_optimization_practice.html)

## 3. 为什么不能沿用 DAPO 的 clip range

DAPO-GRPO 使用 token ratio `[0.8, 1.28]`，即 `clip_low=0.2`、`clip_high=0.28`。GSPO论文的序列ratio使用：

```text
clip_low  = 3e-4
clip_high = 4e-4
ratio     = [0.9997, 1.0004]
```

两者相差约三个数量级。原因不是GSPO“保守一千倍”，而是token ratio和长度归一化sequence ratio的数值尺度不同。若直接给GSPO使用0.2/0.28，实验名义上启用了GSPO，实际上sequence clipping几乎失效。

## 4. 向当前固定VERL版本做最小回移

当前项目的固定VERL版本已有policy-loss registry，但没有`gspo`实现。为了可回退，没有整体升级VERL，而是：

1. 在`core_algos.py`注册独立`gspo` loss；
2. 用response mask排除padding，计算长度归一化sequence log ratio；
3. 采用论文的GSPO-token detach写法，在现有token-shaped actor接口内获得与sequence GSPO等价的数值和梯度；
4. 强制使用`seq-mean-token-mean`，避免长response因为token更多而权重更大；
5. 只有`actor.policy_loss.loss_mode=gspo`时启用，默认`vanilla`路径不变。

回归测试覆盖：loss注册、不同长度sequence的等权总梯度、上下界整条sequence clipping、padding不影响ratio。3项测试全部通过。真正训练的第1步又覆盖了rollout、binary reward、old log-prob、GSPO backward和optimizer step。

## 5. 实际实验经过：先证明算法能跑，再解决吞吐瓶颈

### 5.1 最小可行 GSPO backward

首个 run 为 `rollouts/step108_outcome_gspo_v3_dapo_20260714_055533`，使用 effective batch32。它完成 Step0 和第一个真实 optimizer step，证明本地回移实现可以完成 rollout、二元 reward、old log-prob、GSPO backward 和更新：

| 指标 | 首个 GSPO step1 |
|---|---:|
| raw / accepted group | 96 / 32 |
| mixed group | 39 |
| train reward | 42.58% |
| entropy | 0.1182 |
| grad norm | 0.9258 |
| sequence clip fraction | 20.16% |
| generation / actor update | 707.2s / 83.6s |

它没有 OOM、NaN 或梯度异常，但 rollout 远慢于更新，因此没有继续把慢配置跑满。

### 5.2 失败的并发加速假设

随后分别测试了 `max_num_seqs=16`、agent worker64、Val64 一次性送入和关闭稳定哈希路由：

- agent workers 从16增到64没有数量级收益，因为 worker 本身是异步调度层；
- `max_num_seqs=16` 的首个128-trajectory分片约5分钟，比稳定的8更慢；更大的decode batch降低了单token效率，并放大长轨迹尾部；
- Val64一次排队形成一个更长的straggler尾巴；
- 动态路由改善了瞬时卡间均衡，但没有稳定缩短完整Val64，而且改变了已经验证过的并发语义。

因此最终保留 `max_num_seqs=8`、Val batch16、稳定哈希路由。`max_num_seqs` 是每个 vLLM engine 同时调度的活跃序列上限，不是训练 batch，也不是 G。

### 5.3 真正瓶颈是商品搜索服务

8卡 rollout 时，GPU 会在模型生成阶段满载，随后大量等待单个搜索服务。旧实现每次只消费最多50个商品，却总让 Lucene 返回5000个 hit；64个 agent 并发时服务探活会超时。

修复保持返回语义不变：

1. 无 shop/price/service 后过滤时，Lucene 只取50个 hit；这与旧逻辑逐项完全等价。
2. 有过滤时先取512；不足50个结果才回退到5000。
3. 相同搜索和商品详情增加 single-flight 与 LRU cache，G=8 的重复工具调用只计算一次。
4. waitress worker 限制为64，避免无限线程争用。

五组代表性请求在修复前后的 response SHA256 完全一致。未缓存单请求从约0.11--0.19秒降到0.015--0.022秒；64个不同搜索并发墙钟0.538秒，64个相同搜索只需0.049秒。修复后正式训练期间搜索探活保持正常。

### 5.4 用历史 raw rollout 改善 DAPO proposal distribution

旧 DAPO-GRPO run 保留了3680个原始G=8 group，其中1771 mixed、1664 all-fail、245 all-success。1396条query被采样过；845条至少产生过一次mixed。这845条上的历史mixed率为78.16%。

据此生成 `train_dapo_mixed_priority.parquet`，它只是在线proposal view：

- canonical train1414完全保留；
- view内845条query仍然唯一，没有复制样本；
- 每次仍重新由当前policy采样G=8并计算outcome；
- all-fail/all-success仍不能进入old-log-prob或更新。

这不是复用旧trajectory做off-policy训练，而是利用历史诊断减少明显无效的在线proposal。代价是训练分布更偏向已知边界样本，所以最终泛化仍必须看未参与选择的Val64。

## 6. 最终 8×A800 pilot 配置

正式 run：`rollouts/step108_outcome_gspo_v3_dapo_fast64_20260714_073037`。

| 项目 | 最终值 |
|---|---:|
| checkpoint | Step108 |
| canonical train / proposal / val | 1414 / 845 / 64 |
| G / effective batch / PPO mini-batch | 8 / 64 / 32 |
| rollout engines | 8，TP=1，engine n=1 |
| max_num_seqs / agent workers | 8 / 64 |
| train sampling | temperature 0.4，top-p 0.95 |
| validation sampling | temperature 0.2，top-p 0.9 |
| response / turns | 10240 / 15 |
| LR / KL loss | 1e-6 / 关闭 |
| GSPO sequence ratio clip | `[0.9997,1.0004]` |
| optimizer steps | 12（768个accepted mixed group） |
| validation / save | step6、12 |

相同并发配置的独立 Step0 fast64 probe 得到512条trajectory：terminal/paper ASR 32.03%、mixed rate51.56%、format 100%、truncation 6.84%。正式run为节省时间复用这个Step0，不重复生成。

最初设 `gen_batch=96`，最多2批。12步中9步一批填满，Step3/8/12需要第二批。经验mixed率约65%--68%，96个query的期望mixed数过于贴近64。最终默认改为112：在同一经验率下期望73--76个mixed，能以约17%的单批余量避免整批重采样。该修改只影响后续launcher，已完成run的manifest仍如实记录96。

## 7. 最终结果

### 7.1 Validation

| 指标 | Step0 probe | Step6 | Step12 |
|---|---:|---:|---:|
| terminal / paper ASR | 32.03% | 32.23% | **33.79%** |
| 相对 Step0 | -- | +0.20pp | **+1.76pp** |
| mixed-group rate | 51.56% | 46.88% | **32.81%** |
| format | 100% | 99.84% | 100% |
| workflow valid | -- | 91.02% | 91.41% |
| truncation | 6.84% | 7.81% | **5.47%** |
| JSON failure / server error | 0.39% / 0% | 0.20% / 0% | 0% / 0% |

Step12比Step0多9/512次成功，但bootstrap区间明显重叠，且没有达到预设的+5pp成功门槛。更值得警惕的是mixed rate降到32.81%：ASR略升的同时，group更容易变成all-fail或all-success，可用于继续学习的边界样本在减少。

### 7.2 训练健康度

12步统计：

- accepted train reward均值45.31%，范围41.41%--49.80%；
- entropy均值0.1193，范围0.1130--0.1243，没有坍缩；
- grad norm均值0.535，最大0.647，没有爆炸；
- GSPO clip fraction均值19.48%，范围12.79%--23.78%；lower clip始终为0；
- diagnostic PPO KL接近0；这里仍然没有KL loss。

因此本次GSPO不是“训练跑坏了”。它稳定地做了12次更新，只是稳定优化没有转化成足够大的验证收益。

### 7.3 吞吐和成本

| 项目 | 数值 |
|---|---:|
| 总墙钟 | 8621s（2h23m41s） |
| raw groups / trajectories | 1440 / 11520 |
| accepted mixed groups / trajectories | 768 / 6144 |
| generation | 4974s（82.9min） |
| actor update | 1864s（31.1min） |
| reward scorer | 313s（5.2min） |
| 两次Val64 | 764s（12.7min） |
| 平均 response tokens | 6599 |

训练decode和actor update阶段8卡分别维持约85%--100%和99%--100%。12步中3步补批；若按已锁定的gen112均为一批，raw group预计为1344，比本次再少6.7%，并显著降低step时长方差。

## 8. 与 DAPO-GRPO 的结论

旧 DAPO-GRPO 在step23（736个accepted group）达到Val terminal ASR 34.77%、mixed rate46.88%。本次GSPO在相近预算768个accepted group后为33.79%、mixed rate32.81%。两次run的GPU数、proposal distribution和Step0采样并不完全相同，因此不是严格单变量A/B；但现有证据不支持“GSPO优于GRPO”。

本次不运行test250，原因是Val提升只有1.76pp且没有超过已有GRPO最佳点。上一轮已经证明，在Val没有明确优势时消耗一次正式test并不能解决分布错配问题。

面试时最诚实、也最有价值的表述是：

> 我们把论文级GSPO sequence objective最小回移到固定VERL版本，验证它在10k-token、多轮工具轨迹上数值稳定；随后通过raw trajectory审计和搜索服务profile，把主要基础设施瓶颈从搜索端消掉，并把无效proposal显著减少。但在相近有效group预算下，GSPO只让Val ASR提升1.76pp，低于DAPO-GRPO最佳点，同时mixed rate下降。因此我们没有把“更稳定的优化器”误报为“更好的策略”，而是将后续重点重新放回数据覆盖、query难度边界和val/test分布一致性。

## 9. 产物与可回退性

- 原始run、manifest、trajectory和监控：`rollouts/step108_outcome_gspo_v3_dapo_fast64_20260714_073037/`
- 分析JSON/CSV：`reports/step108_outcome_gspo_v3_dapo_fast64_20260714_073037/analysis/`
- PNG/SVG图：`reports/step108_outcome_gspo_v3_dapo_fast64_20260714_073037/figures/`
- best checkpoint：`checkpoints/shoppingbench-rl-v3-gspo/step108_outcome_gspo_v3_dapo_fast64_20260714_073037_b64_pilot12/best`
- canonical train1414与Step108均未修改；GSPO和priority proposal均由显式launcher/config开启，可直接fallback到原GRPO路径。
