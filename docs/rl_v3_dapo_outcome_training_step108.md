# Step108 Outcome-Only DAPO RL v3 实验记录

Date: 2026-07-11

## 1. 为什么从固定GRPO改为动态采样

上一轮643-query训练中，1280个G8 group有600个all-fail、60个all-success。这些group的组内
advantage为零，却仍占用了old-log-prob、反向传播和optimizer batch。RL v3保留纯二元
`terminal_asr = paper_asr × terminate_success`，但采用DAPO式动态采样：先生成和评分，只有
mixed group进入参数更新。

本实现不是离线永久删除难题。train1414中的query会随policy变化反复被抽取；某题本step全失败，
未来仍可能成为mixed并贡献梯度。

## 2. 数据

| Split | Queries | Unique gold products | Source |
|---|---:|---:|---|
| train | 1414 | 3379 | RL v3静态合格候选，替换2条test冲突样本 |
| validation | 64 | 160 | 商品库重新采样，embedding maximin多样性 |
| test | 250 | 714 | `data/synthesize_voucher_test.jsonl` |

三者query和product overlap均为0。validation中1/2/3/4商品各16条，platform/shop与
fixed/percentage各32条；所有gold shop均未在train/test出现。

## 3. 实现

每次从128个无放回prompt候选中按32-query chunk顺序生成，最多4个chunk。每个query生成G8；
`0 < successes < 8`才进入buffer。buffer满32组后才计算old-log-prob、GRPO advantage和
backward。关闭dynamic sampling时保留原VERL路径。

固定配置：Step108、4×A800、G8、train `T=0.4/top_p=0.95`、validation
`T=0.2/top_p=0.9`、response 10240、15 assistant/user turns、`max_num_seqs=8`、LR 1e-6、
ratio clip `[0.8,1.28]`、无KL、无reward shaping、token-mean loss。

## 4. 启动前证据

- 20项reward、supervision和dynamic group测试通过。
- 4卡worker=8 smoke：format 100%、infra failure 0、mixed 75%，GPU利用率83%–93%。
- 单步真实更新：64个raw group中32 mixed、30 all-fail、2 all-success；只对32 mixed执行更新。
- smoke entropy 0.1114、grad norm 0.873、clip fraction 0.00248，无OOM/NaN。

正式Step0 val64：terminal/paper ASR 33.01%，mixed 40.63%，format 99.90%，workflow valid
90.23%，infra failure 0.20%，response P50 7161、P95 10240。

## 5. 正式训练时间线

正式run：`rollouts/step108_outcome_grpo_v3_dapo_20260711_054926`。

| Step | terminal/paper ASR | mixed groups | truncation | workflow valid | 结论 |
|---:|---:|---:|---:|---:|---|
| 0 | 33.01% | 40.63% | 7.23% | 90.23% | untouched Step108基线 |
| 11 | 31.25% | 40.63% | 7.23% | 90.43% | 短期波动，未改善 |
| 23 | 34.77% | 46.88% | 5.47% | 90.43% | `+1.76pp`，首个最佳checkpoint |
| 34 | 34.77% | 48.44% | 4.88% | 89.26% | ASR持平，mixed与长度健康度改善 |
| 45 | 33.98% | 42.19% | 6.84% | 90.82% | 相比最佳点回落，中期停止 |

截至step34，共生成2752个raw group，实际mixed yield为47.78%；平均每个有效更新需要
2.53个32-query generation chunk。这里必须区分两个容易混淆的指标：日志中的
`acceptance_rate=accepted_groups/generated_groups`只计算buffer最终取走的32组；真实采样效率是
`mixed_groups/generated_groups`，因为buffer装满时最后一个chunk中剩余的mixed也会保留诊断但不进入
本次更新。

step34仍未达到预注册的`Step0 +5pp`成功门槛，但entropy、clip、截断和基础设施均健康，因此不在
噪声较大的单个验证点提前停止，继续到step45执行中期判断。

step45相对Step0仅`+0.98pp`，相对step23/34反而`-0.78pp`，mixed rate也从48.44%降至
42.19%。最近两个验证点没有明确上升，且离`+5pp`成功门槛很远，因此按预注册中期规则在已保存
step45 checkpoint后停止，没有为了凑满“两个epoch”继续烧到step90。val64按terminal ASR选择的
最佳模型为step23；同分时优先更早checkpoint。

## 6. 动态采样实际效率与优化健康度

45个有效update共生成3680个raw query group，即29440条trajectory：

- all-fail 1664组、mixed 1771组、all-success 245组，真实mixed yield为48.13%。
- 每步只取32个mixed group，实际进入old-log-prob/backward的为1440组、11520条trajectory。
- 331个mixed group因最后一个chunk使buffer溢出而仅保留诊断。all-fail、all-success以及这些溢出
  mixed均未进入actor update。
- 因而DAPO避免了17920条trajectory的old-log-prob和backward，但没有避免它们的生成成本；平均
  每个update需要2.56个generation chunk，raw generation平均552秒，actor update平均148秒。

优化过程本身很健康：entropy均值0.1106、范围0.0982–0.1213；grad norm均值0.881，单点最大
2.775但没有持续爆炸；upper clip fraction均值0.215%；观测用PPO KL介于-4.12e-4和2.15e-4。
这里的PPO KL不是loss：训练没有reference-policy KL loss，也没有KL reward。

正式训练运行10.24小时。系统采样中的四卡平均利用率76.9%（包含验证、checkpoint和切换空窗），
P95为100%；显存P95 47.6GiB、最大55.2GiB，温度最高65°C。磁盘从40.6GiB降到22.5GiB，未触发
12GiB安全停止线。

## 7. 唯一一次test250结果

最佳checkpoint锁定后，untouched Step108和step23各在固定test250上只评测一次；每个模型都是
250 query × G8 = 2000条trajectory，采样固定为`T=0.2/top_p=0.9`。test没有参与停止或选模。

| Metric | Step108 | step23 | Difference |
|---|---:|---:|---:|
| paper / terminal ASR | 8.30% | 8.25% | -0.05pp |
| pass@8 | 20.00% | 20.40% | +0.40pp |
| mixed-group rate | 19.20% | 18.40% | -0.80pp |
| all-fail / mixed / all-success | 200 / 48 / 2 | 199 / 46 / 5 | — |
| format mean | 99.79% | 99.85% | +0.06pp |
| workflow valid | 81.05% | 79.50% | -1.55pp |
| infra failure | 0.50% | 1.05% | +0.55pp |
| token-limit noncompletion | 10.65% | 9.95% | -0.70pp |
| mean response tokens | 8031 | 8103 | +72 |

按query配对bootstrap，step23相对Step108的terminal ASR差为-0.05pp，95% CI为
`[-1.45pp, +1.25pp]`，bootstrap改善概率46.6%。所以不能说RL有效，也不能仅凭pass@8的+0.4pp
宣称提升；总体结论是无可检测泛化收益。

分桶揭示了更新方向：step23把单商品ASR从24.43%降到15.34%，但2/3/4商品分别从
11.96/6.41/2.40%升到12.86/7.19/3.21%；shop voucher从5.58%升到6.40%，platform voucher
从10.85%降到9.98%。这说明训练并非“参数完全没动”，而是偏向了训练中更常贡献mixed advantage的
复杂任务，同时破坏了简单任务，宏观均值相互抵消。

step23的infra failure为1.05%（5次server error、8次JSON decode、8次runaway），刚好越过1%健康门，
因此分析器将它标为gate-ineligible；即使忽略这0.05pp越界，其ASR也没有优于Step108。

## 8. 深层原因与反思

这次实验把两个问题分开了：

1. **工程问题解决了。** all-zero/all-one组没有再浪费old-log-prob和反向传播，buffer始终严格32组，
   没有partial update、OOM、NaN或entropy坍缩。DAPO路径本身是成功的。
2. **学习问题没有解决。** “mixed才更新”改变了被优化的数据分布。一个query成为mixed的概率由当前policy
   决定，因此更新样本不是原始1414条query的无偏样本，而是集中在当前决策边界、尤其是多商品和shop
   voucher任务。训练reward约0.4–0.5只能说明accepted mixed组内正负轨迹并存，不能说明全分布ASR上升。
3. **纯二元outcome信用分配仍然太弱。** 一条长轨迹只在最终商品、预算和terminate全部成功时得1。
   mixed组能产生梯度，但梯度无法直接区分搜索词、候选筛选、属性验证、voucher解析中哪一步值得保留。
4. **val64不是test250的可靠代理。** val由Qwen改写并按多样性构造，查询往往保留较强的商品标题线索；
   test是原始正式分布。Step108在val为33.01%，在test只有8.30%，巨大的绝对差说明两者难度和表述分布
   不同。64-query CI也很宽，step23的+1.76pp完全可能是噪声。数据零重叠防止了泄漏，却不自动保证
   validation代表性。
5. **DAPO节省更新成本，不节省生成成本。** 本次只有39.1%的raw trajectories进入backward，生成仍占
   绝大多数wall time；若没有更便宜的在线难度预估或更高mixed yield，训练仍然昂贵。

因此最诚实的面试结论不是“RL失败所以DAPO无效”，而是：我们修复了零advantage计算浪费，并用严格
test证明了边界样本上的纯outcome优化没有转化为总体泛化；进一步分析还定位了复杂任务上升、简单任务
退化以及validation分布错位。

## 9. 下一轮应如何改

- 先重建与正式test同源、同难度的validation；保持product-disjoint，但从正式生成过程分层留出，而不是
  另用更显式的Qwen描述。扩大query数或使用多个固定seed，避免1–2pp噪声驱动选模。
- 动态采样后按原始数据桶做配额或重要性修正，不能让“容易成为mixed”自动等价于“更值得训练”；至少分别
  约束1/2/3/4商品、platform/shop的accepted batch比例。
- 保留outcome-only主reward时，可先做trajectory-level failure attribution作为采样/curriculum诊断；如果仍
  不加reward shaping，就需要更大的有效batch和跨seed复验。若允许改变目标，再实验可验证的过程reward，
  但必须单独做消融，不能把它与DAPO收益混在一起。
- 把raw mixed yield、accepted/raw比例和按桶ASR设为一等监控指标；仅看accepted reward会产生错误乐观。
- 当前Step108应继续作为默认RL起点；step23只保留为这次实验的诊断checkpoint，不应替换生产基线。

## 10. 产物

- 数据：`dataset/shoppingbench_query_rl_v3/`
- 原始训练/验证轨迹：正式run目录下`train/`、`train/raw_dynamic/`和`validation/`
- 聚合报告：`reports/step108_outcome_grpo_v3_dapo_20260711_054926_b32/analysis/`
- 最终选模/停止报告：`reports/step108_outcome_grpo_v3_dapo_20260711_054926_b32/final_selection.json`
- 图像：`docs/figures/rl_v3_dapo_step108/`
- test250报告：`reports/step108_outcome_grpo_v3_dapo_20260711_054926_b32_test250/analysis.json`
- test250图像：`docs/figures/step108_outcome_grpo_v3_dapo_20260711_054926_b32_test250/`
- 最终一次性test250入口：`scripts/run_rl_v3_final_test250.sh`

需要明确区分：`actor/ppo_kl`只是诊断指标；训练没有KL loss，也没有KL reward。
