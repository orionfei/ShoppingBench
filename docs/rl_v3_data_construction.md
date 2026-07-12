# ShoppingBench RL v3 数据构建与 Learnability 筛选

Date: 2026-07-11

## 1. 为什么必须重做数据

正式 outcome-only GRPO 的 643 条 train query 虽然与 test75 的商品数、券类型分布相近，但它们
更像 benchmark，而不是适合当前 Step108 policy 的 curriculum。两 epoch、1280 个 G8 group 中：

| Group state | Groups | Rate |
|---|---:|---:|
| all-fail | 600 | 46.88% |
| mixed | 620 | 48.44% |
| all-success | 60 | 4.69% |

只有 mixed group 产生非零 GRPO advantage。260/643 条 query 在两次观察中始终全失败，16 条始终
全成功，只有 367 条至少出现一次 mixed。数据覆盖没有问题：637 条被采样两次、6 条一次；问题是
当前 policy 无法从大量 query 中采到可比较行为。

难度与 learnability 的关系也非常明确：

| Bucket | Query count | Observed ASR | Any mixed | Always all-fail |
|---|---:|---:|---:|---:|
| 1 product | 51 | 37.3% | 82.4% | 15.7% |
| 2 products | 195 | 34.6% | 63.6% | 30.8% |
| 3 products | 217 | 23.8% | 53.5% | 45.6% |
| 4 products | 180 | 14.8% | 47.2% | 51.7% |
| low constraints | 105 | 33.8% | 70.5% | 25.7% |
| medium constraints | 237 | 30.6% | 61.2% | 35.4% |
| high constraints | 301 | 18.9% | 49.2% | 49.5% |

RL v3 的目标不是降低最终 benchmark 难度，而是先构造能提供稳定 comparison signal 的训练分布，
再通过 curriculum 逐渐接近 test75。

## 2. 不变的训练语义

- 唯一 reward 仍为 `terminal_asr = paper_asr × terminate_success`。
- 不加入 progress、format、tool、length 或 step reward shaping。
- G=8、engine `n=1`、`max_num_seqs=8`，不再引入 G-sensitive 并发混杂。
- official product-disjoint test75 保持 untouched。
- query 生成 LLM 只负责自然语言改写；商品、SKU、attributes、service、shop、price、voucher 和
  ground truth 必须先从真实商品文档中确定，LLM 无权自由编造。

## 3. API 配置

ShoppingBench 目录没有 `.env`。运行时只读取：

```text
/root/project/ResearchHarness/.env
API_KEY   -> fallback query generator API key
API_BASE  -> OpenAI-compatible base URL
MODEL     -> qwen3.6-flash（本项目显式覆盖，不使用该 env 的 MODEL_NAME）
```

本轮发现该 env 中原有 key 返回 401，因此通过进程级 `RL_QUERY_API_KEY` 使用用户更新的 key；没有把新
key 写入 ShoppingBench 或 manifest。launcher 只记录 env 文件路径、provider URL
的 SHA256/host 和模型名。正式批量前必须通过 3-query smoke，并验证 JSON schema 和 query 内容。

## 4. Candidate 生成分布

第一批生成 1500 条候选，不直接称为最终 train。profile 为 `rl-v3-candidate`：

```text
product count:       1 / 2 / 3 / 4 = 20% / 35% / 30% / 15%
constraint level:    low / medium / high = 25% / 50% / 25%
voucher type:        platform / shop = 30% / 70%
discount type:       fixed / percentage = 40% / 60%
budget difficulty:   easy / medium / hard = 25% / 45% / 30%
```

每个候选使用唯一商品 ID。official test75 的所有 reward product ID 在抽样前加入 exclusion set，
从源头保证候选池与 test75 product-disjoint。

约束级别控制每个商品暴露给 query generator 的字段数：

- low：title 加 0–1 个 SKU/attribute/service 条件；
- medium：title 加 1–2 个条件；
- high：title 加 2–4 个条件。

候选池仍保留高难题，但不再让 3–4 商品、高复杂度同时占据训练主体。

## 5. 五阶段数据管线

### Stage A：确定性 plan

从 `resources/documents.jsonl` 抽取真实商品，生成：

- sampled product snapshot；
- evaluator reward ground truth；
- shop/platform voucher；
- threshold、fixed/percentage/cap；
- 可兑现的 price-after-voucher 与 budget；
- constraint complexity 和生成 bucket。

Stage A 不调用 API，可重复、可审计。

### Stage B：自然语言 query

使用 `qwen3.6-flash` 把结构化 requirements 改写成自然购物请求；改写时关闭 thinking，避免推理
token 挤占 512-token JSON 输出预算。voucher/budget 后缀由程序拼接，
不让 LLM 修改数值。输出必须是 `{"query": ...}`；生成支持按 `sample_id` 断点续跑。

### Stage C：静态与 evaluator 审计

每条 candidate 必须通过：

- query/reward/voucher schema；
- 商品 ID 存在，reward 字段可在商品文档中验证；
- shop voucher 的所有 gold 商品同店；
- voucher threshold/discount/cap/budget 重新计算一致；
- query、sample ID、product ID 去重；
- 与 official test75 product ID 零重叠；
- query 不泄露 product ID、ground-truth price 或 evaluator 字段名；
- LLM 没有遗漏主要商品需求或 voucher 文本。

静态审计失败进入 repair pool，不直接静默丢弃。

### Stage D：Step108 learnability probe

每个静态合格 candidate 用锁定的 Step108、`temperature=0.4, top_p=0.95` 跑 G8：

- mixed：直接获得 learnability observation；
- all-fail/all-success：换 seed 再跑一次 G8；
- 不允许用 G4 替代，因为 G 会改变 rollout 行为。

两轮后按经验成功率和 group state 分桶：

| Bucket | Definition | Intended use |
|---|---|---|
| core | 至少一个 mixed，经验 p 约 0.15–0.70 | 主训练 |
| frontier | 至少一次成功，但 p 较低 | 困难训练/探索 |
| easy | p 约 0.70–0.90 | 基础行为巩固 |
| saturated | 持续 all-success | 降权或加难 |
| zero-signal | 两轮共 16 条全失败 | 诊断/改写，不进入主训练 |

理论 mixed 概率为 `1 - p^8 - (1-p)^8`。最终选择同时考虑真实 mixed、可解性、行为覆盖和
query-family 去重，而不是只追求 raw reward variance。

### Stage E：最终 split

pilot 先构建：

```text
train512
core_validation64
target_validation64
official_test75 untouched
```

pilot 证明确实能学习后，扩展为：

```text
train 2000–2500
core_validation128
target_validation128
official_test75 untouched
```

最终 train 的目标桶配比：core 55%、frontier 20%、easy 15%、representative-hard 10%。split
按 product family 和 query family 隔离，不能只随机切行。

## 6. Pilot 数据验收门槛

- 静态 schema/evaluator audit 100% 通过；
- 与 test75 product overlap = 0；
- train512 中至少 70% query 在两轮 G8 中出现 mixed；
- zero-signal 不进入主 train，单独保存原因与 repair 建议；
- core/target validation 的商品与 train product-disjoint；
- 每个 bucket、商品数、券类型、discount 类型、constraint level 都有明确 quota；
- manifest 记录数据 hash、代码 hash、生成模型、API host、seed、商品文档 hash、exclusion hash、
  每阶段 accepted/rejected/repaired 数量；
- 所有中间 plan、API raw output、metadata 和审计报告可恢复，不覆盖旧 RL v2。

## 7. 本轮产物路径

```text
data/rl_v3/
  excluded_test75_product_ids.txt
  candidates_1500.plan.jsonl
  candidates_1500.query.jsonl
  candidates_1500.query.meta.jsonl
  candidates_1500.audit.jsonl
  candidates_1500.static_eligible.jsonl
  candidates_1500.repair.jsonl
  candidates_1500.audit_report.json
  manifest.json

dataset/shoppingbench_query_rl_v3_probe/
  # Stage108 G8 probe 后再创建
```

RL v2、正式训练轨迹和 Step108 checkpoint 均保持不变，随时可以 fallback 和做对照。

## 8. 2026-07-11 实际构建结果

本轮只完成 Stage A–C，到 Step108 rollout 与 RL 训练之前停止：

| Metric | Result |
|---|---:|
| deterministic candidates | 1500 |
| unique sampled products | 3598 |
| test75 product overlap | 0 |
| API generation accepted | 1500/1500 |
| API/JSON failed attempts | 0 |
| unique queries | 1500 |
| static eligible | 1414 (94.27%) |
| repair pool | 86 (5.73%) |

86 条 repair 样本全部因为完整复制至少一个商品标题。它们没有 schema、voucher、预算或 test 泄漏
错误，但会把购物搜索降级为精确标题匹配，可能制造虚假的 all-success，因此不进入静态合格池。
保留它们是为了后续有证据地选择“重写或删除”，而不是静默清洗。

静态合格池分布：

| Dimension | Counts |
|---|---|
| products/query | 1: 289, 2: 496, 3: 418, 4: 211 |
| voucher | platform: 426, shop: 988 |
| discount | fixed: 565, percentage: 849 |
| budget difficulty | easy: 351, medium: 641, hard: 422 |
| constraint complexity | low: 342, medium: 708, high: 364 |

query 长度（包含程序拼接的 voucher/budget 后缀）为 49/80/109/183 words（min/P50/P95/max），
253/438/610/1019 characters。下一步在获得确认后才执行 Stage D：Step108、G8 的 learnability
筛选；当前 1414 条只能称为静态合格候选，不能提前称为最终 RL train。

## 9. 计划变更与最终正式split

上一节是API生成完成时的阶段性结论。随后决定不再离线做Stage D永久筛选：1414条静态合格样本全部
作为train，learnability改由训练时DAPO动态采样处理。all-fail/all-success query不会被永久删除，policy
变化后仍可再次被抽中；只有当次G8为mixed时才进入更新。这保留了难题覆盖，也避免用Step108的一次
采样结果固化训练分布。

正式数据为：

| Split | Queries | Unique products | Product source |
|---|---:|---:|---|
| train | 1414 | 3379 | 全部静态合格RL v3候选；替换2条test冲突商品 |
| validation | 64 | 160 | 未见商品/店铺重新采样，Qwen改写、embedding maximin |
| test | 250 | 714 | 完整`data/synthesize_voucher_test.jsonl` |

train/validation/test之间query overlap和reward product overlap均为0。val64中1/2/3/4商品各16条，
platform/shop各32条，fixed/percentage各32条。test250保持正式原始分布：商品数1/2/3/4为
22/70/80/78，platform/shop为129/121。

最终文件、SHA256、分布和审计结果记录在`dataset/shoppingbench_query_rl_v3/manifest.json`与
`report.json`。本次训练后的重要反证是：product-disjoint和表面桶平衡并不足以保证validation代表test；
新生成val的措辞比正式test暴露更多标题线索，Step108 ASR分别为33.01%和8.30%。下一版数据必须从
正式query生成机制同源留出validation，而不是单独生成一个看似更“泛化”的集合。
