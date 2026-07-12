# 一次 G 敏感的异步 Rollout 故障：从模型退化到 vLLM 并发治理

## 摘要

在 ShoppingBench 的 GRPO/验证 rollout 中，我们遇到了一个反直觉问题：使用同一个
`global_step_108` checkpoint、相同提示词和相同采样参数时，两次独立的 `G=4` 运行明显
好于一次 `G=8` 运行。后者不仅任务成功率下降，还出现了工具调用 JSON 损坏、重复片段和
超长 runaway 输出。

我们最终确认，问题并不是“模型在 G=8 时能力下降”，也不是 GRPO 的 group size 会直接
改变单条生成。真正改变的是执行拓扑：`G` 通过外层 prompt repeat 增加同一批次提交的
trajectory 数量，进而改变 async agent worker、vLLM server 和 scheduler 的瞬时并发形状。

进一步检查发现，项目虽然配置了 `rollout.max_num_seqs`，但 async vLLM server 创建
`AsyncEngineArgs` 时没有传入该参数，因此预期的调度并发限制实际上没有生效。在两张 GPU、
两个 rollout engine 上一次提交 64 条长、多轮工具调用轨迹时，高并发与 prefix caching、
CUDA Graph/批形状变化共同放大了生成不稳定性。

我们没有继续追到 vLLM 内部具体哪个 kernel、cache block 或数值路径导致 logits 偏移，
因此不能声称修复了 vLLM 的底层根因。但我们通过控制变量实验找到了稳定、可验证且可回退
的工程方案：修复 `max_num_seqs` 的参数传递，并将每个 vLLM engine 同时调度的序列数限制
为 8。该方案使 `G=8` 恢复正常，同时没有牺牲原本表现良好的 `G=4`。

## 1. 问题背景

实验使用：

- Checkpoint：`global_step_108`
- GPU：2 张 A800
- Rollout：verl async multi-turn agent loop + vLLM 0.8.5.post1
- Temperature：0.2
- Top-p：0.9
- 最大 response token：10240
- 最大 assistant turn：15
- Tensor parallel size：1
- Agent loop worker：8

这里的 `G` 表示每个 query 采样的 trajectory 数。在当前实现中，验证代码先按
`val_kwargs.n` 对 prompt 做外层重复，再把重复后的 batch 分发给 agent workers。因此：

```text
8 queries × G=4 = 32 trajectories
8 queries × G=8 = 64 trajectories
```

它并不是让一条 vLLM 请求在内部返回 4 或 8 个候选；每条请求仍然只取一个输出。

## 2. 最初观察到的异常

历史结果中：

- 一次 `G=8`，64 条输出，成功数为 0，format 约为 0.173；
- 两次独立 `G=4`，合计同样 64 条输出，成功率为 12.5%，format 为 1.0。

如果 `G` 只改变样本数量，这两种运行的总体分布应该接近。实际差异远超合理采样方差，
而且表现为格式和协议整体崩溃，不只是任务答案变差。这提示我们优先调查生成基础设施，而
不是直接归因于 checkpoint 能力。

## 3. 建立并逐步排除假设

### 3.1 假设：采样参数没有对齐

首先确认 checkpoint、temperature、top-p、response length、最大 turn 和 prompt 数据完全
一致。该假设被排除。

### 3.2 假设：G 被错误地传入 vLLM 的内部 `n`

如果每条请求实际上生成 8 个候选，而上层只消费第一个，可能造成额外的 RNG 消耗和调度
差异。代码检查和运行时日志表明，请求级 `SamplingParams.n` 仍为 1。实验 runner 还显式将
engine generation config 的 `n` 固定为 1，异常依然存在，因此这不是主因。

### 3.3 假设：异步调度和随机数顺序造成不可复现

原实现使用随机 UUID request id、worker 本地的 server 顺序和 least-request routing，且
没有请求级 seed。我们增加了可选的 stable sampling 路径：确定性 request id、请求级 seed、
稳定 server 顺序和稳定路由。

这些改动提高了实验可比性，但在稳定采样开启后，大批量 `G=8` 仍然崩溃。这说明异步 RNG
顺序是干扰因素，却不足以解释格式从 1.0 降到约 0.5 的系统性变化。

### 3.4 关键假设：真正的变量是瞬时 trajectory 并发量

我们设计了等并发对照：

```text
G=8, VAL_BATCH_SIZE=1  -> 每个 wave 8 条 trajectory
G=4, VAL_BATCH_SIZE=2  -> 每个 wave 8 条 trajectory
```

`G=8` 在每个 wave 只有 8 条 trajectory 时恢复正常，format 回到 1.0，任务成功率回到
20.3%。这说明 group size 本身不是触发条件，单个 wave 的总请求压力才是关键变量。

这也解释了为什么“两次 G=4”明显好于“一次 G=8”：前者把 64 个样本拆成了多个较小的
调度 wave，后者把它们集中提交。旧对照实验无意中同时改变了统计采样数和执行拓扑。

## 4. 能追溯到的最深原因

沿着 rollout 配置到 engine 初始化路径检查后，我们发现：

```text
launch script
  -> actor_rollout_ref.rollout.max_num_seqs
  -> Hydra rollout config
  -X-> AsyncEngineArgs(max_num_seqs=...)
```

`max_num_seqs` 出现在 launch script 和 Hydra config 中，但 async vLLM server 构造
`AsyncEngineArgs` 时没有传入它。因此操作者以为 engine 的并发上限是 8 或 16，实际上该值
只停留在配置层，没有控制 vLLM scheduler。

GPU 数量会进一步改变每个 engine 的压力。在 tensor parallel size 为 1 时：

```text
4 GPUs -> 4 rollout engines -> 64 条初始请求约为每个 engine 16 条
2 GPUs -> 2 rollout engines -> 64 条初始请求约为每个 engine 32 条
```

所以从四张卡切换到两张卡，即使 checkpoint 和采样参数不变，也会让每个 engine 面对约两倍
的初始请求压力。

到这里，我们可以确定的因果链是：

```text
G / validation batch 增大
  -> 单个 wave 的 trajectory 数增大
  -> 每个 vLLM engine 的调度压力增大
  -> 未生效的 max_num_seqs 无法提供边界
  -> prefix cache / CUDA Graph / 动态批形状放大生成不稳定性
  -> malformed tool call、runaway output、协议和任务指标崩溃
```

最后一段“vLLM 内部为什么在这个组合下改变生成结果”没有继续追到源码和 CUDA kernel 层。
因此严谨的表述是：我们定位并修复了应用集成层的直接缺陷，也找到了触发 vLLM 不稳定性的
必要运行条件；但没有宣称修复 vLLM 内部最底层的实现问题。

## 5. 控制变量实验结果

以下实验均使用相同 checkpoint、seed 集合、prompt 和采样参数：

| 配置 | 单 wave trajectory | Format | Protocol | Task | Success |
|---|---:|---:|---:|---:|---:|
| `G=8, VB=1`，无显式 scheduler cap | 8 | 1.000 | 0.845 | 0.182 | 20.3% |
| `G=8, VB=8`，无显式 scheduler cap | 64 | 0.518 | 0.507 | 0.060 | 6.3% |
| `G=8, VB=8`，`enforce_eager=True` | 64 | 0.722 | 0.669 | 0.085 | 7.8% |
| `G=8, VB=8`，关闭 prefix cache | 64 | 0.984 | 0.825 | 0.074 | 14.1% |
| `G=8, VB=8`，`max_num_seqs=4` | 64 | 1.000 | 0.845 | 0.178 | 15.6% |
| `G=8, VB=8`，`max_num_seqs=8` | 64 | 1.000 | 0.863 | 0.207 | 20.3% |
| `G=4, VB=8`，`max_num_seqs=8` | 32 | 1.000 | 0.851 | 0.153 | 12.5% |

这里的测试集不是完整 benchmark，而是固定的 8-query diagnostic probe：

- Parquet：`dataset/probe/sft_clean924_test8_state_local/probe.parquet`
- 原始来源：`data/synthesize_voucher_test.jsonl`
- 原始行号：22、44、56、62、66、77、122、241
- 4 个 shop voucher、4 个 platform voucher
- 4 个 percentage discount、4 个 fixed discount
- 目标商品数分布：1 件 1 题、2 件 2 题、3 件 2 题、4 件 3 题
- 预算余量难度：6 个 tight、2 个 medium
- 阈值难度：2 个 near、3 个 mid、3 个 far

因此固定配置下，G=4 的 32 条轨迹中有 4 条最终成功；G=8 的 64 条轨迹中有 13 条最终
成功。G=8 的前四个确定性 seed 与 G=4 对齐，额外增加每题的第 5～8 条采样。由于只有 8 个
query，这组结果适合验证 rollout 稳定性，不应单独当作模型完整泛化能力的置信估计。

这些结果支持三个判断：

1. CUDA Graph 是放大因素，但关闭它不能解决问题；
2. prefix caching 与高并发的组合是重要放大因素，但关闭 cache 后 task 仍未完全恢复；
3. 让 scheduler cap 真正生效能够同时恢复格式、协议和任务效果，是当前最稳健的方案。

## 6. 实施的工程改动

### 6.1 修复 `max_num_seqs` 参数传递

async server 现在仅在显式开启时把该值传给 vLLM：

```python
engine_kwargs = {}
if bool(config.get("apply_max_num_seqs", False)):
    engine_kwargs["max_num_seqs"] = int(config.max_num_seqs)

engine_args = AsyncEngineArgs(
    ...,
    **engine_kwargs,
)
```

推荐配置：

```bash
ROLLOUT_APPLY_MAX_NUM_SEQS=True
ROLLOUT_MAX_NUM_SEQS=8
```

这不会减少 GRPO 的 `G`，也不会改变 optimizer batch。超过上限的 rollout 请求会在 engine
外部/内部等待，scheduler 每次只处理受控数量的活跃序列。

### 6.2 将 prefix caching 暴露为诊断开关

过去 async server 固定写死 `enable_prefix_caching=True`。现在可通过配置控制：

```bash
ROLLOUT_ENABLE_PREFIX_CACHING=True   # 正常推荐
ROLLOUT_ENABLE_PREFIX_CACHING=False  # 仅用于诊断或紧急隔离
```

最终方案保留 prefix cache，因为 `max_num_seqs=8` 已能稳定生成，同时保留其性能收益。

### 6.3 增加稳定采样控制

为了让 G 对照实验不受异步 request id、路由和 RNG 消耗顺序干扰，我们加入了可选的稳定采样
控制，包括请求级 seed、确定性 request id、稳定 server routing，以及固定 engine-level
`n=1`。这不是解决高并发故障的核心修复，但它是建立可信因果实验的重要基础设施。

## 7. 固定默认值、Fallback 和风险控制

完成因果实验和 G=4 保护性回归后，项目的生产默认值已固定为：

```yaml
max_num_seqs: 8
apply_max_num_seqs: True
enable_prefix_caching: True
stable_sampling.enabled: True
stable_sampling.force_generation_config_n1: True
```

主训练入口、A800 启动封装、checkpoint probe 和专用实验 runner 使用相同默认值，避免某个
上层脚本把 `max_num_seqs` 静默覆盖回 16 或 32。若升级 vLLM 后需要对照旧路径，仍可立即
显式回退：

```bash
ROLLOUT_APPLY_MAX_NUM_SEQS=False
```

我们还专门运行了 `G=4, max_num_seqs=8` 的保护性回归实验。其 format 为 1.0、protocol 为
0.851，与原本良好的 G=4 水平一致，说明修复没有为了对齐 G=8 而牺牲 G=4。

## 8. 最终方案与边界

当前两张 A800 上推荐：

```bash
ROLLOUT_APPLY_MAX_NUM_SEQS=True
ROLLOUT_MAX_NUM_SEQS=8
ROLLOUT_ENABLE_PREFIX_CACHING=True
ROLLOUT_ENFORCE_EAGER=False
ROLLOUT_FREE_CACHE_ENGINE=False
STABLE_ROLLOUT_SAMPLING=True
STABLE_ROLLOUT_FORCE_GENERATION_CONFIG_N1=True
```

这个方案的定位不是“证明 vLLM 从此与并发完全无关”，而是：

- 找到了可重复触发故障的执行条件；
- 找到了项目代码中确凿的参数传递缺陷；
- 用相同输入和采样参数完成了因果 A/B 实验；
- 选择了同时恢复 G=8、保护 G=4、保留 prefix cache 且可随时回退的配置；
- 明确记录了尚未深入的 vLLM 内部边界。

如果后续升级 vLLM、改变模型、上下文长度、GPU 数量或显存配置，应重新做 concurrency sweep，
而不是假设 8 永远是全局最优值。

## 9. 面试中值得讨论的工程判断

这个案例的价值不只是修复一个参数，而是展示了几个可迁移的方法：

1. **先区分统计变量和执行变量。** `G` 表面上是采样组大小，实际上同时改变了 batch repeat、
   worker chunk、请求到达顺序和 engine 并发。
2. **格式整体崩溃通常是基础设施信号。** 如果只是模型能力不足，通常不会伴随大面积 JSON
   损坏和 runaway generation。
3. **用等并发实验拆开混杂变量。** `G=8, VB=1` 对 `G=4, VB=2` 是本次定位最关键的实验。
4. **配置存在不等于配置生效。** 必须沿着 launch script、Hydra config 一直追到第三方库构造
   参数，并用运行时实验验证。
5. **不要把 workaround 描述成上游根因修复。** 我们能证明 scheduler cap 解决当前系统问题，
   但不能在未读 vLLM/CUDA 源码的情况下断言具体底层 bug。
6. **高风险修复必须有 fallback 和保护性回归。** 默认保留旧路径，并单独验证 G=4 不退化。
7. **吞吐和正确性应分别治理。** 保留 G 和 optimizer batch，通过 engine queue/cap 控制瞬时
   并发，比直接缩小训练语义更合理。

## 10. 相关实现与实验材料

- Async vLLM 参数传递：`src/rl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- Rollout 配置：`src/rl/verl/trainer/config/rollout/rollout.yaml`
- 启动参数：`src/rl/run_grpo_qwen3_1_7b_query_verl.sh`
- 两张 A800 启动封装：`src/rl/run_grpo_qwen3_4b_state_folded_a800.sh`
- 可复现实验 runner：`scripts/run_g_root_cause_probe_20260710.sh`
- 原始技术调查记录：`docs/g_rollout_sensitivity_dilemma_20260709.md`
- 实验报告：`reports/g_root_step108_2gpu_*_20260710/`
