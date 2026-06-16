# ShoppingBench 项目总结

## 项目概述

ShoppingBench 是阿里巴巴集团提出的一个端到端购物基准测试，用于评估基于大语言模型(LLM)的代理在真实世界购物场景中的表现。该项目发表于 arXiv (arXiv:2508.04266)。

## 核心特点

### 1. 基准测试数据集
- **规模**：3,310个用户指令
- **训练集**：2,410个样本
- **测试集**：900个样本
- **产品沙盒**：包含超过250万真实世界产品（来自Lazada.com）

### 2. 四种购物意图（复杂度递增）
1. **Products Finder**：根据产品属性查找产品
2. **Knowledge**：基于知识推理找到相关产品
3. **Multi-products seller**：找到销售多个产品的同一商店
4. **Coupon & Budget**：在预算约束下使用优惠券找到最优产品组合

### 3. 工具集（6个API工具）
1. `find_product` - 检索产品列表
2. `view_product_information` - 查看产品详情
3. `python_execute` - 执行Python代码（用于计算折扣、预算等）
4. `web_search` - 网络搜索（用于知识查询）
5. `recommend_product` - 推荐产品
6. `terminate` - 终止会话

### 4. 评估指标
- **ASR (Absolute Success Rate)**：绝对成功率
- **CAR (Cumulative Average of product Relevance)**：累积平均产品相关性
- 包含多个约束分数：产品相关性分数、知识约束分数、商店约束分数、预算约束分数

## 实验结果

### 主要发现
1. **性能表现**：
   - 最佳代理（GPT-4.1）的整体成功率低于50%
   - 简单意图（Products Finder）：GPT-4.1达到59.6% ASR
   - 复杂意图（Coupon & Budget）：GPT-4.1仅30.4% ASR

2. **模型对比**：
   - 闭源模型：GPT-4.1 > Claude-4-Sonnet > GPT-4o
   - 开源模型：DeepSeek-R1表现最佳

3. **训练效果**：
   - 通过轨迹蒸馏和SFT+RL训练，Qwen3-4B模型性能提升30.7%
   - 训练后的模型达到48.7% ASR，超过GPT-4.1的48.2%

## 代码结构

```
ShoppingBench/
├── src/
│   ├── agent/                    # 代理核心代码
│   │   ├── run_rollout.py        # 推理运行脚本
│   │   ├── run_evaluate.py       # 评估脚本
│   │   ├── toolkit/              # 工具实现
│   │   │   ├── base.py           # 工具基类
│   │   │   ├── find_product.py   # 产品查找工具
│   │   │   ├── view_product_information.py
│   │   │   ├── python_execute.py
│   │   │   ├── web_search.py
│   │   │   ├── recommend_product.py
│   │   │   └── terminate.py
│   │   ├── util/                 # 工具函数
│   │   │   ├── llm.py            # LLM调用
│   │   │   └── message.py        # 消息处理
│   │   └── prompt/               # 提示词模板
│   ├── search_engine/            # 搜索引擎
│   ├── sft/                      # 监督微调
│   ├── rl/                       # 强化学习
│   └── statistic/                # 统计分析
├── config/                       # 配置文件
│   ├── rollout/                  # 推理配置
│   ├── ablation_react/           # 消融实验配置
│   └── ...
├── data/                         # 测试数据
├── resources/                    # 资源文件
├── run.sh                        # 运行脚本
├── init_env.sh                   # 环境初始化
└── ShoppingBench.pdf             # 论文
```

## 关键文件说明

### 核心文件
- `src/agent/run_rollout.py`：主推理脚本，实现ReAct循环
- `src/agent/toolkit/base.py`：工具基类定义
- `src/agent/util/llm.py`：LLM调用封装
- `src/agent/util/message.py`：消息格式处理

### 配置文件
- `config/rollout/*.json`：各模型的推理配置
- 包含模型名称、温度、最大token数等参数

### 数据文件
- `data/synthesize_*_test.jsonl`：各意图的测试数据
- `resources/documents.jsonl.gz`：产品文档数据库

## 技术栈

- **搜索引擎**：Pyserini (BM25稀疏检索)
- **Web搜索**：Serper API
- **LLM**：支持OpenAI、Claude、Gemini、Qwen、DeepSeek等
- **训练**：SFT (监督微调) + RL (强化学习，GRPO算法)
- **框架**：Python, Flask, ujson

## 使用方法

### 环境准备
```bash
# 1. 安装Java (推荐JDK21)
# 2. 安装uv
# 3. 解压产品文档
gunzip -c resources/documents.jsonl.gz > resources/documents.jsonl
# 4. 设置API密钥
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="your_base_url"
export SERPER_KEY="your_serper_key"
# 5. 初始化环境
./init_env.sh
```

### 运行推理
```bash
# 以GPT-4.1为例
./run.sh product rollout gpt-4.1
./run.sh shop rollout gpt-4.1
./run.sh voucher rollout gpt-4.1
./run.sh web simpleqa_rollout gpt-4.1
```

### 运行评估
```bash
# 修改run.sh，取消注释评估行，注释推理行
python src/agent/run_evaluate.py config/rollout/gpt-4.1.json
```

## 项目意义

1. **挑战性**：即使是GPT-4.1也仅能达到50%以下的成功率
2. **实用性**：基于真实电商场景，包含250万+真实产品
3. **可扩展性**：提供可扩展的框架生成多样化用户指令
4. **训练效果**：证明通过轨迹蒸馏可以显著提升小模型性能

## 相关链接

- 论文：https://arxiv.org/abs/2508.04266
- GitHub：https://github.com/yjwjy/ShoppingBench
