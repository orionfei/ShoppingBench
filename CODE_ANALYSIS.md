# ShoppingBench 代码分析

## 1. 核心架构

### 1.1 代理系统 (src/agent/)

#### 主要入口
- `run_rollout.py`：推理主脚本，实现ReAct循环
  - `react_loop()`：核心推理循环
  - `think()`：调用LLM进行思考
  - `act()`：执行工具调用
  - `is_terminate()`：判断是否终止

#### 工具系统 (src/agent/toolkit/)
```
base.py              # 工具基类 BaseTool
├── find_product.py  # 产品搜索工具
├── view_product_information.py  # 产品详情查看
├── python_execute.py  # Python代码执行
├── web_search.py    # 网络搜索
├── recommend_product.py  # 产品推荐
└── terminate.py     # 终止工具
```

#### 工具基类设计
```python
class BaseTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, str]
    
    def execute(self, **kwargs):
        raise NotImplementedError()
    
    def to_string(self):
        return f"Name: {self.name}\nDescription: {self.description}\nParameters: {json.dumps(self.parameters)}"
```

#### 工具调用流程
1. 用户输入查询
2. LLM生成思考过程和工具调用
3. 执行工具并获取观察结果
4. 将观察结果反馈给LLM
5. 重复直到调用terminate工具

### 1.2 搜索引擎 (src/search_engine/)

#### 服务器架构
- 使用Flask + Waitress构建REST API
- 基于Pyserini的BM25稀疏检索
- 端口：5631

#### API端点
```
GET /find_product
  参数: q, page, shop_id, price, sort, service
  返回: 产品列表（每页10个，最多5页）

GET /view_product_information
  参数: product_ids (逗号分隔)
  返回: 产品详细信息
```

#### 搜索字段
- 搜索结果字段：product_id, shop_id, title, price, service, sold_count
- 详情字段：product_id, short_description, description, sku_options, attributes

### 1.3 训练系统

#### SFT训练 (src/sft/)
- 基于LLaMA-Factory框架
- 使用轨迹蒸馏生成的训练数据
- 序列长度：20,480 tokens
- 训练5个epoch

#### RL训练 (src/rl/)
- 使用GRPO算法
- 策略模型学习率：1e-6
- 批次大小：128
- 微批次大小：32
- 温度系数：0.2
- 最大上下文长度：16,384 tokens
- 最大生成长度：1,024 tokens

## 2. 数据流

### 2.1 推理流程
```
用户查询 → 系统提示词构建 → LLM思考 → 工具调用 → 环境观察 → 下一轮思考
    ↓
轨迹记录 → 评估 → 结果输出
```

### 2.2 训练流程
```
GPT-4.1生成轨迹 → 拒绝采样过滤 → SFT训练 → RL训练 → 模型评估
```

## 3. 关键实现细节

### 3.1 消息格式
```python
class Message:
    user: str           # 用户输入
    think: str          # 思考过程
    tool_call: list     # 工具调用列表
    obs: list           # 观察结果
    response: str       # 最终响应
```

### 3.2 工具调用格式
```json
{
    "tool_call_id": "unique_id",
    "name": "tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}
```

### 3.3 轨迹记录格式
```json
{
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "completion": {
        "reasoning_content": "...",
        "content": "...",
        "message": {...}
    },
    "extra_info": {
        "step": 1,
        "query": "...",
        "timestamp": 1234567890
    }
}
```

## 4. 配置系统

### 4.1 配置文件结构
```json
{
    "task": "product",                    // 任务类型
    "system_prompt_file": "...",          // 系统提示词文件
    "synthesize_file": "...",             // 输入数据文件
    "rollout_file": "...",                // 输出轨迹文件
    "threads": 4,                         // 并发线程数
    "model_config": {
        "model": "gpt-4.1-2025-04-14",   // 模型名称
        "temperature": 0,                 // 温度参数
        "max_tokens": 8192               // 最大token数
    },
    "exclude_tools": []                   // 排除的工具
}
```

### 4.2 任务类型
- `product`：Products Finder意图
- `shop`：Multi-products seller意图
- `voucher`：Coupon & Budget意图
- `web`：Knowledge意图

## 5. 评估系统

### 5.1 评估指标计算
```python
# 产品相关性分数
r_pro = (I_sim≥0.5 + I_min≤p≤max + |F_t ∩ F_p|) / (2 + |F_t|)

# 知识约束分数
r_kw = 1 if knowledge_attribute in title else 0

# 商店约束分数
r_shop = 1 if n_t == n_p and |S| == 1 else 0

# 预算约束分数
r_budget = 1 if total_price ≤ budget else 0
```

### 5.2 绝对成功率(ASR)
```python
# Products Finder
S_pro = (1/n) * Σ δ(r_pro(i) = 1)

# Knowledge
S_kw = (1/n) * Σ δ(r_pro(i) = 1, r_kw(i) = 1)

# Multi-products seller
S_shop = (1/n) * Σ δ((1/n_i) * Σ r_shop(j) = 1, r_shop(i) = 1)

# Coupon & Budget
S_budget = (1/n) * Σ δ((1/n) * Σ r_budget(j) = 1, r_budget(i) = 1)
```

## 6. 性能优化

### 6.1 并发处理
- 使用multiprocessing实现多进程
- 生产者-消费者模式
- 文件锁保证数据一致性

### 6.2 搜索优化
- BM25稀疏检索
- 分页查询（每页10个，最多5页）
- 多条件过滤（价格、服务、商店）

### 6.3 训练优化
- 序列打包减少padding
- 拒绝采样过滤低质量轨迹
- SFT+RL两阶段训练

## 7. 扩展点

### 7.1 添加新工具
1. 继承BaseTool类
2. 实现execute方法
3. 在toolkit/__init__.py中注册

### 7.2 添加新意图
1. 创建新的测试数据文件
2. 设计新的评估指标
3. 更新配置文件

### 7.3 支持新模型
1. 在config/目录下添加配置文件
2. 更新run.sh脚本
3. 测试模型兼容性

## 8. 依赖库

### 核心依赖
- pyserini==1.0.0：搜索引擎
- Flask：Web服务器
- ujson：JSON处理
- waitress：WSGI服务器
- sentence-transformers：句子编码
- portalocker：文件锁
- duckduckgo_search：DuckDuckGo搜索
- googlesearch-python：Google搜索
- mcp：模型控制协议

### 训练依赖
- transformers：Hugging Face模型
- torch：PyTorch
- deepspeed：分布式训练
- vllm：推理加速
- sglang：服务框架
