# AR 模块架构文档

## 1. 系统概述

AR（Any-order autoRegressive）是一个基于 Transformer 的表格数据近似函数依赖（AFD, Approximate Functional Dependency）发现系统。

**问题定义**：给定一张关系表 $R$ 和阈值 $\varepsilon$，找到所有列组合 $\text{LHS} \to \text{RHS}$，使得条件概率 $P(\text{RHS} \mid \text{LHS}) \geq 1 - \varepsilon$。

**核心创新**：将表格的每一行视为一个短序列，每个位置对应一列。通过随机 mask 任意列子集并训练模型恢复被 mask 的列，模型学会了在**任意观测子集**下做条件概率估计，从而天然支持 AFD 搜索中任意 LHS 组合的查询需求。

---

## 2. 四阶段流水线

```
原始 CSV / NPY
│
▼
[1. preprocess]  → schema.json + train/val/test_tokens.npy
│                列类型推断、词表构建、整数编码、NaN→[NULL]映射
▼
[2. train]       → best.pt + training_history.json
│                双任务联合训练（cond_loss + afd_loss）
▼
[3. calibration] → temperature_scaling.json
│                每列独立温度缩放，校准概率质量
▼
[4. search]      → search_results.jsonl + search_summary.json + discovered_fd/*.txt
                Level-1 枚举 → Beam 扩展 → 双信号打分 → 方向性检验 → 极小性过滤
```

每个阶段产物落盘，下游直接复用，可独立运行。

---

## 3. 预处理阶段（preprocess.py）

### 3.1 列类型分类

预处理阶段将原始数据中的列分为三种建模类型：

| 类型 | 适用场景 | 编码方式 |
|---|---|---|
| `categorical` | 类别型列、高基数文本列 | NFKC 规范化 → 小写 → 词表映射 |
| `discrete_numeric` | 整数型离散列 | `int(float(value))` → 词表映射 |
| `continuous_bucket` | 连续数值列 | 等频分桶（`continuous_bins` 个桶）→ 桶 id |

列类型由两个因素共同决定：
1. **上游预处理器的声明**：`_meta.json` 中的 `categorical_cols`、`continuous_cols`、`discrete_numeric_cols`
2. **列策略系统（column_policy.py）的 `analysis_mode`**：`exact` → categorical/discrete_numeric，`binned` → continuous_bucket

### 3.2 词表构建

每列构建独立词表，前 4 个位置固定为特殊 token：

```
index 0: [NULL]   — 缺失值
index 1: [UNK]    — 未知值（推理时未见过的值）
index 2: [MASK]   — 训练时的占位符
index 3: [RARE]   — 低频值（出现次数 < rare_token_min_freq）
index 4+: 实际 token，按训练集频率降序排列
```

**关键设计**：词表是列级别的，不同列的 token id 相互独立。例如第 0 列的 token id=5 和第 1 列的 token id=5 代表完全不同的语义。

### 3.3 NaN → [NULL] 映射链路

上游预处理器（`utils/preprocess_*.py`）将 NaN 编码为 `-1`（分类列）或保留 `np.nan`（数值列）。AR 预处理阶段通过两条路径将 NaN 映射到 `[NULL]`：

**分类列路径**：
1. 上游：`value_to_idx["Unknown"] = -1`（哨兵），NaN 位置在 `.npy` 中为 `float32(-1.0)`
2. AR `_categorical_token_from_raw`：检测到 `encoded < 0` → 返回 `"[NULL]"`
3. AR `encode_token("[NULL]")` → `null_id = 0`

**数值列路径**：
1. 上游：保留 `np.nan` 到 `.npy` 文件
2. AR `canonicalize_discrete_numeric` / `parse_continuous`：`_is_nullish()` 检测 `float('NaN')` → 返回 `"[NULL]"` / `None`
3. AR `encode_token("[NULL]")` → `null_id = 0`

### 3.4 searchable 标记

每列有两个布尔标记 `searchable_lhs` 和 `searchable_rhs`，决定该列是否参与 FD 搜索：

- **目标列**（target_col）：两侧均禁用
- **常量列**（unique_count ≤ 1）：两侧均禁用
- **标识符列**（role = identifier / quasi_identifier）：仅 LHS 启用（因为近唯一列作为 RHS 会得到平凡的高分 FD）
- **近标识符列**（role = near_identifier）：strict/balanced 模式下禁用，permissive 模式下两侧启用
- **普通列**：根据 `search_space_mode` 和列类型决定

`search_space_mode` 有三种模式：
- `strict`：仅接受策略明确允许的 `exact`/`binned` 列
- `balanced`（默认）：放宽到所有 tokenizable LHS + categorical/discrete_numeric RHS
- `permissive`：最大化召回，包括 bucketed continuous RHS

### 3.5 数据划分

使用固定种子的随机排列做 80/10/10 划分：

```python
rng = np.random.default_rng(seed)
order = rng.permutation(total)
train_ids = order[:int(total * 0.8)]
val_ids = order[int(total * 0.8):int(total * 0.9)]
test_ids = order[int(total * 0.9):]
```

---

## 4. 模型架构（model.py）

### 4.1 AnyOrderConditionalTransformer

```
输入: token_ids [B, C], observed_mask [B, C]
│
├─ value_embeddings[col_id](token_ids[:, col_id])  → [B, C, d_model]  (每列独立嵌入表)
├─ column_embedding(col_ids)                        → [1, C, d_model]  (列身份，广播)
├─ type_embedding(type_ids)                         → [1, C, d_model]  (列类型，广播)
├─ observed_embedding(observed_mask.long())         → [B, C, d_model]  (观测状态)
│
▼ 四路逐元素相加
hidden = value + column + type + observed           → [B, C, d_model]
│
▼ TransformerEncoder (n_layers × Pre-LN, GELU, batch_first)
│
▼ LayerNorm
│
▼ output_heads[col_id](hidden[:, col_id, :])        → [B, vocab_size_col]
```

### 4.2 四路嵌入的设计意图

| 嵌入 | 作用 | 维度 |
|---|---|---|
| `value_embedding` | 编码当前列的具体取值 | 每列独立的 `nn.Embedding(vocab_size, d_model)` |
| `column_embedding` | 告诉模型"这个位置是第几列" | `nn.Embedding(num_columns, d_model)` |
| `type_embedding` | 区分列的建模类型（categorical/discrete_numeric/continuous_bucket） | `nn.Embedding(3, d_model)` |
| `observed_embedding` | 标记该列是观测证据还是被 mask 的 | `nn.Embedding(2, d_model)` |

**关键设计**：`value_embedding` 是每列独立的，因为不同列的 token id 空间相互独立（第 0 列的 id=5 和第 1 列的 id=5 语义不同）。而 `column_embedding`、`type_embedding`、`observed_embedding` 是全局共享的。

### 4.3 架构参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `d_model` | 192 | 隐层维度 |
| `n_heads` | 4 | 注意力头数 |
| `n_layers` | 4 | Transformer 编码器层数 |
| `ffn_dim` | 768 | FFN 中间维度（4×d_model） |
| `dropout` | 0.1 | Dropout 率 |

使用 Pre-LN（LayerNorm 在注意力/FFN 之前）而非 Post-LN，训练更稳定。

---

## 5. 训练阶段（train.py）

### 5.1 双任务联合训练

每个训练 step 同时计算两个损失：

```python
loss = cond_loss + afd_loss_weight × afd_loss
```

**cond_loss（通用条件建模）**：
1. 对每行随机采样一个观测比例（从 `mask_ratios = (0.2, 0.5, 0.8)` 中均匀选取）
2. 随机选择对应数量的列设为 observed，其余设为 unobserved
3. 将 unobserved 列替换为该列的 `[MASK]` token
4. 模型根据 observed 列编码整行，预测所有 unobserved 列
5. 损失 = 所有 unobserved 列的交叉熵之和 / unobserved 列数

**afd_loss（定向依赖发现）**：
1. 从 searchable RHS 列中随机抽取一个目标列 `rhs_col`
2. 从 searchable LHS 列（排除 rhs_col）中随机抽取 1~max_lhs_size 个列作为 LHS
3. 仅将 LHS 列设为 observed，RHS 列设为预测目标
4. 损失 = RHS 列的交叉熵

**目的**：cond_loss 让模型学会通用的"给定部分列恢复其余列"能力；afd_loss 显式强化"只给 LHS 预测 RHS"的依赖发现场景。

### 5.2 类别权重（缓解长尾）

```python
weight(token) = (1 / (count(token) + 1))^gamma
```

每列独立计算，列内归一化。`[MASK]` token 权重设为 0（不鼓励预测占位符）。`gamma = 0.5` 控制权重衰减速度。

### 5.3 训练优化

| 优化项 | 实现 |
|---|---|
| 优化器 | AdamW（lr=1e-3, weight_decay=1e-4） |
| 学习率调度 | OneCycleLR（10% warmup + 余弦衰减） |
| 混合精度 | `torch.amp.GradScaler`（CUDA 下自动启用） |
| 梯度裁剪 | `clip_grad_norm_(max_norm=1.0)` |
| 早停 | 连续 `early_stop_patience` 轮 val_loss 无提升则停止（最少 `min_epochs` 轮） |

### 5.4 Checkpoint 策略

- `last.pt`：每 epoch 覆盖，保存最新状态（含 model_state、model_config、schema_path、epoch）
- `best.pt`：仅在 val_loss 更优时更新，后续推理和校准使用此文件

---

## 6. 温度缩放校准（calibration.py）

### 6.1 问题

模型训练后输出的 softmax 概率可能过于自信或过于保守。对于 AFD 打分，概率质量的准确性直接影响 `s_ent`（熵降低）和 `s_acc`（准确率提升）的可靠性。

### 6.2 方法

为每个 searchable RHS 列独立学习一个温度参数 $T$：

```
P(y | x) = softmax(logits / T)
```

优化目标是最小化验证集上的 NLL（负对数似然）。使用 Adam 优化器，100 步，学习率 0.05。

### 6.3 校准质量评估

| 指标 | 公式 | 含义 |
|---|---|---|
| NLL | $-\sum \log P(y_i \mid x_i)$ | 负对数似然 |
| Brier Score | $\frac{1}{N}\sum \sum (P_{ij} - \mathbb{1}[y_i=j])^2$ | 概率与真实标签的均方误差 |
| ECE | $\sum_b \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$ | 15 个等宽 bin 的校准误差 |

### 6.4 产物

`temperature_scaling.json`：格式为 `{"temperatures": {"col_name": T, ...}, "metrics": {...}}`。

---

## 7. 支撑集估计（support.py）

### 7.1 职责

从训练集中统计 LHS 列组合的经验分布，为搜索阶段提供"哪些 LHS 取值组合在训练集中出现过"以及"每个组合的经验 RHS 分布"。

### 7.2 构建流程（`build_support`）

```
train_tokens[:, lhs_cols]
│
▼ np.unique（按 LHS 值组合分组）
│
├─ 单列 LHS：直接 unique
├─ 双列 LHS：mixed-radix 编码（left × vocab_size_right + right）
├─ 多列 LHS：void-view trick（将多列视为定长字节做 unique）
│
▼ 过滤特殊 token（NULL/UNK/MASK/RARE）
│
▼ 过滤低频模式（min_support_count / min_effective_support_count / min_pure_support_count）
│
▼ 计算每个 LHS 模式的经验 RHS 分布（entropy、top1）
│
▼ 退化模式检测：单值占比 > 95% 的模式置零（如 "Unknown" 填充的 99.7% null 列）
│
▼ 按优先级排序，截取 max_support_rows 行
│
▼ 贝叶斯平滑概率估计
```

### 7.3 概率估计

```python
empirical = counts / n_rows
blend = counts / (counts + support_beta)
independence_product = ∏ marginals[col][lhs_values[:, col]]
probabilities = empirical + independence_weight × (1 - blend) × independence_product
probabilities = probabilities / probabilities.sum()  # 归一化
```

- `support_beta` 控制向独立性假设的收缩程度。小数据集（counts 小）时 blend 接近 0，概率更多依赖独立性乘积，防止过拟合
- `independence_weight`（默认 0.0）控制独立性先验的强度
- `empirical_weights`：`counts / (counts + support_beta)`，用于后续经验混合打分

### 7.4 支撑集优先级排序

```python
priority = (effective_counts + 1)^alpha × non_null_ratio^beta
```

- `alpha = 0.75`：有效计数的权重
- `beta = 0.50`：非空比例的权重

排序后取前 `support_head_rows` 行（高优先级），剩余预算按优先级分位采样。

### 7.5 经验 RHS 统计

对每个 LHS 模式，统计训练集中对应行的 RHS 分布：

- `empirical_entropies`：$H(P_{\text{emp}}(\text{RHS} \mid \text{LHS}))$
- `empirical_top1`：$\max_v P_{\text{emp}}(\text{RHS}=v \mid \text{LHS})$
- `effective_counts`：非特殊 token 的 RHS 行数
- `non_null_ratios`：非空 RHS 比例

这些统计在打分阶段与模型预测混合使用。

---

## 8. 查询引擎（query_engine.py）

### 8.1 推理流程

```python
engine.conditional_summary_batch(rhs_col, evidences)
# → List[QuerySummary(top1_prob, entropy)]
```

1. 将 evidence（列名→值 的字典）编码为 token id
2. 构造 `token_matrix [B, C]`：evidence 列填入 token id，非 evidence 列填入 `[MASK]` id
3. 构造 `observed_mask [B, C]`：evidence 列为 True，其余为 False
4. 调用 `model.encode(masked_inputs, observed_mask)` 得到 hidden states
5. 调用 `model.column_logits(hidden, rhs_col)` 得到 logits
6. 对 logits 除以温度 T 后 softmax 得到概率分布
7. 计算 `entropy` 和 `top1_prob`

### 8.2 批量推理与缓存

- 支持批量查询：`evidences` 列表自动按 `query_batch_size=256` 切分
- LRU 缓存：`SummaryCache`（最多 50000 条），key 为 `(rhs_col, frozenset(evidence))`
- 缓存命中时直接返回，避免重复前向计算

### 8.3 有效 RHS 打分

`conditional_summary_batch_valid_rhs`：将特殊 token（NULL/UNK/MASK/RARE）的概率置零后重新归一化，仅在实际取值上计算 entropy 和 top1。这避免了 `[NULL]` 占比过高时熵被人为压低的问题。

---

## 9. AFD 搜索（search.py）

### 9.1 搜索策略

```
Level-1 枚举：对每个 searchable RHS 列，枚举所有单列 LHS
│
▼ 打分排序，取 top-k beam
│
▼ CAFD 风格自适应剪枝：根据数据集 identifier 强度动态调整阈值
│  pruning_threshold = max(min_s_acc × (0.25 + 0.25 × max_identifier_signal), 0.03)
│
▼ 递归 Beam 扩展：从 top-k 单列 LHS 出发，逐列扩展到 pair、triple ...
│  每次扩展要求：new_score - best_subset_score > delta_gain
│
▼ 双向 FD 检测：对发现的 Level-1 FD，检查反向方向是否也成立
│
▼ 极小性过滤 + 阈值过滤 → 最终结果
```

### 9.2 打分流程（`_score_candidate_core`）

对每个 `(LHS, RHS)` 候选：

#### 步骤 1：构建支撑集

从 `SupportEstimator` 获取 LHS 的经验分布，包含每个 LHS 取值的计数、概率、经验 RHS 分布。

#### 步骤 2：模型查询

对支撑集中每个 LHS 取值，查询模型的条件分布：

```python
for evidence in support.iter_evidences():
    probs = model.encode(masked_inputs, observed_mask)[rhs_col]
    entropy = H(probs)
    top1 = max(probs)
```

#### 步骤 3：计算强度指标

**熵降低强度（s_ent）**：

```python
s_ent(row) = 1 - H(P(rhs|lhs)) / H(P(rhs))
```

其中 $H(P(\text{RHS}))$ 是训练集上的边际熵。`s_ent = 1` 表示 RHS 熵降为 0（完全确定），`s_ent = 0` 表示条件熵等于边际熵（无信息增益）。

**准确率提升强度（s_acc）**：

```python
s_acc(row) = (top1(P(rhs|lhs)) - top1(P(rhs))) / (1 - top1(P(rhs)))
```

`s_acc = 1` 表示 top1 概率从边际水平提升到 1.0，`s_acc = 0` 表示无提升。

**加权期望**：

```python
model_s_ent = Σ P(lhs) × s_ent(lhs)
model_s_acc = Σ P(lhs) × s_acc(lhs)
model_score = α × model_s_ent + (1-α) × model_s_acc
```

其中 `α = score_alpha = 0.60`。

#### 步骤 4：经验混合

经验分数直接从训练集统计中计算（不经过模型）：

```python
empirical_s_ent = Σ P(lhs) × s_ent_empirical(lhs)
empirical_s_acc = Σ P(lhs) × s_acc_empirical(lhs)
```

自适应混合权重：

```python
# 基础混合权重
empirical_blend = base + high_card_base × rhs_card_support + bonus × rhs_card_support × lhs_identifier_signal

# identifier 直接加成：LHS 是近唯一列时，经验信号天然可靠
if lhs_identifier_signal >= 0.65:
    empirical_blend += 0.65 × lhs_identifier_signal

# 自适应增强：当经验准确率远超模型准确率时，模型可能存在 [NULL] 偏差
if empirical_s_acc - model_s_acc > 0.15 and empirical_s_acc >= 0.90:
    empirical_blend += min(acc_gap, 0.75)
    effective_model_weight = max(model_score_weight × 0.3, 0.10)

# 最终混合
s_ent = model_s_ent + empirical_bonus_weight × max(empirical_s_ent - model_s_ent, 0)
s_acc = model_s_acc + empirical_bonus_weight × max(empirical_s_acc - model_s_acc, 0)
```

**设计意图**：
- 模型分数泛化能力强但对高基数列（如 ID 列）可能低估依赖强度
- 经验分数精确但受数据稀疏影响
- `empirical_blend` 根据 RHS 基数、LHS identifier 信号、模型/经验差距自适应混合

#### 步骤 5：覆盖因子

```python
row_factor = log1p(support.num_rows) / log1p(coverage_row_target)
effective_row_factor = log1p(effective_rows) / log1p(coverage_effective_row_target)
mass_factor = retained_mass / coverage_mass_target
signal = mean([row_factor, effective_row_factor, mass_factor, weighted_non_null_ratio])
coverage_factor = (1 - penalty_weight) + penalty_weight × signal
```

惩罚支撑集过小的候选（行数不足、有效行数不足、覆盖质量不足）。

#### 步骤 6：组因子

当 LHS 和 RHS 来自不同列组（如 dblp10k 的 p1-side 和 p2-side）时，施加惩罚：

```python
group_factor = 1.0  # 同组或无组
group_factor = cross_group_penalty  # 不同组（soft 模式）
group_factor = 0.0  # 不同组（hard 模式）
```

列组通过列名中的数字前缀/后缀自动检测（如 `p1author` 和 `p2author` 的 group 分别为 `"1"` 和 `"2"`）。

#### 步骤 7：最终分数

```python
base_score = α × s_ent + (1-α) × s_acc
score = clip(base_score × coverage_factor × group_factor, 0, 1)
```

### 9.3 方向性检验（`score_candidate`）

防止将 `B→A` 误报为 `A→B`。

**方向边距计算**：

```python
reverse_score = max(model_s_acc(B→A) for A in lhs_cols)  # 反向依赖的模型准确率
direction_margin = model_s_acc(A→B) - reverse_score
```

**方向边距模式**（`direction_margin_mode`）：
- `single_only`（默认）：仅对单列 LHS 检查方向边距，多列 LHS 跳过（因为真正的反向 `C→A,B` 计算代价太高）
- `all`：所有候选都检查
- `off`：不做方向检查

**自适应放松**：
- 双向 FD（reverse_score ≥ 0.99）：方向边距要求降至 -1.0（不做限制）
- 高反向分数（reverse_score ≥ 0.85）：方向边距要求降低 70%
- 高经验准确率但低模型准确率（empirical_s_acc ≥ 0.90, model_s_acc < 0.60）：模型在高 null 列上存在偏差，方向边距不可靠，放松要求

### 9.4 极小性过滤（`_minimality_filter`）

按 RHS 分组，每组内按子集大小升序排列：

```python
for candidate in rhs_candidates:
    for prev in accepted_for_rhs:
        if prev.lhs_cols ⊂ candidate.lhs_cols:
            gain = candidate.score - prev.score
            if gain <= delta_gain:
                candidate.redundant = True
            if prev.score >= 0.95 and gain < 0.05:
                candidate.redundant = True  # 已有高分子集，严格要求增益
```

**高基数列的复合 LHS 惩罚**：当 LHS 包含唯一比例 > 0.01 的列时，`delta_gain` 从默认 0.05 提升到 0.08，避免高基数列的冗余组合。

### 9.5 双向 FD 检测（`_detect_bidirectional`）

对发现的 Level-1 FD（单列 LHS），显式检查反向方向：

```python
if A→B discovered:
    reverse_candidate = score_candidate(B→A)
    if reverse_candidate.model_s_acc >= 0.85
       and A→B.model_s_acc >= 0.85
       and reverse_candidate.score >= min_score × 0.8:
        report B→A as bidirectional FD
```

### 9.6 候选打分数据结构（`CandidateScore`）

| 字段 | 含义 |
|---|---|
| `s_ent` / `s_acc` | 混合后的熵/准确率强度 |
| `score` | 最终分数（含 coverage、group 惩罚） |
| `model_score` | 纯模型分数（不含经验混合） |
| `empirical_score` | 纯经验分数 |
| `empirical_blend` | 经验混合权重 |
| `empirical_bonus_weight` | 经验加成权重 = blend × (1 - effective_model_weight) |
| `reverse_score` | 反向依赖强度（用于方向性检验） |
| `direction_margin` | 正向 model_s_acc - 反向 model_s_acc |
| `coverage_factor` | 支撑集覆盖质量因子 |
| `group_factor` | 跨组惩罚因子 |
| `support_rows` | 支撑集行数 |
| `effective_support_rows` | 非特殊 token 的有效行数 |
| `retained_mass` | 支撑集覆盖的训练集质量比例 |
| `weighted_non_null_ratio` | 加权非空 RHS 比例 |
| `expected_entropy` / `expected_top1` | 模型期望熵/top1 |
| `empirical_expected_entropy` / `empirical_expected_top1` | 经验期望熵/top1 |
| `marginal_entropy` / `marginal_top1` | 边际分布的熵/top1 |
| `bidirectional` | 是否为双向 FD |

---

## 10. 自动配置（auto_profile.py）

### 10.1 数据画像

对数据集抽样（最多 20000 行），分析以下维度：

| 维度 | 计算方式 | 含义 |
|---|---|---|
| `identifier_lhs_strength` | LHS 列中 identifier 信号的均值/75分位数 | 左侧列的标识符强度 |
| `low_card_rhs_strength` | RHS 低基数列的信号强度 | 低基数 RHS 列的比例 |
| `high_card_rhs_strength` | RHS 高基数列的信号强度 | 高基数 RHS 列的比例 |
| `identifier_pattern_median/p75` | 近唯一列的模式数统计 | LHS 稀疏度 |

### 10.2 自动调整的参数

根据画像结果，`apply_auto_profile` 自动调整 20+ 个搜索参数：

```python
# 训练参数
afd_loss_weight = 0.25 + 0.20 × empirical_need        # [0.25, 0.45]
max_epochs = 10 + round(10 × empirical_need)           # [10, 24]
early_stop_patience = 3 + round(3 × empirical_need)    # [3, 6]

# 支撑集参数
support_beta = 12 + 36 × high_card_rhs_strength        # [12, 48]
min_support_count = 2 if high_card ≥ 0.45 else 3
max_support_rows = 1024 + round(1024 × high_card)       # [1024, 2048]

# 分数阈值
base_threshold = 0.88 - 0.18 × cardinality_diversity   # [0.70, 0.88]
min_s_ent = min_s_acc = min_score = base_threshold

# 经验混合参数
model_score_weight = 0.80 - 0.45 × empirical_need      # [0.35, 0.80]
empirical_high_card_base = 0.10 + 0.30 × high_card     # [0.10, 0.45]

# 搜索空间
max_lhs_size = 3 if searchable_lhs ≥ 8 and high_card ≥ 0.3 else 2
```

---

## 11. 列信号工具（column_signals.py）

### 11.1 identifier 信号

```python
def column_identifier_signal(role, unique_ratio, empirical_weight):
    if role in {"identifier", "quasi_identifier"}:
        return 1.0
    unique_ratio_signal = ramp(unique_ratio, 0.02, 0.20)
    if role in IDENTIFIER_HINT_ROLES:  # high_card_categorical, entity_name, geo_code, code_categorical
        return max(unique_ratio_signal, empirical_hint)
    return max(unique_ratio_signal, ramp(unique_ratio, lhs_identifier_start_ratio, lhs_identifier_full_ratio))
```

### 11.2 模式数估计

```python
def estimate_pattern_count(observed_unique_count, observed_unique_ratio, observed_rows, total_rows):
    scaled = observed_unique_ratio × total_rows
    return min(max(observed, scaled), total_rows)
```

从抽样观测推断全量数据的离散模式规模。

---

## 12. 配置体系（config.py）

### 12.1 嵌套 Dataclass 结构

```
PipelineConfig
├── PathConfig        # 文件路径（repo_root, dataset_name, 各产物路径）
├── DataConfig        # 数据预处理（seed=20260421, train/val/test_ratio, continuous_bins=32, rare_token_min_freq=1）
├── ModelConfig       # 模型架构（d_model=192, n_heads=4, n_layers=4, ffn_dim=768, dropout=0.1）
├── TrainingConfig    # 训练超参（batch_size=256, lr=1e-3, max_epochs=15, afd_loss_weight=0.25, mask_ratios=(0.2,0.5,0.8)）
├── CalibrationConfig # 校准参数（max_examples_per_rhs=512, optimizer_steps=100）
└── SearchConfig      # 搜索参数（50+ 参数，详见 search.py 打分流程）
```

### 12.2 配置加载顺序

```
default_config(dataset_name)
  → apply_dataset_overrides()     # 数据集级别结构性配置（如 dblp10k 的 permissive 模式）
  → apply_auto_profile()          # 数据驱动的自动调参
  → apply_runtime_overrides()     # 运行时手动覆盖（runtime_settings.py，最高优先级）
  → validate_config()             # 合法性检查
```

**优先级**：runtime_settings.py > auto_profile > dataset_overrides > defaults

### 12.3 运行时覆盖（runtime_settings.py）

```python
ACTIVE_DATASET_NAME = "biocase_gathering"  # 当前默认数据集
ACTIVE_MAX_LHS_SIZE = 3                    # 最大 LHS 列数
ACTIVE_SEARCH_SPACE_MODE = "balanced"      # 搜索空间模式

DATASET_SEARCH_OVERRIDES = {
    "claims": {"support_beta": 24.0, "min_s_acc": 0.85, ...},
    "dblp10k": {"model_score_weight": 0.35, "group_match_mode": "hard", ...},
    ...
}
```

---

## 13. 列策略系统（main/column_policy.py）

AR 通过 `datasets.py` 调用 `main/column_policy.py` 的 `build_column_policy_map` 获取每列的策略。策略决定：

| 字段 | 含义 |
|---|---|
| `role` | 列角色：`identifier`/`quasi_identifier`/`near_identifier`/`constant`/`normal` |
| `family` | 列族：`categorical`/`numeric` |
| `analysis_mode` | 分析模式：`exact`（精确匹配）/ `binned`（分桶）/ `approx`（近似） |
| `rhs_mode` | RHS 搜索模式：`exact` / `off` |
| `unique_count` / `unique_ratio` | 唯一值数量/比例 |
| `empirical_weight` | 经验权重（用于 identifier 信号） |

---

## 14. 数据流全景

```
train_tokens.npy [N, C]
        │
        ├──→ SupportEstimator（统计经验分布，lru_cache(maxsize=256)）
        │       │
        │       ├─ lhs_values, counts, probabilities
        │       ├─ empirical_entropies, empirical_top1
        │       └─ effective_counts, non_null_ratios
        │
        └──→ TokenRowDataset → DataLoader → AnyOrderConditionalTransformer
                                                    │
                                              best.pt（校准后温度 T）
                                                    │
                                             QueryEngine
                                                    │  ┌─ SummaryCache (LRU, 50000)
                                                    │  └─ conditional_summary_batch_valid_rhs
                                                    │
                                             AFDSearcher
                                                    │  ┌─ _score_candidate_core（双信号打分）
                                                    │  ├─ score_candidate（+方向性检验）
                                                    │  ├─ search_rhs（Beam 扩展）
                                                    │  ├─ _minimality_filter
                                                    │  └─ _detect_bidirectional
                                                    │
                                        discovered_fd/*.txt + search_summary.json
```

---

## 15. 关键设计决策

### 15.1 为什么用 Any-Order 而不是自回归？

AFD 搜索需要任意 LHS 子集作为条件（如 `{col_a, col_c} → col_b`）。固定顺序的自回归模型只能从左到右建模，无法高效支持任意条件组合。Any-Order 训练通过随机 mask 让模型学会在任意观测子集下预测其余列，推理时只需构造对应的 observed_mask 即可查询任意 LHS→RHS。

### 15.2 为什么每列独立嵌入？

表格数据中不同列的值域语义完全不同（第 0 列的 "1" 和第 1 列的 "1" 代表不同事物）。共享嵌入表会导致语义混淆。每列独立的 `value_embedding` 和 `output_head` 确保模型正确区分不同列的值空间。

### 15.3 为什么需要四路嵌入？

- `value_embedding`：编码具体取值（"北京" vs "上海"）
- `column_embedding`：编码列身份（第 3 列是 "city"）
- `type_embedding`：编码列类型（categorical vs continuous_bucket）
- `observed_embedding`：编码观测状态（是否作为条件输入）

四路相加让模型在单次前向中同时感知"是什么值"、"是哪一列"、"是什么类型"、"是否被观测"。

### 15.4 为什么同时用模型分数和经验分数？

模型分数泛化能力强但对高基数列（如 ID 列）可能低估依赖强度（模型难以记住所有 ID→属性的映射）。经验分数精确但受数据稀疏影响（小样本 LHS 模式不可靠）。`empirical_blend` 根据列的基数和 identifier 信号自适应混合两者，在模型可靠时信任模型，在经验可靠时信任经验。

### 15.5 为什么需要方向性检验？

强依赖 `A→B` 往往伴随着较弱的反向依赖 `B→A`（因为 B 的取值范围通常更大）。不做方向性检验会产生大量反向误报。通过 `direction_margin = model_s_acc(A→B) - model_s_acc(B→A)` 检验正向依赖是否显著强于反向。

### 15.6 为什么 `support_beta` 要做贝叶斯平滑？

对于出现次数少的 LHS 模式，纯经验概率方差很大。平滑后向独立性假设收缩（`blend = count / (count + beta)`），使低频模式的权重更保守，避免少数几行数据主导打分。

### 15.7 为什么排除 RHS 特殊 token？

训练集中的 `[NULL]` 可能占 RHS 的很大比例（如 99.7% null 的列）。如果不排除，模型对任何 LHS 查询都会给出以 `[NULL]` 为主的分布，entropy 被人为压低，s_ent 虚高。`conditional_summary_batch_valid_rhs` 将特殊 token 概率置零后重新归一化，确保打分只反映实际取值的确定性。

### 15.8 退化模式检测

支撑集构建时检测"单值占比 > 95%"的退化模式（如上游 fillna 产生的 "Unknown" 填充），将其计数置零。这些模式产生的是平凡的经验分布（本质上等于边际分布），会淹没真正有信息量的稀疏模式。

---

## 16. 支持的数据集

| 数据集 | 行数 | 有效列数 | 特点 |
|---|---|---|---|
| adult | 48842 | 13 | 经典人口普查数据，fnlwgt 被排除 |
| claims | 95000+ | 16 | 航班索赔数据，日期列作离散特征 |
| hospital | 100000+ | 9 | 医院数据，Sample 列作离散数值 |
| dblp10k | 10000 | 18 | 文献数据，p1/p2 双侧结构，permissive 模式 |
| tax | 200000+ | 8 | 税务数据，混合连续/离散列 |
| biocase_gathering | 90991 | 6 | 生物采集数据，GT: Gath_AreaDetail→Gath_Country_Name |
| biocase_namedareas | 137710 | 6 | 命名区域数据，5 个 GT FD |
| biocase_highertaxon | 562958 | 3 | 高阶分类数据，GT: HigherTaxonName→HigherTaxonRank |
| biocase_identification | 91799 | 8 | 物种鉴定数据，14 个 GT FD |

---

## 17. 与 main/ 模块的关系

main/ 是另一套基于 Normalizing Flow（标准化流）的 CAFD 发现方案。两者共享：

- `traindata/` 下的数据集（.npy）
- `rule_mining/groundtruth/` 下的真值文件
- `main/column_policy.py` 的列策略系统
- `utils/dataUtils.py` 的数据加载工具

| 对比 | AR 模块 | main 模块 |
|---|---|---|
| 模型 | Transformer (AnyOrderConditionalTransformer) | Normalizing Flow (nflows) |
| 核心思想 | 随机 mask 任意列预测其余列 | 从 flow 采样 + 经验数据融合 |
| 打分 | 模型条件概率 + 经验混合 | NF 采样计数 + 经验计数融合 |
| 搜索 | Beam 扩展 + 方向性检验 | Beam 扩展 + CMI 剪枝 |
| 输出 | `*_conditional_discovered.txt` | `*_model_driven_discovered.txt` |
