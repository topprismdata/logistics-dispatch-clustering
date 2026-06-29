---
title: Beverage Corp DC 自动派单 RL — 设计文档 (基于 NotebookLM 分析)
date: 2026-06-26
status: draft
notebook_id: 96680c13-4b7a-44c8-b433-b70b6d67d099
sources: 14 个 (5 DC_A + 5 Obsidian VRP/物流 + 4 URL)
---

# Beverage Corp DC 自动派单 RL — 设计文档

## 0. 摘要

基于 NotebookLM (notebook_id: `96680c13-4b7a-44c8-b433-b70b6d67d099`) 对 14 个来源 (DC_A 5 份 + Obsidian VRP/物流论文 5 份 + LaDe/Amazon/VRP URL 4 份) 的联合分析,本研究重新设计了DC_A 的 GNN 方案以适配DC_B (Logistics & Beverage Distribution Center DC_B, 5 个月 22,031 条数据)。

**核心结论**:
1. **预期 Pair Recall 50-60%** (DC_A 93.6% 是查表性质,DC_B 24.8% 共现率不支持查表)
2. **不能直接 Vehicle ID embedding** (动态车辆 + 低共现 = 过拟合) → 改为 vehicle CAPACITY + cross-attention
3. **SOP-1 必须硬编码为 action_mask** (不是 reward 惩罚),PC>216.5 必须独立成线
4. **Reward: 距离紧凑度 + 装载率 + 碎片化惩罚** (不是模仿历史 Pair)
5. **Time-based OOT split** (非随机),3.5/0.5/1 月划分
6. **4 baselines**: OR-Tools / LightGBM Pairwise / Graph2Route / BC vehicle ID (后者预期崩溃)

---

## 1. 问题差异分析 (DC_A vs DC_B)

| 维度 | DC_A | DC_B | 启示 |
|---|---|---|---|
| 数据周期 | 314 天 | **5 个月** | 时间依赖严重,必须 OOT 切分 |
| 路线数 | 4,093 | 4,243 | 数量级相当 |
| 中位路线规模 | **11** 客户 | **2** 客户 | DC_B极度碎片化 |
| 客户共现率 | **100%** | **24.8%** | DC_A是查表,DC_B必须预测 |
| 车辆固定性 | 强 (SOP-8) | **90.8%** | 都很强,但客户组合日变 |
| 容量阈值 | PC>500 | **PC>216.5** | 阈值低得多 |
| SOP-8 性质 | 车-客户名单绑定 | 大车拉大单(容量匹配) | 通用化嵌入 |
| 客户类型字段 | 有 (快餐/零售/学校) | **无** | 需从客户名称推断 |

---

## 2. 架构设计

### 2.1 总架构: ML + OR 混合 (2021 Amazon 范式)

```
┌─────────────────────────────────────────────────────────┐
│            "ML 吸收人类经验 + OR 兜底"                   │
│           混合智能架构 (2021 Amazon 三强共识)            │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Stage 1     │  │ Stage 2     │  │ Stage 3     │
│ 历史数据     │  │ ML/RL      │  │ OR 兜底     │
│ 共现矩阵     │→│ Pairwise    │→│ OR-Tools     │
│ PC 分布      │  │ 决策       │  │ 容量硬约束   │
│ 时间规律     │  │            │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 2.2 Stage 1: 特征工程 + 共现矩阵

```python
# 共现矩阵 (借鉴 LaDe Graph2Route + DC_A GNN Pair Recall 框架)
M_cooccur[i][j] = 客户 i 和 j 历史同车的次数
M_cooccur_norm[i][j] = PMI 归一化 (LaDe 标准做法)

# 客户特征向量
f_c = [
    pc_avg,           # 平均日 PC (核心 SOP-1 特征)
    pc_std,           # PC 波动 (SOP-7 容量软边界)
    n_routes,         # 历史出现次数
    n_vehicles,       # 出现过的不同车辆数 (SOP-8 车辆固定性的反向指标)
    weekday_dist,     # 周一-周日出现频率 (SOP-5)
    is_kd,            # KA 客户标志 (推断)
    ...
]
```

### 2.3 Stage 2: ML/RL 核心 (Pairwise 模型 + IRL)

**关键决策**: **不模仿DC_A的 GNN 端到端车辆 ID 预测**,改为 **Pairwise 泛化二分类**。

#### 模型架构:PairwiseNet-v2

```python
class ZhengdongPairwiseNet(nn.Module):
    """
    关键改动 (vs DC_A的 GNN):
      - 不预测车辆 ID (DC_A验证过会过拟合)
      - 不直接用共现图 (24.8% 共现率太低)
      - 改为编码客户物理属性 + PC 容量匹配
      - 输出两两合并的概率
    """
    def __init__(self, d_model=128):
        self.cust_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [pc, lat, lng, is_above_threshold]
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.veh_encoder = nn.Sequential(
            nn.Linear(1, 64),  # load_capacity_tons (SOP-8 物理匹配)
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.cooccur_encoder = nn.Embedding(num_customers, d_model)  # PMI 共现
        self.pair_comparator = nn.Sequential(
            nn.Linear(d_model * 2 + 1, 64),  # +1 for distance
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # IRL-learned reward head
        self.irl_head = nn.Linear(d_model, 4)  # [distance, util, fragmentation, overload]

    def forward(self, cust_pair, veh_capacity, distance, cooccur_ij):
        e_cust = self.cust_encoder(cust_pair)
        e_veh = self.veh_encoder(veh_capacity)
        e_cooccur = self.cooccur_encoder(cooccur_ij)
        pair_feat = torch.cat([e_cust, e_veh, e_cooccur], dim=-1)
        return self.pair_comparator(pair_feat)
```

### 2.4 Stage 3: OR 硬约束兜底 (SOP-1 必须硬编码)

```python
def create_zhengdong_action_mask(current_load, vehicle_cap, customer_pcs):
    """
    借鉴 Meituan IRL + Amazon 2021 经验:
      - PC>阈值 必须独立成线 (action_mask 直接禁掉)
      - 当前车辆剩余容量 < 客户 PC 时禁掉
      - 这不是 reward 惩罚,是物理约束
    """
    mask_overload = current_load.unsqueeze(0) + customer_pcs.unsqueeze(1) > vehicle_cap.unsqueeze(0)
    mask_threshold = (customer_pcs > PC_THRESHOLD).unsqueeze(0).expand_as(mask_overload)
    return mask_overload | mask_threshold

# 应用
action_mask = create_zhengdong_action_mask(current_load, vehicle_cap, customer_pcs)
logits = logits.masked_fill(action_mask, float('-inf'))
probs = torch.softmax(logits, dim=-1)  # 物理越界的动作概率强制归 0
```

### 2.5 Reward 设计 (借鉴 Meituan MaxEnt IRL + Amazon IRL 惩罚项)

```python
def compute_reward(route_customers, vehicle_capacity, dist_matrix):
    """
    不模仿"历史 Pair 是对的" (查表思维),
    改为引导模型"主动发现最优物理组合"。
    """
    # 1. 距离紧凑度 (引导空间聚集, 类似 TSP)
    route_dist = calculate_tsp_tour(route_customers, dist_matrix)
    distance_reward = -route_dist

    # 2. 装载率 (对抗DC_B碎片化)
    total_pc = sum(c['pc'] for c in route_customers)
    utilization = total_pc / vehicle_capacity
    utilization_reward = utilization * 10.0 if utilization > 0.5 else -5.0

    # 3. 碎片化惩罚 (鼓励合法拼车, 中位 3-4 客户)
    fragmentation_penalty = -2.0 if len(route_customers) <= 2 and total_pc < (vehicle_capacity * 0.5) else 0

    return distance_reward + utilization_reward + fragmentation_penalty
```

### 2.6 训练协议

```python
# Time-based OOT Split (借鉴 NotebookLM 建议)
# 严禁随机切分 (物流数据有强时序依赖)
train_end = datetime(2026, 3, 15)   # 第 1-3.5 月
val_end = datetime(2026, 4, 15)     # 第 3.5-4 月
# 第 5 月 (2026-05) = 测试集

# 关键测试: Test Set 必须包含大量 "历史从未共现过的合法拼车 Pair"
# → 才是泛化能力的试金石
```

---

## 3. 实验设计

### 3.1 4 个 Baselines (按复杂度递进)

| # | 模型 | 类型 | 关键配置 | 预期 Pair Recall | 验证假设 |
|---|---|---|---|---|---|
| **B0** | OR-Tools (CVRP) | 纯 OR | 距离矩阵 + 容量约束 | 30-40% | 物理兜底,验证数据完整性 |
| **B1** | LightGBM Pairwise | 传统 ML | 客户特征 + 距离 + 共现分数 | 45-55% | 传统树模型的天花板 |
| **B2** | Graph2Route | DL SOTA (LaDe) | GNN + Pointer Net | 50-60% | 公开榜单的 reference |
| **B3** | BC Vehicle ID | DL 端到端 (DC_A方案) | MultiheadAttn → Vehicle ID | **< 30%** (预期崩) | 反证DC_A方案不适配DC_B |
| **Ours** | **PairwiseNet-v2** | 通用化嵌入 + IRL + 硬约束 | cross-attention + IRL reward | **55-70%** | 我们的核心方案 |

### 3.2 6 个评估指标

| # | 指标 | 类别 | 阈值/目标 | 测量内容 |
|---|---|---|---|---|
| **M1** | Pair Recall | 分组质量 (核心) | >50% | 真同车客户对被预测同车的比例 |
| **M2** | Pair F1 | 分组质量 | 综合 P/R | 防止"全部塞进一车"作弊 |
| **M3** | KRC (Kendall Rank Correlation) | 序列匹配 | ~50% (LaDe 参考 56.99%) | 路线内部排序与历史一致 |
| **M4** | HR@3 (Hit Rate) | 序列匹配 | ~70% | 预测前 3 个客户是否命中 |
| **M5** | PC Overflow Rate | 物理合规 | **必须 = 0%** | 预测路线超容量的比例 |
| **M6** | Route Size Distribution | 业务合规 | 对比真实分布 | 1-2 客户路线占比 |

### 3.3 报告结构

最终报告应包含:
1. **数据分布对比表** (DC_B vs DC_A vs LaDe)
2. **6 baselines × 6 metrics 矩阵** (4 × 6 = 24 cell)
3. **per-size breakdown** (按客户规模: 0-50, 50-100, 100-150, 150-200, 200+)
4. **关键消融实验**:
   - 去掉 SOP-1 mask → Pair Recall 提升但 overflow rate > 0 (不可接受)
   - 去掉 IRL reward → 预测同DC_A BC (查表思维)
   - 去掉共现 embedding → 性能 vs 有共现的对比 (验证 24.8% 共现率是否值得用)
5. **历史未共现 Pair 的测试** (泛化能力试金石)

---

## 4. 模块拆分 (按依赖顺序)

```
[Stage 0]  data/loader.py          ← DONE
[Stage 0]  sop.py                  ← DONE
[Stage 0]  eda.py                  ← NEW: 深度 EDA 验证 SOP
[Stage 1]  split.py                ← NEW: OOT split by date
[Stage 1]  constraints.py          ← Module 07 (P0): SOP-1 mask + SOP-8 嵌入
[Stage 2]  baselines/ortools.py    ← B0: OR-Tools CVRP
[Stage 2]  baselines/lightgbm_pairwise.py  ← B1: LightGBM 二分类
[Stage 2]  baselines/bc_vehicle_id.py      ← B3: BC 端到端 (DC_A方案)
[Stage 2]  models/pairwise_net_v2.py       ← Ours: 通用化嵌入
[Stage 2]  baselines/graph2route_replica.py ← B2: LaDe SOTA 复刻
[Stage 2]  irl_reward.py            ← IRL reward head + 训练循环
[Stage 3]  evaluator.py            ← Module 04: 6 个指标
[Stage 4]  train_all.py            ← 端到端训练 (跑全部 5 baselines + ours)
[Stage 5]  report.py               ← 生成对比报告 (markdown + JSON)
```

---

## 5. 关键决策点 (NotebookLM 分析输出)

### D1: 容量阈值 PC>216.5 vs DC_A PC>500
- DC_A阈值高(>500) → 物理约束稀疏,模型有弹性
- DC_B阈值低(>216.5) → 物理约束密集,**必须 hardcode mask** 而不是 reward
- 决策: **Mask 而非 reward penalty** (NotebookLM 推荐)

### D2: Vehicle ID embedding 是错的
- DC_A GNN 直接 embedding vehicle ID → 在DC_B 24.8% 共现率下会过拟合
- 决策: **改为 vehicle CAPACITY embedding + Cross-Attention** (NotebookLM 推荐)

### D3: BC 模型预期崩溃
- DC_A BC 59.4% (查表性质)
- DC_B BC 预期 <30% (没有查表的稳定结构)
- 决策: **保留 BC 作为反证 baseline**,证明DC_A方案不适配 (NotebookLM 推荐)

### D4: 不模仿"历史 Pair 是对的"
- DC_A reward = Σ共现分数 (查表思维)
- DC_B应该 reward 距离紧凑度 + 装载率 + 碎片化惩罚 (物理合理性)
- 决策: **IRL-style reward + 物理约束 mask** (NotebookLM 推荐)

### D5: Graph2Route 不直接套用
- LaDe Graph2Route SOTA KRC=56.99% 是 10M+ 数据集训练的结果
- DC_B只有 5 个月 22k 条,数据量不足 1/400,Graph2Route 实际性能会断崖式下跌
- 决策: **复刻 Graph2Route 架构但承认其作为 reference**,不预期 SOTA (NotebookLM 评估)

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| BC baseline 反而性能意外高 (车辆固定 90.8%) | 中 | 反证失败 | 必须分析为什么:可能是车牌号记忆而非泛化 |
| SOP-1 mask 太严,所有路线都变成 1 客户 | 低 | 失去分组意义 | 加 mask 微调:允许 PC 阈值 × 0.8 的少量合车 |
| 5 个月数据 OOT 切分后训练集太小 | 中 | 模型欠拟合 | BC 模仿 + 数据增强 (cross-day pairing) |
| 共现 embedding 因 24.8% 稀疏而带来噪音 | 中 | 拖累 Pairwise | 消融实验,对比有/无共现 embedding |
| IRL reward 学到的是局部最优 | 高 | 性能差于预期 | 多种 reward 组合消融 + IRL vs BC 对比 |

---

## 7. Sources (NotebookLM 已摄入 14 个)

1-5. DC_A 5 份 Obsidian 文档 (SOP 提取 + SOP 详解 + RL 设计 + 范式转移 + RTM 可口可乐)
6. Toth & Vigo "Vehicle Routing Problems" (Obsidian 精读)
7. Golden "The Vehicle Routing Problem: Latest Advances" (Obsidian 精读)
8. Nalepa "Smart Delivery Systems" (Obsidian 精读)
9. "A Methodology for Data-Driven Decision-Making in Last Mile" (Obsidian 精读)
10. "12,800字 Deepseek 物流网络优化" (Obsidian 精读)
11. LaDe HuggingFace dataset page (URL)
12. LaDe GitHub (URL)
13. Amazon Last Mile 2021 Challenge paper (arxiv 2407.05285)
14. Vehicle Routing Problem Wikipedia (URL)

**NotebookLM session ID**: `8c58c42f-c8a0-432e-8bcf-5ba3d99be5df` (3 turn conversation)