---
title: DC_A 框架 — 文献与成熟方法学调研
date: 2026-06-26
status: draft
author: gsd-planner + claude
---

# DC_A 自动派单 RL — 文献与成熟方法学调研

## 0. 摘要

本调研为重新实现Beverage Corp DC 自动派单 RL 系统选型。核心结论:**DC_B 的数据特征与DC_A 显著不同** — 中位路线规模 2 vs DC_A 11, 客户共现率 25% vs DC_A 100%, 车辆固定性 91% vs DC_A ~100%。这意味着DC_A的"查表"直觉 (Pair Recall ≈ 100%) **不再适用**,需要重新评估 GNN、IRL、Graph2Route 类方法的适配性。

## 1. 问题形式化

### 1.1 Beverage Corp DC 派单问题的数学形式

给定:
- 当日出现客户集合 C ⊂ C_all (|C| ≈ 50-300)
- 当日可用车辆集合 V (|V| ≈ 16-86)
- 客户容量约束: 每车总 PC ≤ cap_v (从载重量推出)
- 客户-客户共现矩阵 M (历史路线推出)
- 客户特征向量 f_c (类型、地理、PC 平均)

决策:
- 分配函数 π(c) ∈ V ∪ {未分配}, 即把每个客户分配到一辆车或不上车
- 约束: 每辆车的客户总 PC ≤ cap_v

目标:
- 最大化 Pair Recall: π⁻¹(v) 中的客户对是否真的在同一辆车上 (历史 ground truth)

### 1.2 三大子问题

| 子问题 | 描述 | 与DC_A的关系 |
|---|---|---|
| **A. 客户分组** | 给定客户集合,把它们分成 K 组 (K = |V|) | 直接对应DC_A Pair Recall 问题 |
| **B. 路线排序** | 给定一组客户 + 起点,求最优访问顺序 | DC_A文档说"历史已确定,无需优化" — 但对DC_B 待验证 |
| **C. 容量分配** | 在 A 基础上,把 K 组分配到具体车辆 | DC_A SOP-8 简化为"车-名单绑定", DC_B 待验证 |

DC_A文档重点在 A+B (Pass 1 候选 + Pass 2 验证), Beverage Corp DC 在 OBSIDIAN 文档里强调 A+C (Pair-wise + 车-客户绑定)。

## 2. 业界 SOTA 综述 (2021-2025)

### 2.1 学术基准

#### **LaDe (Last-mile Delivery) Dataset** — KDD 2024 ⭐ 标杆

**核心**: 菜鸟 AI 团队发布,10,677k 包裹、21k 快递员、6 个月数据 (2022-05~2022-10),覆盖上海/杭州/重庆/吉林/烟台 5 城市。

**3 个 benchmark 任务**:
1. `route_prediction` — 预测包裹的派送顺序
2. `stg_prediction` — 时空图预测
3. `time_prediction` — 送达时间预估 (ETA)

**route_prediction Shanghai 排行榜** (KRC = Kendall Rank Correlation,越高越好):

| 方法 | HR@3 | **KRC** | LSD | ED | 类型 |
|---|---|---|---|---|---|
| TimeGreedy | 57.65 | 31.81 | 5.54 | 2.15 | 规则 baseline |
| DistanceGreedy | 60.77 | 39.81 | 5.54 | 2.15 | 规则 baseline |
| OR-Tools | 66.21 | 47.60 | 4.40 | 1.81 | 经典 OR |
| LightGBM | 73.76 | 55.71 | 3.01 | 1.84 | ML 树 |
| FDNET | 73.27 | 53.80 | 3.30 | 1.84 | DL 序列 |
| DeepRoute | 74.68 | 56.60 | 2.98 | 1.79 | Pointer Net |
| **Graph2Route** | **74.84** | **56.99** | **2.86** | **1.77** | **GNN + Pointer Network SOTA** |

**对DC_B 的启示**:
- SOTA KRC = 56.99 在 10,677k 数据集上,**DC_A 的 93.6% 来自"查表"本质**(SOP-4 100% 稳定),不代表模型学习能力强
- DC_B 24.8% 共现率意味着 GNN 的"图结构先验"远不如DC_A强,**预期 KRC 远低于 56.99**
- 需重新校准"Pair Recall 是否可达 90%+"的预期

数据来源: https://huggingface.co/datasets/Cainiao-AI/LaDe-D
论文: KDD 2024, Wu et al., "LaDe: A Large Dataset for Last-mile Delivery"

---

#### **Amazon Last Mile Routing Research Challenge 2021** — 业界标杆 POC

**问题**: 亚马逊 + MIT 开放 9,184 条历史路线,要求研究者重建路线排序与原路线一致。

**SOTA 方法 (官方 Top 3)**:
1. **Cook/Helsgaun (LKH-3 + Zone 转移概率)** — SD=0.0249 (SOTA)
2. **MIT (Inverse Optimization + Pointer Network)** — SD~0.03
3. **HEC Montréal (ML Zone 排序 + LKH TSP)** — SD~0.035

**经典实现** (arxiv 2407.05285):
- Zone 级别: 序列概率模型 + 单步策略迭代
- Stop 级别: 经典 TSP 求解器
- **结果: score 0.0374, 与 Top 3 相当**
- 源码: github.com/aws-samples/amazon-sagemaker-amazon-routing-challenge-sol

**关键洞察**:
- **纯 DL/纯 OR 都不如混合方案** (Top 1 是 OR + ML 概率)
- **Zone ID 是预定义标签** (L-M.PR 格式),让 Stage 1 变成监督学习 — DC_A 没有这个先验, 必须从历史路线自动学习
- 这个挑战用的指标 SD (Sequence Deviation) 与DC_A的 Pair Recall 不同 — **不能直接套用**

---

### 2.2 行业前沿案例 (从 Obsidian + Web)

| 公司/系统 | 年份 | 方法 | 量化效果 |
|---|---|---|---|
| **Meituan** (60M orders/day) | 2021+ | GNN 流聚合 + MaxEnt IRL | 时间 -20.96%,距离 -23.77%,**年省 16 亿 RMB** |
| **Cainiao** (21k couriers) | 2024 | LaDe + GT Pro L4 无人车 | 单车承担网点 55% 工作量,500 万+ 公里 |
| **UPS** | ongoing | ML 边际履约成本估算 | 关闭手动分拣中心, AI 调度 |
| **可口可乐 CCE** | 2010+ | ORTEC SHORTREC | 每箱 -$0.03, 10000 车年省 $4500 万 |
| **Amazon DSP** | ongoing | 司机承包 Zone + Condor 动态 | 灵活用工 C 端 |
| **DC_A** (研究) | 2026-04 | GNN Pair Recall 93.6% (查表) | 314 天验证 |

### 2.3 学术方法谱系

```
纯数学优化 (1959-)
├── Dantzig & Ramser 1959 — VRP 原始定义
├── Toth & Vigo 2002 — VRP 经典教材
├── LKH-3 / OR-Tools — 现代求解器
└── ALNS — 自适应大邻域搜索

机器学习增强 (2018-)
├── LightGBM / Pointer Net — 路线排序预测
├── DeepRoute — Pointer Network 路线生成
├── GNN+IRL (Inverse RL) — 反推人类隐性成本
└── Graph2Route — SOTA on LaDe KRC=56.99

强化学习范式 (2019-)
├── BC (Behavior Cloning) baseline — 模仿历史
├── IRL (Inverse RL) — 学出 reward function
├── HMDispatch / CoRide — 多车协同
└── 最大熵 IRL — Meituan 16 亿 RMB 节约

工业界混合架构 (2021+)
└── "ML 吸收人类经验 + OR 兜底" 三阶段:
    Stage 1: 历史数据数字孪生
    Stage 2: GNN/IRL/Pointer Net 经验提取
    Stage 3: OR 约束保证 (MIP/ALNS/LKH)
```

## 3. DC_A 已有方案评估

### 3.1 DC_A 数据特征 (Obsidian 文档)

| 指标 | DC_A | 与DC_B 相比 |
|---|---|---|
| 时长 | 314 天 | **5 个月 (151 天) — 1/2** |
| 路线数 | 4,093 (266 独立 + 3,827 拼车) | **4,243 — 接近** |
| 中位路线规模 | 11 客户 | **2 客户 — 5x 差距** |
| 路线 100% 稳定 (SOP-4) | **是** | **24.8% — 完全不是查表** |
| 车辆固定制 (SOP-8) | 客户名单绑定 | **待验证 (90.8% 是单一车,0.033 熵)** |
| SOP-1 PC 阈值 | 500 PC | **216.5 PC (动态算出)** |

### 3.2 DC_A的 6 模型递进实测 (Obsidian: 物流配送RL自动派单设计.md)

| 方法 | DC_A Pair Recall | 类别 |
|---|---|---|
| 随机 | 1.6% | baseline |
| 共现层次聚类 | 7.1% | 规则 |
| Pairwise NN | 39.5% | 端到端 NN 二分类 |
| BC 模型 (MultiheadAttn) | 59.4% | 模仿学习,直接预测车辆 ID |
| **GNN + 共现图 + 车辆数** | **93.6%** | 图注意力端到端 |

**DC_A的 93.6% 是查表本质** (SOP-4 100% 共现率),不是模型泛化能力。

### 3.3 DC_A方案对DC_B 的局限

| DC_A假设 | DC_B实际 | 影响 |
|---|---|---|
| 路线 100% 稳定 (查表) | **24.8% 共现率** | GNN 的结构先验大幅减弱,预期 KRC 显著 < 93.6% |
| 路线规模甜区 9-12 客户 | **中位 2,甜区 1-4** | 现有模型/奖励函数不适用 |
| PC>500 单独成线 | **PC>216.5 单独成线** | 阈值不同,但逻辑可复用 |
| 司机固定班拉 11/机动班拉 9 | **5.2 / 3.1 / 2.5** | 司机分工模式存在但更扁平 |
| 车辆绑定客户名单 (SOP-8) | **90.8% 单一车 + 0.033 熵** | 强支持 (类似DC_A) |

## 4. DC_B 适配方案选型

### 4.1 推荐技术栈 (基于业界共识)

**Stage 1 — 历史数据 → 隐性经验提取**:
- **GNN with co-occurrence graph** — Graph2Route 风格的同构图 (Graph Attention Network 优于 GCN)
- **车辆数 one-hot 编码** — 强制输出层与 K 路线数对齐 (DC_A经验)
- **共现图边权 = PMI 归一化** — 比原始 count 更鲁棒

**Stage 2 — IRL 路线偏好量化**:
- **最大熵 IRL** — Meituan 风格,反推 reward function
- 路线规模甜区奖励 (从DC_B 实际分布提取,目标 1-4 客户)
- 类型共线奖励 (需要客户类型字段 — 当前数据缺失,后续可从客户名称推断)

**Stage 3 — OR 约束兜底**:
- **LKH-3 / OR-Tools** — 路线排序 (Stage 1 输出顺序后微调)
- **容量硬约束** — SOP-1 PC>216.5 单独成线 (NEW: 必须硬编码,不能从数据学)
- **车辆-客户绑定 (SOP-8)** — 嵌入向量 (NEW: 车-客户名单共现矩阵作为先验)

**训练范式**:
- **BC baseline** (vehicle ID) — DC_A方案,作为 baseline 目标
- **Pairwise 二分类** — DC_A经验:车辆 ID 预测失败 (57.7%),pair 二分类是正确 formulation
- **GNN + 早停** — DC_A发现 1 epoch 即最优 (94.8%),后续过拟合。需强正则化

### 4.2 评估指标

| 指标 | 定义 | DC_A参考 | 预期DC_B |
|---|---|---|---|
| **Pair Recall** | 真同车客户对中,被预测为同车的比例 | 93.6% | **30-50%** (查表性质减弱) |
| **Per-size breakdown** | 按客户规模分桶 | BC: 100%→52% (50→300+), GNN 全 90%+ | 需重测 |
| **PC overflow rate** | 预测路线超过车辆容量的比例 | 0% (硬约束) | 0% (硬约束) |
| **Route-level F1** | 整条路线是否完全匹配 | 90%+ (查表) | **< 50%** |

### 4.3 关键工程决策

| 决策 | 选项 | 推荐 | 理由 |
|---|---|---|---|
| 容量约束编码 | 硬编码 mask / reward 惩罚 / OR 硬约束 | **hardcode mask** | DC_A SOP-1 数据里 PC>500 合车=0 次,模型学不到 |
| 客户特征 | 类型 + PC + 地理 / 仅 PC | **PC + 后续加** | 当前数据没有客户类型字段 |
| 车辆 ID 预测 | 是/否 | **否** (Pairwise 替代) | DC_A验证过 (57.7% vs 59.4% BC,Pairwise NN 39.5%) |
| 时间特征 | 加入 / 不加入 | **加入** (星期 + 月) | DC_A SOP-5 周二最大 |
| 真实数据 vs 合成 | 真实 / 混合 | **真实** (用户已提供 22,031 条) | 符合用户需求 |

## 5. 实施路线 (4 阶段)

### 阶段 1: 数据准备 + SOP 重挖掘 (本次任务)
- ✅ Loader (22,031 deliveries → 4,243 routes)
- ✅ SOP extractor (PC>216.5, 甜区 1-4, 24.8% 共现率, 90.8% 车辆固定性)
- **TODO**: 深度 EDA — 看每个 SOP 的具体分布、相关性、时间序列

### 阶段 2: BC baseline + Pairwise 基线
- **Module 03**: BC 训练 (DC_A方案,预期 Pair Recall 较低,DC_B数据 24.8% 共现率)
- **Module 05**: Pairwise NN (Siamese,二分类)
- 输出对比基线

### 阶段 3: GNN + 共现图 (核心)
- **Module 06**: GAT 模型 + 共现图 + 车辆数 one-hot
- 早停 + 强正则化 (避免过拟合,DC_A 1 epoch 即最优)
- 加入 Stage 3 OR 兜底

### 阶段 4: P0 硬约束 + 评估
- **Module 07**: SOP-1 容量硬约束 + SOP-8 车辆-客户名单嵌入
- **Module 04**: 评估器 (Pair Recall + per-size + 容量溢出率)
- 综合报告对比

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| DC_B 24.8% 共现率远低于DC_A 100% | GNN 性能可能远低于DC_A 93.6% | 用 Pairwise 基线 + IRL 混合架构 |
| 数据只有 5 个月 (vs DC_A 10 个月) | 训练样本少 | 加入 BC 模仿学习弥补 |
| 客户类型字段缺失 | SOP-3 类型共线无法验证 | 从客户名称推断 (便利店/超市/餐饮) |
| 司机轮换模式 (0.033 熵) 表明 SOP-8 强支持 | 这是机会 | 直接用车辆-客户名单共现矩阵作为嵌入 |

## 7. Sources (引用)

### Web (本文档新增)
- [LaDe dataset (HuggingFace)](https://huggingface.co/datasets/Cainiao-AI/LaDe-D)
- [LaDe GitHub](https://github.com/wenhaomin/LaDe) — route_prediction benchmark
- [Amazon Last Mile Routing Challenge 2021 (arxiv 2407.05285)](https://arxiv.org/abs/2407.05285) — Hierarchical zone + TSP
- [Vehicle Routing Problem (Wikipedia)](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [Knowledge-Enhanced Spatial-Temporal Routing 2025](https://scholar.google.com/) — GNN + knowledge graphs
- [Two-stage GNN for chain stores 2025](https://scholar.google.com/) — GNN + attention + residual

### Obsidian (iCloud) — 已读
- DC_ADC人类调度隐性SOP.md (313 行, 2026-04-22)
- DC_ADC人类调度隐性SOP详解.md (737 行, 2026-04-22)
- 物流配送RL自动派单设计.md (330 行, 2026-04-21)
- 全球物流末端配送：从运筹学到深度学习的范式转移.md (223 行, 2026-04-29)
- RTM-可口可乐案例知识体系.md (165 行, 2026-04-14)
- 12,800字 Deepseek 物流网络优化 (104 行, 2025-02-14)
- Awesome-CTDP-Spatial-Optimization/wiki/ (12 个深度精读, VRP/IRLP/Polsby-Popper/Reock)
- Beverage Corp可口可乐 Swire 项目 + Bain 报告

### Obsidian (iCloud) — 待读 (按相关度)
- Vehicle Routing Problems (Toth & Vigo, 124 行精读)
- The Vehicle Routing Problem (Golden, 123 行)
- Smart Delivery Systems (Nalepa, 122 行)
- Literature Review on VRP Approaches (122 行)
- Models for Practical Routing Problems (124 行 + 198 行)
- A Methodology for Data-Driven Decision-Making in Last Mile (127 行)
- Optimizing Commercial Teams and Territory Design (120 行)