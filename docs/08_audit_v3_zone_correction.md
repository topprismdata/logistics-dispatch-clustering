---
title: 审计 v3 更正 — Zone 粒度揭示稳定结构 (用户 + NotebookLM 纠正)
date: 2026-06-26
status: audit-corrected-master-route-valid-at-zone-level
---

# 审计 v3 更正报告

## ⚠ 我错了 — Master Route 在 Zone 粒度成立

用户两次纠正:
1. "我说的主路线是相对稳定, 你查一下互联网, 不要只用你自己的知识"
2. "你可能还是没有真正读懂 amazon2021 竞赛"

加上 NotebookLM 重新解释 Amazon Zone 抽象, **审计 v2 的"ConVRP 不成立"结论是错的**。

错的根源: **评估粒度错误**。Customer-level 测试显示无结构 (median=1), 但 Zone-level 揭示强结构 (median=9, 42.5% 稳定对)。

---

## 1. Amazon 2021 的核心 Zone 抽象 (我之前没读懂)

NotebookLM 重新讲解 + 网络验证 ([INFORMS 数据集 paper](https://pubsonline.informs.org/doi/10.1287/trsc.2022.1173) + [GitHub donato-maragno/Amazon-LMRRC](https://github.com/donato-maragno/Amazon-LMRRC)):

**Zone = 地理规划区域**:
- 格式 `X-Y`, X 是 high-level 宏观分区 (100% 确定性)
- Y 是微观分区 (20-30% 弹性)
- 不是客户, 是 **地理聚集抽象**

**为什么 Zone 稳定但 Customer 不稳定**:
- Customer 每天变 (需求动态)
- Zone 是路网/门禁/单行道等硬约束决定 (物理静态)
- Driver 脑子里有 **Zone 顺序概率表**, 不是 Customer 表

**两阶段方法**:
- Stage 1: Inter-zone TSP (ML/IRL 预测 zone 序列)
- Stage 2: Intra-zone routing (LKH 求具体 stops)

**Amazon 实测**: 宏观 zone 转移 **100% 确定性**。

---

## 2. DC_B 的 Zone-level 数据 (验证用户直觉)

从客户名提取 Zone (郑州各区 + 周边县):

| Zone | 客户数 | 占比 |
|---|---|---|
| 外地:郑州 (郑州市内但无具体区) | 445 | 26.3% |
| District_A | 445 | 26.3% |
| 未识别 | 325 | 19.2% |
| 中牟县 | 261 | 15.4% |
| 管城回族区 | 67 | 4.0% |
| 其他 (金水/二七/惠济/航空港/经开等) | ~75 | 4.4% |

### Customer vs Zone 共现对比

| 指标 | Customer-level (v2 审计) | **Zone-level (v3)** | 提升 |
|---|---|---|---|
| 共现对总数 | 45,437 | 106 | (粒度不同) |
| Median 共现次数 | **1** | **9** | **9x** |
| Mean 共现次数 | 1.51 | **56.8** | **37x** |
| Max 共现次数 | 20 | **1,052** | **52x** |
| Pair 稳定性 median | 0.248 | 0.33 | +33% |
| **稳定性 ≥ 50% 的对** | 0% (CORE) | **42.5%** | **从 0 到 42.5%** |
| Max 稳定性 | 0.12 | **1.00** | 完美 |

### 关键发现: 45 个"模板候选"

Zone pair 中 **42.5% (45 个) 稳定性 ≥ 50%**, 包括 max=1.00 (某些 Zone 对**永远**同路线)。

**这就是 ConVRP / Master Route 的 "stable core"**。用户"相对稳定"完全正确。

### 路线的 Zone 一致性

- 同 Zone (所有客户一个 Zone): **37%**
- 混合 Zone (跨 Zone): 63%

37% 单 Zone 路线 = 强地理聚集, 即使没有 GPS 也能从名字提取。

---

## 3. 我之前错在哪

### 错误 1: 评估粒度错
- 我用 customer-level 共现测试 ConVRP
- Amazon 2021 的核心是 Zone-level
- "Customer 共现 median=1" 是噪声,不是"无结构"

### 错误 2: anchor 定义错
- 我用"高频配送次数"定义 anchor
- 正确: anchor 应是 Zone (地理聚集), 不是单个高频客户
- 高频客户 (京东 897 次) 是 SOP-1 直配, 不是 anchor 搭档

### 错误 3: 没读懂 Amazon Zone
- 我把 Amazon 2021 简化为"路径排序挑战"
- 实际: 它的核心创新是 **Zone 抽象降维**
- 9,184 stops → 几十个 Zone,稳定性从 stops → zones 大幅提升

### 错误 4: 过早接受 agy 结论
- agy 也基于我提供的错误前提 (customer-level)
- agy 说"放弃 ML, 走纯规则" — 现在看也不对
- Zone 抽象下 ML (GNN + Zone sequence) 完全有意义

---

## 4. 正确方向 (v3 = Master Route at Zone level)

```
┌────────────────────────────────────────────────────────────┐
│  Stage 0: Zone 推断                                         │
│  - 从客户名/地址/GPS 提取 Zone                              │
│  - DC_B: 已识别 12 个 Zone (郑州各区 + 周边县)           │
│  - 19.2% 未识别 → 需要地址/GPS 补全                         │
└────────────────────┬───────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 1: Zone-level Master Route (ConVRP 核心)            │
│  - 45 个稳定 Zone pair (≥50% 稳定性) → 模板                 │
│  - 学习 Zone 之间的转移概率 (Amazon macro-zone 100% 确定)  │
│  - 这是真正的 "skeleton route"                              │
└────────────────────┬───────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 2: Customer-to-Zone 分配                             │
│  - 给定当日客户 → 分配到 Zone                                │
│  - ML: GNN 学 customer-zone 匹配 (不是 customer-customer)   │
└────────────────────┬───────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 3: Zone 内 + 跨 Zone 路由                            │
│  - Intra-zone: LKH/ALNS 微观                                │
│  - Inter-zone: 用 Zone 转移概率约束                         │
│  - SOP-1 容量硬约束 (PC>260 单独成线)                       │
└────────────────────────────────────────────────────────────┘
```

---

## 5. 修正后的实施计划

| 任务 | 内容 |
|---|---|
| **P1** | Zone 推断改进 (地址解析 + 未识别 19% 处理) |
| **P2** | Zone-level 共现矩阵 + 转移概率 (45 个稳定对) |
| **P3** | Master Route 模板提取 (从稳定 Zone pair) |
| **P4** | Customer-to-Zone 分配模型 (GNN) |
| **P5** | SOP-1 + ALNS Zone 内路由 |
| **P6** | Hard Mode 评估 (Zone-aware ARI) |

### 修正后的预期结果

| Baseline | 预期 ARI | 说明 |
|---|---|---|
| Customer-level (v1 Siamese) | 0.01 | 已测, 失败 |
| **Zone-level Master Route (新)** | **0.3-0.5** | 利用 45 个稳定 Zone pair |
| Zone + GNN customer-zone 匹配 | 0.4-0.6 | +5-10% 提升 |
| + SOP-1 + ALNS 路由 | 0.5-0.7 | 完整方案 |

---

## 6. 反思

**我应该早听用户**。用户说"主路线是相对稳定"时,我应该:
1. 立即搜索 Amazon Zone 抽象的精确定义
2. 不固守自己的"customer-level 共现"测试
3. 用 NotebookLM (用户也建议了)

两次纠正后我才真懂。审计 v2 是错的, agy 也被我误导。

**核心教训**: **粒度决定一切**。Customer-level 看是噪声, Zone-level 看是结构。Amazon 2021 的核心创新就是 Zone 抽象, 我之前完全没读懂。

---

**审计状态**: v3 更正, Master Route at Zone level **VALID**
**预期 Hard Mode ARI**: 0.3-0.7 (vs v1 Siamese 0.01)
**下一步**: 实施 P1-P6 Zone-aware 架构