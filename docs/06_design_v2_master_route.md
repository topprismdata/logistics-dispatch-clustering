---
title: 设计更新 v2 — Master Route (ConVRP) 架构
date: 2026-06-26
status: design-pivot-after-audit
source: 用户洞察 "搜 master route / 骨架设计 / 高频客户"
---

# 设计更新 v2: 从 Siamese Pair 预测 → Master Route (ConVRP) 架构

## 0. 为什么改方向

审计 (audit.md) + Hard Mode 测试揭示 Siamese Pair 预测方案的根本问题:
- Hard Mode ARI = **0.010** (随机水平)
- 平均预测簇 11.7 客户 (vs 真实 ~3)
- Precision 7.9% (Easy mode 的 100% 是幻觉)

用户洞察 + 文献搜索找到正确方向: **Master Route / ConVRP / PVRP**。
高频客户形成稳定"骨架路线",低频客户挂到骨架上 — 这是真实业务逻辑,也是成熟运筹学方法。

---

## 1. 文献依据

| 概念 | 来源 | 核心思想 |
|---|---|---|
| **ConVRP (Consistent VRP)** | Groër/Golden/Wasil 2009 | 司机-客户一致性优先, 同司机每周跑同区域 |
| **PVRP (Periodic VRP)** | 经典 OR | 客户在时间窗内多次访问, 分配访问日 + 卡车 |
| **Skeleton Route** | 多篇 transit/routing 论文 | 关键节点形成骨架, 其他节点插入 |
| **Anchor Customers** | ConVRP 文献 | 高频/大客户绑定特定路线作为锚点 |
| **Cluster-First Route-Second** | PVRP 经典启发式 | 先聚类客户 (按日/卡车), 再做路径排序 |
| **Master Route** (工业) | Route4Me/RouteMagic/Kiva Logic | 可重用路线模板, 用于周期配送 |

**学术搜索引用**:
- [Groër et al. ConVRP](https://link.springer.com/content/pdf/10.1007/978-0-387-39934-8.pdf) — "develops a rough (skeleton) route which can then be adapted and modified to accommodate actual node locations"
- [Solving Customer-to-Truck Assignment (PVRP)](https://www.redalyc.org/journal/477/47753681016/html/) — "groups customers visited the same day by the same truck as close as possible using centroid-based clustering"
- [Cluster-First Route-Second Heuristic for PVRP](https://ejournal.umm.ac.id/index.php/industri/article/download/8787/pdf/48760)

---

## 2. DC_B 数据的长尾结构 (验证想法)

| 频次桶 | 客户数 | 占比 | 贡献路线次 | 角色 |
|---|---|---|---|---|
| 1 次 (偶发) | 247 | 14.6% | 1% | 一次性,临时插入 |
| 2-5 次 (低频) | 470 | 27.7% | 7% | 临时插入 |
| 6-20 次 (中频) | 685 | 40.4% | 36% | 主力,挂载到骨架 |
| 21-50 次 (高频) | 250 | 14.8% | 33% | 高价值,挂载到骨架 |
| 51-100 次 (核心) | 34 | 2.0% | 10% | **anchor 候选** |
| 100+ 次 (骨架) | 8 | 0.5% | 13% | **master anchor** |

**Top 8 骨架客户** (贡献 13% 路线次):
1. 北京京东世纪信息技术有限公司 — 897 次 (~6 次/天)
2. 天津小蚁科技有限公司 (小米) — 850 次
3. 郑州旭之姣商贸有限公司 — 415 次
4. 上海盒马物联网有限公司 (盒马) — 226 次
5. 河南景田中央厨房有限公司 — 147 次
6. 河南全伊商贸有限公司 (avg_pc=506 大客户) — 140 次
7. 郑州康品源商贸有限公司 — 102 次
8. 0600233998 (?) — 101 次

**这 8 + 34 = 42 个高频客户 = anchor 客户**,贡献 23% 路线次。

---

## 3. 新架构: Master Route (ConVRP-style)

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 0: 频次分层                                            │
│  - 42 个 anchor 客户 (>50 次)                                │
│  - 250 个高频 (21-50 次)                                     │
│  - 685 个中频 (6-20 次)                                      │
│  - 470 个低频 (2-5 次)                                       │
│  - 247 个偶发 (1 次)                                         │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Anchor 骨架路线 (Master Route)                     │
│  - 用历史共现矩阵找出 anchor 之间的稳定组合                   │
│  - 每个 anchor 配 1-2 个 anchor "搭档" (高 PMI)              │
│  - 形成 ~20 个 master routes (anchor 组合)                   │
│  → 这是 ConVRP 的 "skeleton route"                           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: 高频/中频客户挂载 (Cluster-First)                  │
│  - 对每个非 anchor 客户, 找最相似的 master route             │
│  - 相似度 = Siamese pair prob + 地理距离 + PC 容量匹配        │
│  - 容量约束硬检查 (SOP-1 mask)                                │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: 低频/偶发客户插入 (Greedy Insertion)               │
│  - 按 PC 容量剩余插入                                        │
│  - 若所有路线满载 → 单独成线                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 与原 Siamese 方案对比

| 维度 | Siamese Pair (原方案) | Master Route (新方案) |
|---|---|---|
| 形式 | 学 N² pair 概率, 聚类 | 先 anchor 骨架, 再挂载 |
| 复杂度 | O(N²) 对 | O(N × K anchor) |
| Hard Mode ARI | 0.010 (随机) | 预期 > 0.3 (利用结构) |
| 解释性 | 黑盒概率 | 业务可解释 (anchor → cluster) |
| 容量约束 | 难 (后处理) | 天然 (每步硬约束检查) |
| 学术依据 | Pairwise NN | ConVRP + PVRP + Skeleton |
| 计算成本 | 高 | 低 |

---

## 5. 实施计划 (阶段 1.5 v2)

| 任务 | 输出 |
|---|---|
| **M1** Master route 构建 (从 anchor 共现矩阵) | 20-30 个稳定 anchor 组合 |
| **M2** Siamese 改为 anchor-customer 相似度 (而不是 customer-customer) | 改 forward: query=anchor, key=customer |
| **M3** Cluster-First 挂载算法 | 非-anchor 客户 → master route 分配 |
| **M4** SOP-1 容量硬约束 mask | PC>260 不可合车 |
| **M5** Hard Mode 评估 (ARI + Partition F1) | 真实数字 |
| **M6** ConVRP baseline (纯规则, 不用 ML) | 比较 ML 是否真有价值 |

---

## 6. 决策记录

### D1: master route 是正确方向 (用户 + 文献一致)
- 用户洞察: "高频客户骨架"
- 学术: ConVRP / PVRP / Skeleton Route / Anchor Customers
- 工业: Route4Me / RouteMagic / Kiva Logic 的 Master Route 功能
- 数据: DC_B 42 个 anchor 客户贡献 23% 路线次 (符合 Pareto)

### D2: Siamese 不废弃, 降级为 anchor-customer 相似度
- 原 customer-customer pair 预测 → Hard Mode ARI=0.01 失败
- 改为 query=anchor, key=customer → 简化任务 + 利用 anchor 稳定性
- 这才是 Siamese 在 ConVRP 框架下的正确用法

### D3: 容量约束从 reward 改为 mask
- SOP-1 PC>260 必须硬约束 (action mask)
- NotebookLM 早就建议这个, 但阶段 1 没实现
- 现在 ConVRP 框架下每步插入都检查容量, 天然合规

### D4: ConVRP 规则 baseline 作为下界
- 不用 ML, 纯 anchor 共现 + 容量约束
- 如果 ML 跑不过这个 baseline → ML 没价值
- 如果跑得过 → ML 学到了规则没捕捉的东西

---

**Sources**:
- [Groër et al. ConVRP](https://link.springer.com/content/pdf/10.1007/978-0-387-39934-8.pdf)
- [Customer-to-Truck Assignment PVRP](https://www.redalyc.org/journal/477/47753681016/html/)
- [Cluster-First Route-Second PVRP](https://ejournal.umm.ac.id/index.php/industri/article/download/8787/pdf/48760)
- [Route4Me Master Routes](https://support.route4me.com/repeating-route-templates-for-recurring-schedule-delivery-routes/)
- [Memgraph VRP Overview](https://memgraph.com/blog/diving-into-the-vehicle-routing-problem)