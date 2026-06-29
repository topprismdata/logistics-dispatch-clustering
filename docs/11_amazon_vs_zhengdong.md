---
title: Amazon 2021 vs DC_B — 数据结构对比
date: 2026-06-27
status: key-finding
---

# Amazon 2021 vs DC_B — 两个完全不同的问题

## 核心对比

| 维度 | Amazon 2021 | DC_B | 差距 |
|---|---|---|---|
| **Stops/路线 (median)** | **151** | **2** | **75x** |
| GPS 覆盖率 | 100% | 0% | -100pp |
| Zone ID | 有 (P-12.3C 格式) | 无 | — |
| 每路线 Zone 数 | 20 | N/A | — |
| 训练路线数 | 6,112 | 2,892 | 2x |
| **问题类型** | **Zone 排序** | **客户分组** | **不同问题** |
| 容量约束 | cm³ 体积 | PC (箱) | 单位不同 |

## 问题形式化差异

### Amazon 2021: **Zone Sequencing (路径排序)**
```
输入: 一条路线的 151 个 stops (已分组到 20 个 zones)
输出: stops 的访问顺序
核心: Zone 之间的转移概率 (macro zone 100% 确定)
评估: Sequence Deviation (SD) — 预测顺序 vs 真实顺序
```

### DC_B: **Customer Grouping (容量分组)**
```
输入: 当日 N 个客户 + 各自 PC (订单量)
输出: 客户 → 车辆的分配 (分组)
核心: 哪些客户应该拼一辆车 (容量约束)
评估: ARI — 预测分组 vs 真实分组
```

**这两个问题不可互换**:
- Amazon 的 hierarchical TSP 解的是"给定 stops, 怎么排序"
- DC_B 要解的是"给定客户, 怎么分组"
- Amazon 的 Zone 抽象在DC_B 不适用 (没 Zone, 没 GPS)

## Amazon 第二名 ("Permission Denied") 方案回顾

论文: [arXiv:2302.02102](https://arxiv.org/abs/2302.02102) "Amazon Last-Mile Delivery Trajectory Prediction Using Hierarchical..."

方案: **Hierarchical TSP**
1. **Macro**: 学 Zone 之间的访问顺序概率 (PPM 马尔可夫模型)
2. **Micro**: 每个 Zone 内部用 LKH-3 解 TSP 排序 stops
3. **关键**: 依赖 Zone ID + GPS + travel_times (1.8GB 距离矩阵)

**在DC_B 不可用**:
- ❌ 没有 Zone ID → 无法做 macro 层
- ❌ 没有 GPS → 无法算距离矩阵
- ❌ 没有 travel_times → 无法做 micro TSP
- ❌ 路线只有 2 个客户 → 排序是伪命题

## IO 论文 (arXiv:2307.07357) 回顾

TU Delft 的 IO 论文是独立的学术分析, 不是 "Permission Denied" 团队自己的代码:
- IO 学的是 **cost function** (edge weights)
- 我们的 IO 实现已验证: PMI 共现 θ=10.29 是 95% 信号
- IO + Louvain = 0.537 ≈ 纯 Louvain 0.540

## 对DC_B 的启示

1. **不能照搬 Amazon 方案** — 问题不同 (排序 vs 分组)
2. **Louvain 是正确选择** — 因为DC_B 是分组问题, 共现图 + 社区发现是自然解法
3. **Zone 抽象无法应用** — 没有 GPS/Zone, 无法做地理聚类
4. **容量约束 (PC) 是唯一硬约束** — 用户已确认

## Amazon 数据 Schema (已下载)

```
data/amazon2021/
├── train_route_data.json (75MB)     # 6,112 routes × stops (lat/lng/zone_id/type)
├── train_actual_sequences.json (9MB) # 真实 stop 访问顺序
├── eval_actual_sequences.json (4MB)  # 评估集顺序
└── Readme.txt                        # CC BY-NC 4.0 license
```

Stop 字段: `lat, lng, type, zone_id` (每个 stop 有 GPS + Zone)
Route 字段: `station_code, date, departure_time, executor_capacity_cm3, route_score, stops`

## 结论

**用户建议非常正确** — 下载 Amazon 数据后, 发现两个问题本质不同:
- Amazon = **宏观路径排序** (151 stops, Zone sequencing)
- DC_B = **微观客户分组** (2 customers, capacity grouping)

Amazon 第二名的 hierarchical TSP **在DC_B 不适用**。
Louvain (ARI 0.540) 是DC_B 分组问题的正确方案。
IO 实验已证明 PMI 共现是 95% 的信号, 其它特征冗余。