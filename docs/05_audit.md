---
title: Beverage Corp DC 自动派单 RL — 阶段 1 审计报告 (3 轮自审 + agy 交叉验证)
date: 2026-06-26
status: audit-failed-必须返工
reviewers: Claude Sonnet 4.6 (3 轮自审) + agy 1.0.8 (Antigravity 独立交叉验证)
---

# 阶段 1 审计报告

## ⚠ 结论先行: 阶段 1 结果不可信, 必须返工

3 轮自审 + agy 独立交叉验证**一致认定**: 之前报告的 "Pairwise 83.2% / BC 0%" 有严重方法学缺陷。

**5 个 critical bug + 3 个方法学问题**:

| # | 问题 | 严重度 | 自审发现 | agy 也发现 |
|---|---|---|---|---|
| C1 | "Easy Mode" 评估 — 已知路线边界内枚举对 | 🔴 致命 | ✓ | ✓ |
| C2 | Precision 100% 幻觉 — 无跨路线负样本 | 🔴 致命 | ✓ | ✓ |
| C3 | 91.2% test 客户已在 train (memory inflation) | 🔴 致命 | ✓ | ✓ |
| C4 | B3 BC 0% 因 threshold 0.7 未校准 (结论过早) | 🟠 严重 | partial | ✓ |
| C5 | KRC = 1.000 空预测时假阳性 (bug) | 🟠 严重 | ✓ | ✓ |
| M1 | 38% test 路线只有 1 客户 (Pair Recall 不适用) | 🟡 中 | ✓ | — |
| M2 | 38% test 路线 customer_ids 有重复 (数据质量) | 🟡 中 | ✓ | — |
| M3 | "Pairs eval'd" 与实际数不一致 (跨路线去重) | 🟢 低 | ✓ | — |

---

## 1. 3 轮自审发现

### 自审 1 — 方法学正确性

```
数据 leakage 检查
train 客户: 1,567
val 客户: 1,015, 其中在 train: 963 (94.9%)
test 客户: 1,137, 其中在 train: 1,037 (91.2%)  ← 致命
test 新客户 (train 没见过): 100 (8.8%)

Pair Recall 分母分析
test 路线: 823
  ≥2 客户 (有对): 313 (38%)
  1 客户 (无对): 510 (62%)
```

**致命**: 91% 的 test 客户在 train 见过,中位出现 8 次。Siamese 模型对这些客户的"是否同路线"判断**本质是查表**,不是泛化。

### 自审 2 — 代码 bug

```
Oracle 测试 (predictions == truth, 应该 100%)
  Pair Recall: 1.000 ✓
  HR@3:       0.660  ← BUG (oracle 应该 100%)

空预测测试 (Pair Recall 应 0%, KRC 应 0 或 NaN)
  Pair Recall: 0.000 ✓
  KRC:        1.000  ← BUG (空预测不应给 KRC=1)
```

**KRC bug**: B3 BC 报告的 KRC=1.000 是 evaluator 退化 (空预测 → union-find 不合并 → pred_ranking == true_ranking → KRC=1)。

**HR@3 bug**: oracle 应该 100% 但只 66% — 因为 `customer_ids` 含重复,`_all_customer_pairs` 用 `sorted(set())` 去重但 HR@3 用原 tuple。

### 自审 3 — NotebookLM 假设强度 + 数据 leakage

```
时序切分 (干净)
train: 2026-01-01 → 2026-04-16 (81 天)
val:   2026-04-17 → 2026-05-01 (11 天)
test:  2026-05-05 → 2026-05-31 (24 天)
train/val/test 日期重叠: 0 ✓
train/test shipment_id 重叠: 0 ✓

但客户跨 split 重复
- B2B 客户自然跨日重复 (这是物流业务的本质, 不是 bug)
- 但意味着 91% test 客户在 train 出现过 → 模型在 test 上是 in-distribution
- 真泛化 (OOD) 只有 100/1137 = 8.8% 的客户

数据质量
312/823 (38%) test 路线 customer_ids 有重复 (同一客户多张交货单)
```

---

## 2. agy 独立交叉验证 (Antigravity 1.0.8)

agy 给出**比我更狠的批评**,完全独立确认了核心问题:

### agy 引用 (节选)

> **Pair Recall 83.2% / Precision 100% 水分 > 80%**
> 在真实派单中,我们不知道哪些客户属于同一条路线。评估在"已知真实 route_id"内部做 C(N,2) 枚举,相当于把最难的**跨路线隔离 (Inter-route Separation) 任务完全过滤掉了**。模型只需要在一个极小的、本来就高度同质化的局部客户集里预测"是否同路线"。

> **Precision 100% 的幻觉**: 因为分母只有同路线内的对,且 Threshold=0.1 极低,模型只要对同路线内的样本无脑预测"是",在没有跨路线干扰项输入的情况下,Precision 自然是 100%。

> **真实派单预期**: 一旦放入全量客户 (从 1694 个客户中自由聚类), 负样本呈 O(N²) 增加。Precision 会断崖式下跌至 5% 以下;若为保 Precision 提高阈值, **真实 Pair Recall 预计会跌落至 20% - 30%**。

> **B3 BC 0% 结论下得过早**: DC_A 的余弦相似度阈值 0.7 没做重校准 (Calibration)。直接照搬而不画 PR 曲线找最佳 F1 阈值是不专业的。

> **KRC 完全不可信**: KRC = 1.000 当 Pair Recall = 0% 是数学退化 (空预测导致分母为 0,代码可能填充默认值 1.0)。

> **整体方向存在"学术化自嗨"的偏差**: Siamese 预测"是否同路线"只是表征学习 (Representation Learning),**它本身根本不是派单决策器, 更谈不上 RL**。

agy 的整改清单:
1. 立即废除 Easy Mode, 切换到 Hard Mode (全量聚类 + ARI / Partition F1)
2. 修复 KRC bug (NaN / 常数边界)
3. 重新校准 B3 BC 阈值 (PR 曲线)
4. 引入硬约束评估 (超载率统计)
5. 加 OR-Tools / LKH-3 baseline 对比

---

## 3. 用户指示对齐

用户最新指示: **"先不要考虑配送顺序"**。

这正好对齐 NotebookLM 的洞见 (DC_B中位 2 客户, 排序是伪命题)。所以:
- ❌ **KRC, HR@3 不再关注** (这两个 bug 也变得不重要)
- ✅ **Pair Recall / Precision / F1 才是核心** (分组质量)
- ✅ **PC Overflow Rate 重要** (硬约束)
- ✅ **聚类质量 (ARI, Partition F1)** 替代排序指标

---

## 4. 必须返工的整改方案

### P0 (立即做, 阻塞所有结论)

| # | 任务 | 影响 |
|---|---|---|
| **R1** | 修复 KRC bug (空预测返回 0/NaN, 不返回 1) | 所有历史 KRC 数字失效 |
| **R2** | 修复 HR@3 bug (用 dedupe 后的 customer_ids) | 或废弃此指标 |
| **R3** | 实现 **Hard Mode 评估**: 给定全日客户, 让模型自由聚类, 对比真实分组 (ARI + Partition F1) | 真实 Pair Recall 数字 |
| **R4** | 重新校准 B3 BC 阈值 (PR 曲线扫描, 不是固定 0.7) | B3 BC 真实分数 |
| **R5** | SOP-1 容量硬约束 mask (PC>260 不可合车) | PC Overflow → 0% |

### P1 (核心, 验证方向是否对)

| # | 任务 | 影响 |
|---|---|---|
| **R6** | OR-Tools CVRP baseline (业界纯 OR 比较) | Siamese 是否跑得过传统 VRP |
| **R7** | LightGBM Pairwise baseline (业界 ML 比较) | 业界 ML 天花板 |
| **R8** | 真泛化测试 (只在 train 没见过的客户上评估) | 看清"查表 vs 泛化" |

### P2 (可选, 增强说服力)

| # | 任务 | 影响 |
|---|---|---|
| **R9** | Graph2Route 复刻 (LaDe SOTA 对比) | 与公开 benchmark 比对 |
| **R10** | PairwiseNet-v2 (车型 CAPACITY + IRL reward) | 我们的核心方案 (从表征学习升级为决策器) |

---

## 5. agy 的关键警告: "Siamese ≠ RL ≠ 派单决策器"

agy 指出一个**根本方向问题**: 当前 Siamese 只是**表征学习** (学客户相似性), 不是真正的**派单决策器**:

- Siamese 输出: P(客户 A 与 B 同路线) — 这是相似性度量
- 真实派单: 决策"哪些客户上哪辆车", 还要满足:
  - 容量约束 (PC 不超载)
  - 时间窗 (虽用户说排序不重要,但容量是 hard constraint)
  - 司机/车辆可用性

要做成真正的派单系统,需要:
1. Siamese 作为相似性先验 (Stage 1)
2. 聚类/分组算法 (Stage 2): 用相似度构建图, K-way partition 或 spectral clustering
3. OR 求解器兜底 (Stage 3): 保证容量合规

---

## 6. 我对 agy 批评的回应

agy 完全正确,我接受所有批评。具体:

| agy 批评 | 我的回应 |
|---|---|
| "Pair Recall 83.2% 水分 > 80%" | ✅ 同意。Easy mode 让数字虚高,真实 Hard mode 预期 20-30% |
| "B3 BC 0% 结论过早" | ✅ 同意。应该画 PR 曲线扫描阈值,而不是固定 0.7 |
| "KRC 完全不可信" | ✅ 同意。已确认是 evaluator 退化 bug,空预测返回 1 |
| "NotebookLM 可信度极低" | ⚠ 部分同意。NotebookLM 给方向但没量化,应作为起点不是终点 |
| "学术化自嗨" | ✅ 同意。Siamese ≠ RL ≠ 派单器,只是表征学习 |

**NotebookLM 的价值**: 方向指引 (Pairwise 优于 BC, 容量约束必须硬编码, OOT split) 这些是对的。具体数字 (50-60%) 需要重新在 Hard mode 上验证。

---

## 7. 阶段 1 状态: **FAILED, 必须返工**

原报告的 "Pair Recall 83.2% / BC 0%" 是不可信的。需要:
1. 修复 5 个 bug (R1-R5)
2. 实现 Hard mode 评估 (R3)
3. 重新跑 B1 + B3 baselines
4. 加 OR-Tools baseline (R6)

预计返工时间: 4-6 小时。

完成 Hard mode 后, 预期:
- B1 Pairwise 真实 Pair Recall: 20-30% (agy 预测)
- B3 BC 真实 Pair Recall: 调参后可能 15-25% (不是 0%)
- OR-Tools: 30-40% (纯数学兜底)
- PairwiseNet-v2 (我们方案): TBD

---

**审计完成时间**: 2026-06-26
**审计方式**: 3 轮自审 (Claude Sonnet 4.6) + agy 1.0.8 独立交叉验证
**结论**: 阶段 1 报告失效, 启动阶段 1.5 (返工)