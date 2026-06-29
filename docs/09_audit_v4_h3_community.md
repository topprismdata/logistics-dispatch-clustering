---
title: 审计 v4 — Zone 抽象方向对, 但粒度仍错 (H3 + 社区发现才能破 ARI 0.2)
date: 2026-06-26
status: audit-v4-zone-right-granularity-wrong
reviewers: Claude Sonnet 4.6 (自审) + agy 1.0.8 + NotebookLM + 用户
---

# 审计 v4 报告

## 进化轨迹

| 版本 | 假设 | Hard Mode ARI | 判定 |
|---|---|---|---|
| v1 Siamese Pair | customer-customer 相似 | 0.010 | ❌ 失败 |
| v2 ConVRP (customer) | 高频客户 = anchor | 0% 稳定 | ❌ 失败 |
| v3 ConVRP (Zone) | Zone 共现稳定 | **0.070** | ⚠ 部分对, 粒度仍错 |
| **v4 H3 + 社区发现** | 待验证 | 目标 0.3+ | 🔬 |

**结论**: Zone 抽象方向是对的 (用户洞察 + Amazon 文献),但**行政 Zone 太粗**,需要 H3 hex grid 或图社区发现。

---

## 1. v3 Zone 假设的真实强度 (严格审查)

### 1.1 严格模板候选 (min_routes≥10 AND stab≥0.5, 排除污染 Zone)

**只 13 个真正稳定模板** (v3 报 42.5% 是被异常值污染):
- District_A - Zone_1: 97% (shared=28)
- District_A - District_B: 80% (shared=114)
- District_A - Zone_2: 79% (shared=87)
- 中牟县 - 航空港区: 69% (shared=41)
- 管城区 - 管城回族区: 64% (shared=129)
- 等 13 对

→ 在 263 种 Zone × carrier 组合中,**只 13 个稳定 (5%)**。

### 1.2 Zone-as-cluster Hard Mode (真实评估)

```
ARI:                    0.070  (vs Siamese 0.010, +0.06)
Partition Recall:       33.8%  (Recall 上限被 Zone 边界锁死)
Partition Precision:    11.7%  (avg cluster 10.9 vs 真实 ~3)
Avg predicted cluster:  10.9   (Zone 太粗, 一 Zone 多路线)
```

→ **ARI 0.07 是伪结构**。Zone 比客户好 7x,但仍远不够。

### 1.3 Zone 内有"次结构"

- 同 Zone 单日多路线: **36.8%** 的 Zone-day 有多条路线 (median 2, max 7)
- → Zone 内还要再拆,Zone 不是终点粒度

---

## 2. agy 独立验证 (第 3 轮)

agy 给出与我自审一致的判断 + 关键建议:

> **ARI 0.07 = 伪结构 (带偏差的噪声)**
> 95%+ 配送区域处于无序高波动状态,只 13 个稳定模板覆盖不了。

> **分层处理治标不治本**
> Layer 1 (Zone) 把 Recall 上限锁死 33.8%, 加 Layer 2 (carrier) 割裂地理, 加 Layer 3 (容量) 只能提 Precision 但 Recall 继续跌。
> **预期 ARI 仅 0.12-0.17, 突破不了 0.2**。

> **真正能让 ARI 破 0.3 的方向**:
> 1. **H3 L8/L9 hex grid** (半径 100-300m) 替代行政 Zone
> 2. **客户共现图社区发现** (Louvain/Leiden) — 找天然紧密客户群
> 3. **Anchor-based 动态吸附聚类** + 距离衰减函数

> **防错流程**:
> 1. Baseline Shield — 必须对比 K-means/随机,提升 < 50% 直接废弃
> 2. Bound Audit — 先算跨边界路线比例,Recall 上限 < 60% 一票否决
> 3. Distribution Audit — 禁止只看 mean,必须 p25/p50/p75/p95 分位

---

## 3. 用户的"分层处理"对, 但层要重新定义

用户建议"分层"是对的, 但我之前理解错:

| 错误分层 (v3) | 正确分层 (v4) |
|---|---|
| Layer 1 行政 Zone | Layer 0 **H3 hex grid** (100-300m) |
| Layer 2 Zone × carrier | Layer 1 **图社区** (Louvain on customer co-occurrence) |
| Layer 3 容量拆分 | Layer 2 **Anchor 吸附** (核心客户动态聚拢) |
| — | Layer 3 **容量 + 时间窗拆分** |

行政 Zone 是邮政/规划边界, 不是物流实际边界。**H3 hex grid 是 Uber 开源的全球网格系统**, 对物流场景更合适。

---

## 4. 为什么我又错了一次

### 错误 5: Zone 定义粒度错 (v3 → v4)
- 行政 Zone (District_A/District_B) 覆盖太大, 一 Zone 多路线
- 应该用 H3 hex (100-300m) 这种细粒度

### 错误 6: 没对比基线
- 我直接报 ARI 0.07, 没对比 K-means/随机基线
- 不知道这 0.07 是真信号还是噪声 +1 σ

### 错误 7: 只看平均数
- v3 报 "42.5% 稳定对" 是 mean, 被 shared=1 的小样本污染
- 应该看 p25/p50/p75/p95 分布

### agy 防错流程采纳
未来每个 baseline 必须过 3 关:
1. **Baseline Shield**: ARI 必须比 K-means + 50% 以上
2. **Bound Audit**: 跨边界路线 < 60% Recall 上限 → 否决
3. **Distribution Audit**: 报分位数, 不只 mean

---

## 5. v4 真正方向

### 架构 v4: H3 + 图社区 + Anchor 吸附

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 0: H3 Hex Grid 推断                                   │
│  - 客户位置 → H3 L8/L9 cell (100-300m)                       │
│  - 替代行政 Zone (太粗)                                      │
│  - 需要 GPS / 地址解析 (19.2% 未识别需补全)                  │
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: 客户共现图社区发现                                  │
│  - 节点: 客户, 边权: 历史共现次数 (PMI 归一化)                │
│  - Louvain/Leiden 算法分社区                                 │
│  - 这才是"相对稳定"的真正 master route 单元                  │
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: Anchor-based 动态吸附                              │
│  - 核心客户 (社区中心) 作为 anchor                           │
│  - 周边 customer 按距离衰减 + 时空重合吸附                   │
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: 容量 + 时间窗拆分                                  │
│  - SOP-1 PC>260 单独成线                                     │
│  - ETA 时间窗约束                                            │
│  - ALNS 局部寻优                                             │
└──────────────────────────────────────────────────────────────┘
```

### 必须先做的 baseline 对比 (agy Baseline Shield)

| Baseline | 预期 ARI | 说明 |
|---|---|---|
| 随机聚类 | 0.0 | floor |
| K-means (按 GPS k=N 路线数) | 0.1-0.2 | 简单地理 baseline |
| 行政 Zone | **0.07** (已测) | v3 结果 |
| **H3 hex grid** | 0.15-0.25 | v4 Stage 0 |
| **图社区 Louvain** | 0.25-0.40 | v4 Stage 1 |
| + Anchor 吸附 + SOP-1 | 0.35-0.50 | v4 完整 |

**通过条件**: v4 任一阶段 ARI 必须 > K-means × 1.5, 否则废弃。

---

## 6. 数据缺口 (阻塞 v4)

| 缺口 | 影响 | 解决 |
|---|---|---|
| **无客户 GPS** | H3 grid 推不出 | 需要送达方地址, 或从客户名+区推断中心点 |
| **19.2% 未识别 Zone** | H3 无法覆盖 | 地址清洗 + 地理编码 |
| **无 ETA 时间窗** | Stage 3 时间约束缺失 | 数据里有 卸货时间 (unload_time) 可推 |
| **客户类型字段** | SOP-3 类型共线无法验证 | 从客户名推断 (便利店/餐饮/超市) |

**关键**: 没有 GPS, H3 方向受阻。但**社区发现不需要 GPS** (只需共现图)。可以先做社区发现验证。

---

## 7. 防错流程 (agy 教训)

未来每一步都必须:
1. **先算 Bound**: 任何聚类粒度的 Recall 上限 (跨边界路线比例)
2. **必须对比 K-means baseline**: 提升 < 50% 直接废弃
3. **报分布不报 mean**: p25/p50/p75/p95 + 异常值检查
4. **怀疑自己结论**: 每 5 步停下来问 "这是真信号还是粒度假象?"
5. **用 NotebookLM + agy 交叉验证**: 不只信自己

---

## 8. 阶段状态

- v1/v2/v3: 均失败 (粒度错)
- v4: 方向对 (H3 + 社区发现), 但需要数据 (GPS) 才能完全验证
- **可以先做的 (不需 GPS)**: 客户共现图社区发现 (Louvain) → 验证 ARI 提升

**下一步选择**:
- A) 跑 Louvain 社区发现 baseline (不需 GPS, 直接验证图结构)
- B) 等用户提供地址/GPS 数据
- C) 再做一轮审计 (用户已要求过 4 次审计, 可能还要)

---

**审计方式**: 自审 + agy 1.0.8 + NotebookLM + 用户多轮纠正
**结论**: Zone 方向对, 但粒度仍错 (行政 Zone → H3 hex + 图社区)
**预期 v4 ARI**: 0.25-0.50 (vs 当前 0.07)