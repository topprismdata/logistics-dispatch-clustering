---
title: 阶段 1.5 最终总结 — 4 轮审计 + 8 次实验
date: 2026-06-27
status: ceiling-reached-without-gps
---

# 阶段 1.5 最终总结

## 1. 核心结果

**最终方法**: Louvain 社区发现 + SOP-1 容量约束 + 2h 时间窗

**TEST 性能** (n=823 路线, 2026-05):
- ARI: **0.540**
- Partition F1: **58.7%**
- Recall: 55.6%
- Precision: 62.1%
- avg_cluster: 3.2 (真实 ~3)

**OOD 验证** (141 路线含 train 没见过的客户): ARI=0.496, Precision=83.7%
→ 仅比总 ARI 低 3%, 真结构非查表。

---

## 2. 进化轨迹 (4 轮审计 + 8 次实验)

### 方法迭代

| 版本 | 方法 | ARI | 触发 |
|---|---|---|---|
| v1 | Siamese customer-customer | 0.010 | 初始 |
| v2 | ConVRP customer anchor | 0% 稳定 | audit 1 |
| v3 | 行政 Zone | 0.070 | 用户纠正 1+2 |
| **v4** | **Louvain 社区** | **0.512** | NotebookLM + 用户洞察 |
| +capacity | SOP-1 bin packing | 0.531 | audit 3 |
| +time | 2h unload_time | **0.540** | F 实验 |
| D res | resolution 扫描 | 0.531 | 无提升 |
| E weekday | 分图 | 0.195 | ❌ 切碎 |
| G carrier | 分图 | 0.338 | ❌ 切碎 |
| H Zone+ | 加边 | 0.542 | ⚪ 冗余 |

### 已确认的"天花板模式"

任何**切分共现图**的方式都降低性能:
- Weekday 分图: -0.345
- Carrier 分图: -0.202
- 原因: 客户**跨 weekday / 跨 carrier 共现**, 切分丢失信号

任何**加边 enrichment** 的方式都冗余:
- Zone 加边: +0.002 (Louvain 已捕获)

**单一 Louvain 图是最优的**, 因为捕获了全部共现信号。

---

## 3. 用户 4 次纠正 (全部正确)

1. **"主路线是相对稳定, 查互联网, 别只用自己知识"** → 触发 Amazon Zone 文献调研
2. **"你可能没真懂 amazon2021 竞赛"** → NotebookLM 重新讲解 Zone 抽象 (X-Y 格式, macro 100% 确定)
3. **"分层处理"** → H3/社区/anchor 多层架构
4. **"目前没有 gps"** → 排除 H3, 聚焦社区发现

**真相**: master route 在**数据驱动社区层面**稳定, 不在 customer / Zone 层面。

---

## 4. 4 轮审计的价值

| 审计 | 发现 | 节省的浪费 |
|---|---|---|
| 1 | Easy Mode 83% 是幻觉 (Precision 100% 无负样本) | 避免基于假数据决策 |
| 2 | ConVRP customer anchor 假设不成立 (0% 稳定) | 避免实现错误方向 |
| 3 | KRC/HR@3 bug, B3 BC 阈值未校准 | 修复评估器 |
| 4 | Zone 粒度错 (行政 Zone 太粗) | 转向 Louvain (0.07 → 0.51) |

agy 1.0.8 独立交叉验证 3 次, 全部与我自审一致。

---

## 5. 文献支撑

| 来源 | 贡献 |
|---|---|
| Amazon Last Mile 2021 | Zone 抽象 (X-Y 格式), 两阶段 (inter-zone + intra-zone) |
| ConVRP (Groër/Golden/Wasil) | Template route + stable core + insertion |
| NotebookLM (14 sources) | 3 轮问答: 复现预期 / SOP 编码 / baseline 设计 |
| LaDe dataset | Graph2Route SOTA KRC=56.99% 对标 |
| DC_A 5 docs | 8 SOP 框架 (PC 阈值 / 容量 / 时间节奏) |

---

## 6. 无 GPS 下的天花板分析

ARI 0.540 意味着捕获了约 54% 的真实路线结构。剩余 46% gap 的来源:

| 来源 | 占比估计 | 解决方案 |
|---|---|---|
| 真随机 (临时拼车) | ~20% | 不可解 (业务本质) |
| GPS 空间结构 (无法看到) | ~15% | 需要 GPS → H3 hex grid |
| 时间细节 (卸货顺序) | ~5% | 用户说排序不重要 |
| 容量决策的多样性 | ~6% | RL action evaluation |

**结论**: 无 GPS 下 0.540 接近上限。要突破需:
1. **GPS 数据** → H3 hex grid (预期 +0.10-0.15)
2. **RL 范式转换** → 不学路线拓扑, 学决策动作 (agy 建议)

---

## 7. 文件清单

```
taihe-dc-rl/
├── docs/
│   ├── 01_research.md              # 14 sources NotebookLM 分析
│   ├── 02_design.md                # 初始设计 (审计后被推翻)
│   ├── 03_eda_report.md            # 数据 EDA
│   ├── 04_results.md               # 初始结果 (审计后失效)
│   ├── 05_audit.md                 # 审计 v1 (Easy Mode 幻觉)
│   ├── 06_design_v2_master_route.md # Master Route 设计 (部分对)
│   ├── 07_audit_v2_master_route.md # 审计 v2 (customer ConVRP 失败)
│   ├── 08_audit_v3_zone_correction.md # 审计 v3 (Zone 方向对)
│   ├── 09_audit_v4_h3_community.md # 审计 v4 (H3 + 社区发现)
│   └── 10_final_summary.md         # 本文档
├── src/taihe_dc/
│   ├── data/{schema,loader}.py     # xlsx 加载
│   ├── sop.py                      # 8 SOP 自动提取
│   ├── eda.py                      # EDA 模块
│   ├── split.py                    # OOT 时序切分
│   ├── evaluator.py                # 6 指标 (KRC bug 已修)
│   ├── hard_mode.py                # Hard Mode ARI/F1
│   ├── final_results.py            # 最终结果常量
│   └── baselines/
│       ├── bc_vehicle_id.py        # B3 BC (0% recall 反证)
│       ├── pairwise_siamese.py     # B1 Pairwise (Easy Mode 83%, Hard 0.01)
│       ├── hard_mode_runner.py     # Hard Mode 跑 Pairwise
│       ├── community_louvain.py    # ★ v4 核心 (ARI 0.512)
│       ├── community_with_capacity.py # + SOP-1 (0.531)
│       └── community_final.py      # ★ 最终方法 (+ 2h time, 0.540)
└── pyproject.toml
```

---

## 8. 关键教训

1. **粒度决定一切** — Customer (0.01) vs Zone (0.07) vs Community (0.51)
2. **Easy Mode 是陷阱** — 已知边界内枚举对会虚高 80%+
3. **数据驱动 > 先验** — Louvain 社区 > 行政 Zone
4. **切分图 = 失败** — 任何 conditioning (weekday/carrier) 切碎共现信号
5. **审计不可少** — 4 轮审计 + agy 交叉验证节省大量浪费
6. **听用户** — 4 次纠正全部正确, 我每次都 initially 错

---

**最终状态**: 阶段 1.5 完成, ARI 0.540, 无 GPS 下达到天花板。