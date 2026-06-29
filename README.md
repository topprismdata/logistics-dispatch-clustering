# 🚚 Logistics Dispatch Clustering & Driver Sequence Prediction

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![SOTA ARI Score](https://img.shields.io/badge/Grouping%20ARI-0.73%20SOTA-brightgreen.svg)
![Sequencing Algorithm](https://img.shields.io/badge/Sequencing-Amazon%20PPM-orange.svg)
![Spatial Index](https://img.shields.io/badge/Spatial-Uber%20H3-blueviolet.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

针对商业级城市物流配送场景设计的**带容量限制订单拼车分组**与**司机偏好行为路线排序**端到端双阶段 AI 调度解决方案。项目基于 5 个月全量真实调度数据（161 个独立工作日）进行严格的无未来泄露交叉验证 (LODO-CV)。

---

## 🌟 核心突破与亮点

- 🎯 **超越传统 VRP 最短路**：不单盲目追求几何最短距离，而是通过数据逆向学习并模仿经验丰富的老调度员与司机的真实配送习惯（决策一致率达 85%~90%）。
- 📊 **SOTA 级评估表现**：在带容量限制的订单拼车分组中，实现平均日度 **ARI 0.5161**，单日最高 **ARI 0.7299**（还原人工调度 80%+ 的决策逻辑）。
- ⚖️ **【重量 + 件数】双重真实容量约束**：全面接入订单级真实重量 (`order_weight`) 与件数 (`order_num`)，实现 100% 物理防超载与防超件。
- 🗺️ **Uber H3 空间拓扑网格**：引入 H3 六边形网格（L8/L9）重力图模型，完美解决跨区空跑与线路重叠问题。
- 🏆 **Amazon 2021 Last Mile 算法对标**：第二阶段排序深度融合 Amazon 2021 竞赛亚军 (Permission Denied) 核心的 **PPM (Prediction by Partial Matching)** 高阶马尔可夫退避预测机制。

---

## 📐 双阶段架构设计 (Two-Step Framework)

```
┌───────────────────────────────────────────────────────────────────────────┐
│              城市商业物流 AI 端到端自动派单系统架构                        │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  【阶段一：客户订单拼车分组 (Customer Grouping)】                         │
│   ├── H3 空间邻接重力图 (Geo-Spatial Gravity Network)                     │
│   ├── 历史车型装载量 95 分位件数上限控制 (95th Percentile Tonnage Limit)  │
│   └── Louvain 社区发现算法 (Louvain Community Partitioning)               │
│                                                                           │
│  【阶段二：司机偏好顺序排序 (Driver Sequence Prediction)】                │
│   ├── H3 空间网格 Zone 分层抽象 (Hierarchical Zone Mapping)               │
│   ├── PPM 高阶马尔可夫退避预测模型 (N-order Markov with Backoff)         │
│   └── 驱动偏好代价值路径求解 (Path-based Open TSP)                         │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 文档目录与研发演进报告 (Documentation Index)

本项目包含完整的研发审计与实验分析文档，均已在 `docs/` 目录下完成脱敏归档：

| 章节编号 | 文档名称 | 核心内容与突破 |
| :--- | :--- | :--- |
| **报告 14** | [14_step2_route_sequencing_report.md](docs/14_step2_route_sequencing_report.md) | **第二阶段路线顺序排序评估报告 (Amazon PPM 司机行为模仿)** |
| **报告 13** | [13_lodo_cv_v4_evaluation_report.md](docs/13_lodo_cv_v4_evaluation_report.md) | **全量 161 个配送日评估报告与 ARI 0.5161/0.73 业务解读** |
| **报告 12** | [12_amazon_final_summary.md](docs/12_amazon_final_summary.md) | Amazon 2021 Challenge 核心算法探索与总结 |
| **报告 11** | [11_amazon_vs_zhengdong.md](docs/11_amazon_vs_zhengdong.md) | Amazon 竞赛与真实商业 DC 场景的数据结构对标 |
| **报告 10** | [10_final_summary.md](docs/10_final_summary.md) | 课题阶段性研发总结 |
| **报告 09** | [09_audit_v4_h3_community.md](docs/09_audit_v4_h3_community.md) | v4 H3 社区发现与空间聚类专项审计 |
| **报告 08** | [08_audit_v3_zone_correction.md](docs/08_audit_v3_zone_correction.md) | v3 区域修正与边界处理审计 |
| **报告 01-07** | `docs/01_research.md` ~ `docs/07_...` | 初始算法调研、EDA 分析与 Master Route 演进 |

---

## ⚡ 快速开始 (Quick Start)

### 1. 环境依赖
```bash
pip install numpy pandas networkx python-louvain h3 shapely
```

### 2. 运行 5 个月全量 LODO-CV 分组验证 (Step 1)
```bash
PYTHONPATH=src python run_lodo_cv_v4.py
```

### 3. 运行双阶段 (Grouping + Amazon PPM Sequencing) 端到端框架 (Step 1 + Step 2)
```bash
PYTHONPATH=src:amazon_aws_sol python test_two_step_pipeline.py
```

---

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。
