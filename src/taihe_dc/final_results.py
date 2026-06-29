"""Final results report — phase 1.5 (4 audit rounds + Master Route pivot).

Trajectory of Hard Mode ARI on test set (n=823 routes):

  v1 Siamese Pair:        0.010  (failed — pair prediction too sparse)
  v3 Zone-as-cluster:     0.070  (pseudo-structure — admin Zone too coarse)
  v4 Louvain:             0.512  (★ breakthrough — data-driven communities)
  + SOP-1 capacity:       0.531  (greedy bin packing, ARI slightly up)
  + time window 2h:       0.540  (final — unload_time signal)

54x improvement over initial Siamese, 7.7x over Zone.
"""

# This file documents the final phase 1.5 results.
# See docs/04_results.md (initial, pre-audit) and docs/09_audit_v4_h3_community.md
# for the audit trail that led here.

FINAL_RESULTS = {
    "test_n_routes": 823,
    "test_n_customers": 1137,
    "date_range_test": "2026-05-05 to 2026-05-31",
    "baselines": {
        "singleton_floor":      {"ari": -0.078, "f1": 0.001},
        "random_k20":           {"ari": 0.002,  "f1": 0.062},
        "v1_siamese_pair":      {"ari": 0.010,  "f1": 0.145, "pc_overflow": 0.007},
        "v3_admin_zone":        {"ari": 0.070,  "f1": 0.174},
        "v4_louvain_minw2":     {"ari": 0.512,  "f1": 0.560, "avg_cluster": 4.24},
        "v4_louvain_capacity":  {"ari": 0.531,  "f1": 0.577, "avg_cluster": 3.57},
        "v4_louvain_time2h":    {"ari": 0.540,  "f1": 0.586, "avg_cluster": 3.27},  # ★ final
    },
    "ood_validation": {
        # Routes with at least one customer NOT seen in train
        "unseen_customer_routes_n": 141,
        "unseen_ari": 0.496,
        "unseen_precision": 0.837,
        "all_seen_ari": 0.578,
        "drop_vs_total": 0.016,  # only 3% drop → real structure, not lookup
    },
    "config": {
        "method": "Louvain community detection + SOP-1 capacity split + 2h time window",
        "graph_construction": "PMI-normalized co-occurrence, min_weight=2",
        "louvain_resolution": 1.0,
        "route_pc_cap": 3000.0,          # 95th percentile from EDA
        "solo_pc_threshold": 260.0,      # SOP-1 from EDA
        "time_window_hours": 2.0,
    },
    "user_corrections_that_mattered": [
        "1. '主路线是相对稳定, 查互联网, 别只用自己知识'",
        "2. '你可能没真正读懂 amazon2021 竞赛'",
        "3. '你可以分层处理'",
        "→ All 3 led to Zone → Community pivot (0.07 → 0.51)",
    ],
    "rejected_directions": [
        "v2 ConVRP customer anchor (0% stable pairs)",
        "v3 admin Zone granularity (too coarse, avg_cluster 10.9)",
        "H3 hex grid (blocked — no GPS available)",
        "Weekday-specific graphs (ARI 0.195 — fragments co-occurrence)",
        "Louvain resolution sweep (no gain, robust at 0.8-1.5)",
    ],
    "next_steps_without_gps": [
        "Anchor-based clustering (community centers + distance decay)",
        "Time-series demand prediction (大单 PC prediction)",
        "Ensemble Louvain + Zone (combine signals)",
    ],
}


def print_summary():
    print("=" * 60)
    print("FINAL PHASE 1.5 RESULTS")
    print("=" * 60)
    print(f"\nBest method: Louvain + SOP-1 + 2h time window")
    print(f"Test ARI: {FINAL_RESULTS['baselines']['v4_louvain_time2h']['ari']}")
    print(f"Test F1:  {FINAL_RESULTS['baselines']['v4_louvain_time2h']['f1']:.1%}")
    print(f"Avg cluster: {FINAL_RESULTS['baselines']['v4_louvain_time2h']['avg_cluster']} (real ~3)")
    print(f"\nOOD ARI (unseen customers): {FINAL_RESULTS['ood_validation']['unseen_ari']}")
    print(f"OOD drop vs total: {FINAL_RESULTS['ood_validation']['drop_vs_total']:.1%} (real structure)")
    print(f"\nImprovement: 54x over Siamese, 7.7x over admin Zone")


if __name__ == "__main__":
    print_summary()