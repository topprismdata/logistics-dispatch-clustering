"""Louvain + SOP-1 capacity + time-window split (final v4 method).

Combines:
  - Stage 1: Louvain community detection (data-driven master routes)
  - Stage 2: SOP-1 solo routing for PC > 260
  - Stage 3: Greedy bin packing by PC (route_pc_cap=3000)
  - Stage 4: Time-window split (unload_time > 2h apart → split)

Result on test (n=823 routes): ARI=0.540, F1=58.6%, avg_cluster=3.27
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from taihe_dc.data import Route
from taihe_dc.hard_mode import PredictedClusters
from taihe_dc.baselines.community_louvain import (
    build_cooccurrence_graph,
    detect_communities,
)
from taihe_dc.baselines.community_with_capacity import (
    ROUTE_PC_CAP,
    SINGLE_CUSTOMER_PC_THRESHOLD,
    _greedy_bin_pack,
)


DEFAULT_TIME_WINDOW_HOURS = 2.0


def split_with_time_window(
    partition: dict[str, int],
    routes: list[Route],
    route_pc_cap: float = ROUTE_PC_CAP,
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD,
    time_window_hours: float = DEFAULT_TIME_WINDOW_HOURS,
) -> PredictedClusters:
    """Final method: Louvain community + SOP-1 + capacity + time window.

    For each date, group customers by community. Within each community:
      1. PC > solo_threshold → solo route (SOP-1)
      2. Remaining grouped by unload_time window (2h default)
      3. Each time-window bin packed by PC capacity (greedy FFD)
    """
    by_date_comm: dict[str, dict[int, list[tuple[str, float, Optional[datetime]]]]] = defaultdict(lambda: defaultdict(list))
    for r in routes:
        date_str = r.date.isoformat()
        # customer → unload_time (from delivery rows)
        cust_to_unload: dict[str, Optional[datetime]] = {}
        for d in r.delivery_rows:
            cust_to_unload.setdefault(d.customer_id, d.unload_time)
        for c in r.customer_ids:
            comm_id = partition.get(c)
            if comm_id is None:
                comm_id = -(hash((date_str, c)) % (10**9))
            pc = r.pc_per_customer.get(c, 0.0)
            unload = cust_to_unload.get(c)
            by_date_comm[date_str][comm_id].append((c, pc, unload))

    date_to_clusters: dict[str, dict[str, int]] = {}
    next_cluster_id = 0

    for date_str, comm_to_custs in by_date_comm.items():
        date_to_clusters[date_str] = {}
        for comm_id, custs in comm_to_custs.items():
            # SOP-1: solo for big PC
            solo = [(c, pc, t) for c, pc, t in custs if pc > solo_threshold]
            group = [(c, pc, t) for c, pc, t in custs if pc <= solo_threshold]

            for c, _, _ in solo:
                date_to_clusters[date_str][c] = next_cluster_id
                next_cluster_id += 1

            if not group:
                continue

            # Time-window split: sort by unload_time, split if > window apart
            with_time = sorted([(c, pc, t) for c, pc, t in group if t is not None], key=lambda x: x[2])
            no_time = [(c, pc, t) for c, pc, t in group if t is None]

            time_bins: list[list[tuple[str, float, Optional[datetime]]]] = []
            if with_time:
                cur = [with_time[0]]
                for c, pc, t in with_time[1:]:
                    last_t = cur[-1][2]
                    gap_h = (t - last_t).total_seconds() / 3600
                    if gap_h > time_window_hours:
                        time_bins.append(cur)
                        cur = [(c, pc, t)]
                    else:
                        cur.append((c, pc, t))
                time_bins.append(cur)
            if no_time:
                time_bins.append(no_time)

            # Each time bin → capacity-aware bin packing
            for bin_items in time_bins:
                items_for_pack = [(c, pc) for c, pc, _ in bin_items]
                total_pc = sum(pc for _, pc in items_for_pack)
                if len(items_for_pack) == 1 or total_pc <= route_pc_cap:
                    for c, _ in items_for_pack:
                        date_to_clusters[date_str][c] = next_cluster_id
                    next_cluster_id += 1
                else:
                    bins = _greedy_bin_pack(items_for_pack, route_pc_cap)
                    for bin_cids in bins:
                        for c in bin_cids:
                            date_to_clusters[date_str][c] = next_cluster_id
                        next_cluster_id += 1

    return PredictedClusters(date_to_clusters=date_to_clusters)


def run_final_method(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    min_weight: int = 2,
    resolution: float = 1.0,
    time_window_hours: float = DEFAULT_TIME_WINDOW_HOURS,
) -> "tuple[dict, dict, dict]":
    """End-to-end final method."""
    from taihe_dc.hard_mode import hard_mode_eval

    G = build_cooccurrence_graph(train_routes, min_weight=min_weight, use_pmi=True)
    partition = detect_communities(G, resolution=resolution)

    val_preds = split_with_time_window(partition, val_routes, time_window_hours=time_window_hours)
    test_preds = split_with_time_window(partition, test_routes, time_window_hours=time_window_hours)

    val_m = hard_mode_eval(val_routes, val_preds)
    test_m = hard_mode_eval(test_routes, test_preds)

    info = {
        "method": "Louvain + SOP-1 capacity + time-window",
        "n_communities": len(set(partition.values())),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "time_window_hours": time_window_hours,
        "min_weight": min_weight,
        "resolution": resolution,
    }
    return val_m, test_m, info