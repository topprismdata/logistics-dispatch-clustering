"""Louvain + SOP-1 capacity post-processor.

Per design v4 Stage 3 + audit: Louvain communities may exceed vehicle capacity.
For each date, communities whose total PC > capacity must be split into
multiple sub-routes (greedy bin packing by PC).

Constraints:
  - SOP-1: customer PC > SINGLE_CUSTOMER_PC_THRESHOLD must be solo route
  - Total route PC <= ROUTE_PC_CAP (DEFAULT 3000, from 95th percentile)

Expected effect:
  - ARI stays roughly the same (splitting preserves most structure)
  - PC Overflow Rate → 0% (was 0.9% before)
  - avg_cluster_size drops toward real ~3
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from taihe_dc.data import Route
from taihe_dc.hard_mode import PredictedClusters
from taihe_dc.baselines.community_louvain import (
    build_cooccurrence_graph,
    detect_communities,
)


# Capacity thresholds (from audit EDA on 郑东 DC)
SINGLE_CUSTOMER_PC_THRESHOLD = 260.0  # SOP-1: customer > 260 PC → solo
ROUTE_PC_CAP = 3000.0                 # 95th percentile route PC


def _greedy_bin_pack(
    items: list[tuple[str, float]],
    cap: float,
) -> list[list[str]]:
    """First-fit-decreasing bin packing: sort by PC desc, fit into bins of cap.

    Returns list of bins, each bin is a list of customer_ids.
    """
    items_sorted = sorted(items, key=lambda x: -x[1])
    bins: list[list[tuple[str, float]]] = []  # each bin: [(cid, pc), ...]
    bin_loads: list[float] = []

    for cid, pc in items_sorted:
        # Try to fit in existing bin
        placed = False
        for i, load in enumerate(bin_loads):
            if load + pc <= cap:
                bins[i].append((cid, pc))
                bin_loads[i] += pc
                placed = True
                break
        if not placed:
            # New bin (even if single item exceeds cap — must be solo per SOP-1)
            bins.append([(cid, pc)])
            bin_loads.append(pc)
    return [[cid for cid, _ in bin] for bin in bins]


def split_overloaded_communities(
    partition: dict[str, int],
    routes: list[Route],
    route_pc_cap: float = ROUTE_PC_CAP,
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD,
) -> PredictedClusters:
    """For each date, group customers by community, then split communities
    that exceed route_pc_cap into multiple sub-routes (greedy bin packing).

    SOP-1: customers with PC > solo_threshold go to their own bin (solo).
    """
    # Build date → community → customers
    by_date_comm: dict[str, dict[int, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in routes:
        date_str = r.date.isoformat()
        for c in r.customer_ids:
            comm_id = partition.get(c)
            if comm_id is None:
                # Unseen customer — assign to a synthetic unique community
                comm_id = -(hash((date_str, c)) % (10**9))
            pc = r.pc_per_customer.get(c, 0.0)
            by_date_comm[date_str][comm_id].append((c, pc))

    # Build date → customer → cluster_id
    date_to_clusters: dict[str, dict[str, int]] = {}
    next_cluster_id = 0

    for date_str, comm_to_custs in by_date_comm.items():
        date_to_clusters[date_str] = {}
        for comm_id, custs in comm_to_custs.items():
            # Separate SOP-1 solo customers (PC > solo_threshold)
            solo = [(c, pc) for c, pc in custs if pc > solo_threshold]
            group = [(c, pc) for c, pc in custs if pc <= solo_threshold]

            # Each solo customer gets its own cluster
            for c, _ in solo:
                date_to_clusters[date_str][c] = next_cluster_id
                next_cluster_id += 1

            # Bin-pack the group respecting capacity
            if group:
                # If single customer, just one bin
                if len(group) == 1:
                    date_to_clusters[date_str][group[0][0]] = next_cluster_id
                    next_cluster_id += 1
                else:
                    total_pc = sum(pc for _, pc in group)
                    if total_pc <= route_pc_cap:
                        # No split needed — keep community intact
                        for c, _ in group:
                            date_to_clusters[date_str][c] = next_cluster_id
                        next_cluster_id += 1
                    else:
                        # Split into bins
                        bins = _greedy_bin_pack(group, route_pc_cap)
                        for bin_cids in bins:
                            for c in bin_cids:
                                date_to_clusters[date_str][c] = next_cluster_id
                            next_cluster_id += 1

    return PredictedClusters(date_to_clusters=date_to_clusters)


def run_louvain_with_capacity(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    min_weight: int = 2,
    route_pc_cap: float = ROUTE_PC_CAP,
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD,
) -> "tuple[dict, dict, dict]":
    """Louvain + SOP-1 capacity split. End-to-end.

    Returns: (val_metrics, test_metrics, info)
    """
    G = build_cooccurrence_graph(train_routes, min_weight=min_weight, use_pmi=True)
    partition = detect_communities(G, resolution=1.0)

    val_preds = split_overloaded_communities(partition, val_routes, route_pc_cap, solo_threshold)
    test_preds = split_overloaded_communities(partition, test_routes, route_pc_cap, solo_threshold)

    from taihe_dc.hard_mode import hard_mode_eval
    val_m = hard_mode_eval(val_routes, val_preds)
    test_m = hard_mode_eval(test_routes, test_preds)

    info = {
        "n_communities": len(set(partition.values())),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "route_pc_cap": route_pc_cap,
        "solo_threshold": solo_threshold,
    }
    return val_m, test_m, info