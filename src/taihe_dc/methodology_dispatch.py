"""Methodology-driven dispatch: learn dispatcher's decision process.

Inspired by Amazon 2021 methodology:
  1. Hierarchical decomposition (not flat clustering)
  2. Learn implicit decision rules (not just co-occurrence)
  3. Constrained optimization with learned priors

Key difference from pure Louvain:
  Louvain: static community detection (who groups with whom historically)
  This: dynamic decision simulation (how dispatcher groups TODAY's customers)

Three-layer decision process (mirrors human dispatcher):
  Layer 1: SOP-1 triage — PC > threshold → solo (physical constraint)
  Layer 2: Community grouping — Louvain prior + today's PC adjustments
  Layer 3: Residual assignment — remaining customers by capacity fit

The CRITICAL insight from Amazon: learn the DISPATCHER'S LOGIC, not just outcomes.
We extract: preferred group sizes, PC utilization patterns, community stability.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from taihe_dc.data import Route
from taihe_dc.baselines.community_louvain import build_cooccurrence_graph, detect_communities
from taihe_dc.baselines.community_with_capacity import (
    ROUTE_PC_CAP,
    SINGLE_CUSTOMER_PC_THRESHOLD,
    _greedy_bin_pack,
)
from taihe_dc.hard_mode import PredictedClusters, hard_mode_eval


@dataclass
class DispatcherProfile:
    """Learned profile of human dispatcher's decision patterns.

    Extracted from historical routes — this IS the 'implicit rules' that
    Amazon methodology teaches us to quantify.
    """

    # Layer 1: SOP-1 triage threshold (PC above → solo)
    solo_pc_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD

    # Layer 2: Community structure (Louvain)
    partition: dict[str, int] = field(default_factory=dict)

    # Layer 3: Dispatcher preferences (learned from history)
    preferred_group_sizes: dict[int, float] = field(default_factory=dict)  # size → frequency
    preferred_pc_utilization: dict[str, float] = field(default_factory=dict)  # stats
    avg_route_pc: float = 0.0
    median_route_pc: float = 0.0
    p25_route_pc: float = 0.0
    p75_route_pc: float = 0.0

    # Customer features (for dynamic grouping)
    cust_avg_pc: dict[str, float] = field(default_factory=dict)
    cust_freq: dict[str, int] = field(default_factory=dict)
    cust_zone: dict[str, str] = field(default_factory=dict)

    # Community coherence: how often do same-community customers actually group?
    community_coherence: dict[int, float] = field(default_factory=dict)

    n_communities: int = 0


def extract_dispatcher_profile(train_routes: list[Route]) -> DispatcherProfile:
    """Extract the human dispatcher's decision patterns from historical routes.

    This is the 'learn from humans' step — we quantify:
    - What PC threshold triggers solo routing?
    - What group sizes does the dispatcher prefer?
    - What PC utilization patterns are common?
    - How coherent are Louvain communities in practice?
    """
    # Build co-occurrence graph + Louvain
    G = build_cooccurrence_graph(train_routes, min_weight=2, use_pmi=True)
    partition = detect_communities(G, resolution=1.0)

    # Route statistics (dispatcher's output patterns)
    route_pcs = sorted(r.route_pc_total for r in train_routes)
    route_sizes = sorted(r.n_customers for r in train_routes)

    # Preferred group sizes (what sizes does dispatcher produce most?)
    size_freq = Counter(r.n_customers for r in train_routes)
    total_routes = len(train_routes)
    preferred_sizes = {k: v / total_routes for k, v in size_freq.items()}

    # Customer features
    cust_pc_sum = defaultdict(float)
    cust_count = defaultdict(int)
    for r in train_routes:
        for c in r.customer_ids:
            cust_count[c] += 1
            cust_pc_sum[c] += r.pc_per_customer.get(c, 0.0)
    cust_avg_pc = {c: cust_pc_sum[c] / max(1, cust_count[c]) for c in cust_pc_sum}

    # Community coherence: for each community, how often do its members
    # actually appear on the same route vs different routes?
    comm_members = defaultdict(set)
    for c, comm_id in partition.items():
        comm_members[comm_id].add(c)

    comm_coherence = {}
    for comm_id, members in comm_members.items():
        if len(members) < 2:
            continue
        # Count how many route-pairs are same-route vs total pairs
        member_routes = defaultdict(set)
        for r in train_routes:
            for c in set(r.customer_ids) & members:
                member_routes[c].add(r.route_id)

        same_route = 0
        total_pairs = 0
        member_list = list(members)
        for i in range(len(member_list)):
            for j in range(i + 1, len(member_list)):
                r1 = member_routes.get(member_list[i], set())
                r2 = member_routes.get(member_list[j], set())
                total_pairs += 1
                if r1 & r2:
                    same_route += 1
        comm_coherence[comm_id] = same_route / max(1, total_pairs)

    # Zone extraction
    import re
    ZONE_PATTERNS = [
        "郑东新区", "高新区", "经开区", "航空港区", "金水区", "二七区",
        "中原区", "管城回族区", "管城区", "惠济区", "上街区",
        "中牟县", "CBD", "龙子湖", "龙湖", "白沙",
    ]
    def extract_zone(name):
        if not name: return "?"
        for pat in ZONE_PATTERNS:
            if pat in name: return pat
        if "郑州" in name: return "郑州其他"
        return "?"

    cust_zone = {}
    for r in train_routes:
        for d in r.delivery_rows:
            if d.customer_id not in cust_zone:
                cust_zone[d.customer_id] = extract_zone(d.customer_name)

    n = len(route_pcs)
    profile = DispatcherProfile(
        partition=partition,
        preferred_group_sizes=preferred_sizes,
        avg_route_pc=sum(route_pcs) / n,
        median_route_pc=route_pcs[n // 2],
        p25_route_pc=route_pcs[n // 4],
        p75_route_pc=route_pcs[3 * n // 4],
        cust_avg_pc=cust_avg_pc,
        cust_freq=dict(cust_count),
        cust_zone=cust_zone,
        community_coherence=comm_coherence,
        n_communities=len(set(partition.values())),
    )

    # Print learned profile
    print("═" * 60)
    print("  Dispatcher Profile (learned from human decisions)")
    print("═" * 60)
    print(f"\nLayer 1 — SOP-1 Triage:")
    print(f"  Solo threshold: PC > {profile.solo_pc_threshold:.0f}")

    print(f"\nLayer 2 — Community Structure:")
    print(f"  Communities: {profile.n_communities}")
    coherent = sum(1 for v in comm_coherence.values() if v > 0.3)
    print(f"  Coherent communities (>30% same-route rate): {coherent}/{len(comm_coherence)}")

    print(f"\nLayer 3 — Dispatcher Preferences:")
    print(f"  Route PC: median={profile.median_route_pc:.0f}, IQR=[{profile.p25_route_pc:.0f}, {profile.p75_route_pc:.0f}]")
    top_sizes = sorted(preferred_sizes.items(), key=lambda x: -x[1])[:5]
    print(f"  Top group sizes: {', '.join(f'{k}cust({v:.0%})' for k, v in top_sizes)}")

    return profile


def dispatch_with_profile(
    profile: DispatcherProfile,
    target_date,
    customer_pcs: dict[str, float],
) -> PredictedClusters:
    """Dispatch using learned profile — simulates human decision process.

    Layer 1: PC > threshold → solo
    Layer 2: Community grouping + PC-aware refinement
    Layer 3: Residual capacity-aware assignment
    """
    date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)

    # Layer 1: SOP-1 triage
    solo = {c: pc for c, pc in customer_pcs.items() if pc > profile.solo_pc_threshold}
    remaining = {c: pc for c, pc in customer_pcs.items() if pc <= profile.solo_pc_threshold}

    # Layer 2: Community grouping with PC-aware refinement
    # Group by community, then check if group is "good" based on learned preferences
    comm_groups = defaultdict(list)
    for c, pc in remaining.items():
        comm_id = profile.partition.get(c)
        if comm_id is None:
            comm_id = -(hash((date_str, c)) % (10**9))
        comm_groups[comm_id].append((c, pc))

    cluster_map = {}
    next_id = 0

    # Solo customers
    for c in solo:
        cluster_map[c] = next_id
        next_id += 1

    # Layer 2 + 3: For each community group, decide final routing
    for comm_id, members in comm_groups.items():
        members.sort(key=lambda x: -x[1])  # sort by PC desc

        # Check coherence: is this community "stable"?
        coherence = profile.community_coherence.get(comm_id, 0.0)

        if coherence > 0.3 and len(members) <= 5:
            # Stable community, small group → keep together if capacity allows
            total_pc = sum(pc for _, pc in members)
            if total_pc <= ROUTE_PC_CAP:
                for c, _ in members:
                    cluster_map[c] = next_id
                next_id += 1
            else:
                # Split by capacity
                bins = _greedy_bin_pack(members, ROUTE_PC_CAP)
                for bin_cids in bins:
                    for c in bin_cids:
                        cluster_map[c] = next_id
                    next_id += 1
        else:
            # Unstable community or large group → capacity-driven split
            # Use learned preferred group sizes to guide splitting
            total_pc = sum(pc for _, pc in members)
            if total_pc <= ROUTE_PC_CAP and len(members) <= 6:
                # Small enough → keep together
                for c, _ in members:
                    cluster_map[c] = next_id
                next_id += 1
            else:
                # Split: aim for preferred group sizes (median ~3)
                bins = _greedy_bin_pack(members, ROUTE_PC_CAP)
                for bin_cids in bins:
                    for c in bin_cids:
                        cluster_map[c] = next_id
                    next_id += 1

    return PredictedClusters(date_to_clusters={date_str: cluster_map})


def run_methodology_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
) -> "tuple[dict, dict, DispatcherProfile]":
    """End-to-end: learn dispatcher profile → plan → evaluate."""
    profile = extract_dispatcher_profile(train_routes)

    # Evaluate
    from taihe_dc.hard_mode import hard_mode_eval

    val_preds_list = []
    test_preds_list = []
    val_date_clusters = {}
    test_date_clusters = {}

    for r in val_routes:
        ds = r.date.isoformat()
        if ds not in val_date_clusters:
            cust_pcs = {}
            for rr in val_routes:
                if rr.date.isoformat() == ds:
                    for c in rr.customer_ids:
                        cust_pcs[c] = rr.pc_per_customer.get(c, 0.0)
            preds = dispatch_with_profile(profile, r.date, cust_pcs)
            val_date_clusters.update(preds.date_to_clusters)

    for r in test_routes:
        ds = r.date.isoformat()
        if ds not in test_date_clusters:
            cust_pcs = {}
            for rr in test_routes:
                if rr.date.isoformat() == ds:
                    for c in rr.customer_ids:
                        cust_pcs[c] = rr.pc_per_customer.get(c, 0.0)
            preds = dispatch_with_profile(profile, r.date, cust_pcs)
            test_date_clusters.update(preds.date_to_clusters)

    val_m = hard_mode_eval(val_routes, PredictedClusters(date_to_clusters=val_date_clusters))
    test_m = hard_mode_eval(test_routes, PredictedClusters(date_to_clusters=test_date_clusters))

    return val_m, test_m, profile