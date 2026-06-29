"""IO + Louvain hybrid: IO learns feature-weighted costs, Louvain finds communities.

IO alone (greedy merge) failed (ARI 0.162) because bias causes over-merging.
But IO's learned θ confirms PMI is dominant (θ=10.29).

This module combines:
  1. IO learns θ from historical pair features (PMI, PC, freq, zone, cap)
  2. Build graph with IO-weighted edges (not just PMI)
  3. Louvain community detection on IO-weighted graph
  4. Capacity + time-window post-processing
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from taihe_dc.data import Route
from taihe_dc.baselines.community_louvain import detect_communities
from taihe_dc.baselines.community_with_capacity import ROUTE_PC_CAP, SINGLE_CUSTOMER_PC_THRESHOLD
from taihe_dc.baselines.community_final import split_with_time_window
from taihe_dc.baselines.inverse_optimization import (
    PairFeatureExtractor,
    InverseOptimizationDispatcher,
    N_FEATURES,
    FEATURE_NAMES,
)
from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters
import networkx as nx


def build_io_weighted_graph(
    train_routes: list[Route],
    extractor: PairFeatureExtractor,
    theta: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    bias: float,
    min_weight: int = 2,
) -> nx.Graph:
    """Build co-occurrence graph with IO-learned edge weights.

    Edge weight = sigmoid(θ^T φ_normalized + bias) for each co-occurring pair.
    This replaces PMI-only weights with feature-aware weights.
    """
    import math

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    cust_count: dict[str, int] = defaultdict(int)

    for r in train_routes:
        cids = sorted(set(r.customer_ids))
        for c in cids:
            cust_count[c] += 1
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                pair_count[(cids[i], cids[j])] += 1

    G = nx.Graph()
    for c, n in cust_count.items():
        G.add_node(c, count=n)

    for (a, b), cnt in pair_count.items():
        if cnt < min_weight:
            continue
        # Extract features for this pair
        pc_a = extractor.avg_pc.get(a, 0.0)
        pc_b = extractor.avg_pc.get(b, 0.0)
        feat = extractor.extract(a, b, pc_a, pc_b)
        feat_norm = (feat - mean) / std
        z = float(np.dot(theta, feat_norm)) + bias
        prob = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
        # Use probability as edge weight (higher = stronger community tie)
        G.add_edge(a, b, weight=max(0.01, prob))

    return G


def run_io_louvain_hybrid(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    io_epochs: int = 30,
    min_weight: int = 2,
) -> "tuple[dict, dict, dict]":
    """IO + Louvain hybrid baseline.

    Step 1: Train IO to learn θ (feature weights)
    Step 2: Build graph with IO-weighted edges
    Step 3: Louvain community detection
    Step 4: Capacity + time-window post-processing
    """
    print("Step 1: Training IO to learn feature weights...")
    dispatcher = InverseOptimizationDispatcher.fit(train_routes, epochs=io_epochs)

    print("\nStep 2: Building IO-weighted graph...")
    G = build_io_weighted_graph(
        train_routes,
        dispatcher.extractor,
        dispatcher.theta,
        dispatcher._mean,
        dispatcher._std,
        dispatcher._bias,
        min_weight=min_weight,
    )
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nStep 3: Louvain community detection...")
    partition = detect_communities(G, resolution=1.0)
    print(f"  Communities: {len(set(partition.values()))}")

    print("\nStep 4: Capacity + time-window post-processing...")
    val_preds = split_with_time_window(partition, val_routes)
    test_preds = split_with_time_window(partition, test_routes)

    val_m = hard_mode_eval(val_routes, val_preds)
    test_m = hard_mode_eval(test_routes, test_preds)

    info = {
        "method": "IO-weighted Louvain",
        "theta": dispatcher.theta.tolist(),
        "feature_names": FEATURE_NAMES,
        "n_communities": len(set(partition.values())),
        "n_edges": G.number_of_edges(),
    }
    return val_m, test_m, info