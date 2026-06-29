"""Community detection baseline (v4 Stage 1).

Per audit v4 + agy recommendation: instead of administrative Zone (too coarse)
or customer-customer pair prediction (sparse), use Louvain community detection
on the customer co-occurrence graph.

This finds "naturally tight" customer groups — the real master route templates.

Does NOT need GPS (unlike H3 hex grid), only needs historical co-occurrence.

Usage:
    from taihe_dc.baselines.community_louvain import run_community_baseline
    val_m, test_m = run_community_baseline(train_routes, val_routes, test_routes)
"""

from __future__ import annotations

from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
from community import best_partition  # python-louvain

from taihe_dc.data import Route
from taihe_dc.hard_mode import (
    PredictedClusters,
    hard_mode_eval,
    format_hard_mode,
)


def build_cooccurrence_graph(
    routes: list[Route],
    min_weight: int = 2,
    use_pmi: bool = True,
) -> nx.Graph:
    """Build customer co-occurrence graph from training routes.

    Edge weight:
      - raw count if use_pmi=False
      - PMI-normalized if use_pmi=True (better for sparse data)
    """
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    cust_count: Counter = Counter()
    n_routes = len(routes)

    for r in routes:
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
        if use_pmi:
            # Pointwise Mutual Information: log(P(a,b) / (P(a) * P(b)))
            # Higher = stronger than chance co-occurrence
            p_a = cust_count[a] / n_routes
            p_b = cust_count[b] / n_routes
            p_ab = cnt / n_routes
            if p_a * p_b > 0 and p_ab > 0:
                import math
                pmi = math.log(p_ab / (p_a * p_b))
                # Shift to non-negative for Louvain (which needs positive weights)
                weight = max(0.01, pmi + 5)  # shift so min weight is 0.01
            else:
                weight = 0.01
        else:
            weight = float(cnt)
        if weight > 0:
            G.add_edge(a, b, weight=weight)

    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run Louvain community detection. Returns customer → community_id."""
    if G.number_of_nodes() == 0:
        return {}
    # Louvain
    partition = best_partition(G, resolution=resolution, random_state=42)
    return partition


def predict_clusters_community(
    routes: list[Route],
    partition: dict[str, int],
    fallback_strategy: str = "singleton",
) -> PredictedClusters:
    """Assign test customers to clusters using the trained partition.

    Customers seen in train: use their community_id
    Customers unseen in train: fallback (singleton or nearest by name zone)
    """
    by_date: dict[str, dict[str, int]] = defaultdict(dict)
    next_id = max(partition.values(), default=-1) + 1

    for r in routes:
        date_str = r.date.isoformat()
        for c in r.customer_ids:
            if c in partition:
                by_date[date_str][c] = partition[c]
            else:
                # Unseen customer — assign to its own cluster
                if fallback_strategy == "singleton":
                    by_date[date_str][c] = next_id
                    next_id += 1
                else:
                    by_date[date_str][c] = -1  # unknown

    return PredictedClusters(date_to_clusters=dict(by_date))


def run_community_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    min_weight: int = 2,
    resolution: float = 1.0,
) -> "tuple[dict, dict, dict]":
    """Train community detection on train, predict on val/test.

    Returns: (val_metrics, test_metrics, partition_info)
    """
    G = build_cooccurrence_graph(train_routes, min_weight=min_weight, use_pmi=True)
    partition = detect_communities(G, resolution=resolution)

    n_communities = len(set(partition.values()))
    info = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities": n_communities,
        "avg_community_size": G.number_of_nodes() / max(1, n_communities),
        "partition": partition,
    }

    val_preds = predict_clusters_community(val_routes, partition)
    test_preds = predict_clusters_community(test_routes, partition)

    val_m = hard_mode_eval(val_routes, val_preds)
    test_m = hard_mode_eval(test_routes, test_preds)
    return val_m, test_m, info


def random_baseline(routes: list[Route], k_clusters: int, seed: int = 42) -> PredictedClusters:
    """Random clustering baseline — assigns each customer to a random cluster.
    Useful as the floor (agy Baseline Shield).
    """
    import random
    rng = random.Random(seed)
    by_date: dict[str, dict[str, int]] = defaultdict(dict)
    for r in routes:
        date_str = r.date.isoformat()
        for c in r.customer_ids:
            by_date[date_str][c] = rng.randint(0, k_clusters - 1)
    return PredictedClusters(date_to_clusters=dict(by_date))


def singleton_baseline(routes: list[Route]) -> PredictedClusters:
    """Each customer in its own cluster (worst possible recall)."""
    by_date: dict[str, dict[str, int]] = defaultdict(dict)
    for r in routes:
        date_str = r.date.isoformat()
        for i, c in enumerate(r.customer_ids):
            by_date[date_str][c] = i
    return PredictedClusters(date_to_clusters=dict(by_date))