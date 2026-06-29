"""Hard Mode runner: Pairwise Siamese → per-date clustering → ARI/F1.

This is the REAL dispatch evaluation (audit R3). Given ALL customers on each
test date, the Siamese model predicts pair probabilities, then we cluster
(greedy union-find) into routes, then compare against true route assignments.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations
from typing import Optional

import torch

from taihe_dc.data import Route, DispatchDataset
from taihe_dc.baselines.pairwise_siamese import SiamesePairNet
from taihe_dc.hard_mode import (
    PredictedClusters,
    hard_mode_eval,
    format_hard_mode,
    greedy_cluster_from_pairs,
)


def predict_clusters_hard_mode(
    model: SiamesePairNet,
    routes: list[Route],
    cust2idx: dict[str, int],
    threshold: float = 0.5,
    batch_size: int = 4096,
) -> PredictedClusters:
    """For each date, predict clusters from Siamese pair probabilities.

    For each date:
      1. Gather all unique customers that day
      2. For every pair (O(N²)), compute P(same-route)
      3. Greedy union-find clusters pairs with P >= threshold
    """
    model.eval()
    by_date: dict[str, set[str]] = defaultdict(set)
    for r in routes:
        for c in r.customer_ids:
            by_date[r.date.isoformat()].add(c)

    date_to_clusters: dict[str, dict[str, int]] = {}

    with torch.no_grad():
        for date_str, customers_set in by_date.items():
            customers = sorted(customers_set)
            # Filter to known customers (in train vocab)
            known = [c for c in customers if c in cust2idx]
            if len(known) < 2:
                # Trivial: each customer is its own cluster
                date_to_clusters[date_str] = {c: i for i, c in enumerate(customers)}
                continue

            # All pairs among known customers
            pairs = list(combinations(known, 2))
            if not pairs:
                date_to_clusters[date_str] = {c: i for i, c in enumerate(customers)}
                continue

            # Batch predict
            probs: dict[tuple[str, str], float] = {}
            for batch_start in range(0, len(pairs), batch_size):
                batch = pairs[batch_start:batch_start + batch_size]
                c1_idx = torch.tensor([cust2idx[a] for a, _ in batch], dtype=torch.long)
                c2_idx = torch.tensor([cust2idx[b] for _, b in batch], dtype=torch.long)
                extra = torch.zeros(len(batch), 1)
                preds = model(c1_idx, c2_idx, extra).cpu().numpy()
                for (a, b), p in zip(batch, preds):
                    probs[(a, b)] = float(p)

            # Cluster (union-find on pairs >= threshold)
            cluster_map = greedy_cluster_from_pairs(customers, probs, threshold=threshold)
            date_to_clusters[date_str] = cluster_map

    return PredictedClusters(date_to_clusters=date_to_clusters)


def run_hard_mode_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    epochs: int = 15,
    threshold: float = 0.5,
) -> "tuple[HardModeMetrics, HardModeMetrics, dict]":
    """Train Pairwise on train, predict clusters on val/test (Hard Mode).

    Returns: (val_metrics, test_metrics, trained_model_info)
    """
    from taihe_dc.baselines.pairwise_siamese import train_pairwise

    model, cust2idx = train_pairwise(train_routes, val_routes, epochs=epochs)

    val_preds = predict_clusters_hard_mode(model, val_routes, cust2idx, threshold=threshold)
    test_preds = predict_clusters_hard_mode(model, test_routes, cust2idx, threshold=threshold)

    val_m = hard_mode_eval(val_routes, val_preds)
    test_m = hard_mode_eval(test_routes, test_preds)
    return val_m, test_m, {"model": model, "cust2idx": cust2idx}