"""Hard Mode evaluation: cross-route clustering task.

This is the REAL dispatch problem (vs. the 'Easy Mode' that gave inflated
83.2% Pair Recall). Per audit R3:

  Easy Mode: given a route, predict P(cust A, cust B) for intra-route pairs
             → 83.2% recall but mostly lookup + no cross-route negatives
  Hard Mode: given ALL customers on a date, predict groupings (clusters)
             → real-world task, O(N²) pairs including negatives

Hard Mode metrics:
  - ARI (Adjusted Rand Index): clustering similarity [-1, 1], chance = 0
  - Partition F1: pair-level precision/recall across the full partition
  - Pair Recall (Hard): TP / (TP + FN) computed over ALL pairs that day,
                         not just within true routes

Usage:
    from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters
    preds = model.predict_clusters(per_date_customers)
    metrics = hard_mode_eval(true_routes, preds)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from collections import defaultdict
from typing import Callable

from taihe_dc.data import Route


@dataclass
class PredictedClusters:
    """Model output: customer_id -> predicted cluster_id, per date."""
    # date -> { customer_id -> cluster_id }
    date_to_clusters: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class HardModeMetrics:
    ari: float                         # Adjusted Rand Index (per-date averaged)
    partition_recall: float            # TP / (TP + FN) over ALL pairs
    partition_precision: float         # TP / (TP + FP) over ALL pairs
    partition_f1: float
    n_dates: int
    n_customers_total: int
    n_true_routes_total: int
    n_pred_clusters_total: int
    avg_cluster_size: float
    notes: list[str] = field(default_factory=list)


def _build_true_partition(routes: list[Route]) -> dict[str, dict[str, int]]:
    """Group routes by date, assign each customer a true cluster id (= route_id index)."""
    by_date: dict[str, list[Route]] = defaultdict(list)
    for r in routes:
        by_date[r.date.isoformat()].append(r)

    true_partition: dict[str, dict[str, int]] = {}
    for date_str, day_routes in by_date.items():
        cid_to_cluster: dict[str, int] = {}
        for cluster_idx, r in enumerate(day_routes):
            for c in set(r.customer_ids):
                # A customer could appear on multiple routes same day — keep first
                if c not in cid_to_cluster:
                    cid_to_cluster[c] = cluster_idx
        true_partition[date_str] = cid_to_cluster
    return true_partition


def _adjusted_rand_index(true_labels: list[int], pred_labels: list[int]) -> float:
    """Compute ARI between two clusterings.

    ARI = (RI - E[RI]) / (max(RI) - E[RI])
    Range: [-1, 1], where 1 = perfect, 0 = random, <0 = worse than random.
    """
    from math import comb
    n = len(true_labels)
    if n < 2:
        return 1.0  # trivial

    # Build contingency table
    table: dict[tuple[int, int], int] = defaultdict(int)
    a_counts: dict[int, int] = defaultdict(int)
    b_counts: dict[int, int] = defaultdict(int)
    for a, b in zip(true_labels, pred_labels):
        table[(a, b)] += 1
        a_counts[a] += 1
        b_counts[b] += 1

    sum_comb_a = sum(comb(c, 2) for c in a_counts.values())
    sum_comb_b = sum(comb(c, 2) for c in b_counts.values())
    sum_comb_table = sum(comb(c, 2) for c in table.values())
    total_comb = comb(n, 2)

    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0  # perfect or trivial
    return (sum_comb_table - expected) / (max_index - expected)


def _pair_metrics(true_labels: list[int], pred_labels: list[int]) -> tuple[float, float]:
    """Pair-level precision/recall across ALL pairs."""
    n = len(true_labels)
    if n < 2:
        return 1.0, 1.0

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_same = true_labels[i] == true_labels[j]
            pred_same = pred_labels[i] == pred_labels[j]
            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return recall, precision


def hard_mode_eval(
    true_routes: list[Route],
    predictions: PredictedClusters,
) -> HardModeMetrics:
    """Evaluate clustering predictions against true routes per date."""
    true_partition = _build_true_partition(true_routes)

    aris: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    n_customers_total = 0
    n_true_routes_total = 0
    n_pred_clusters_total = 0
    cluster_sizes: list[int] = []

    for date_str, true_cid_to_cluster in true_partition.items():
        if date_str not in predictions.date_to_clusters:
            continue

        pred_cid_to_cluster = predictions.date_to_clusters[date_str]
        # Common customers (only evaluate those that appear in BOTH)
        common = sorted(set(true_cid_to_cluster) & set(pred_cid_to_cluster))
        if len(common) < 2:
            continue

        true_labels = [true_cid_to_cluster[c] for c in common]
        pred_labels = [pred_cid_to_cluster[c] for c in common]

        aris.append(_adjusted_rand_index(true_labels, pred_labels))
        rec, prec = _pair_metrics(true_labels, pred_labels)
        recalls.append(rec)
        precisions.append(prec)

        n_customers_total += len(common)
        n_true_routes_total += len(set(true_labels))
        n_pred_clusters_total += len(set(pred_labels))
        for cluster_id in set(pred_labels):
            cluster_sizes.append(sum(1 for c in pred_labels if c == cluster_id))

    n_dates = len(aris)
    if n_dates == 0:
        return HardModeMetrics(
            ari=0.0, partition_recall=0.0, partition_precision=0.0, partition_f1=0.0,
            n_dates=0, n_customers_total=0, n_true_routes_total=0, n_pred_clusters_total=0,
            avg_cluster_size=0.0,
            notes=["No dates with both true and predicted partitions."],
        )

    avg_ari = sum(aris) / n_dates
    avg_rec = sum(recalls) / n_dates
    avg_prec = sum(precisions) / n_dates
    avg_f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0.0
    avg_cluster_size = sum(cluster_sizes) / max(1, len(cluster_sizes))

    notes = []
    if avg_rec < 0.3:
        notes.append(f"Low partition recall ({avg_rec:.1%}): model misses many same-route pairs")
    if avg_prec < 0.3:
        notes.append(f"Low partition precision ({avg_prec:.1%}): model predicts too many false pairs")
    if avg_ari < 0.1:
        notes.append(f"ARI near chance ({avg_ari:.3f}): clustering ≈ random")

    return HardModeMetrics(
        ari=avg_ari,
        partition_recall=avg_rec,
        partition_precision=avg_prec,
        partition_f1=avg_f1,
        n_dates=n_dates,
        n_customers_total=n_customers_total,
        n_true_routes_total=n_true_routes_total,
        n_pred_clusters_total=n_pred_clusters_total,
        avg_cluster_size=avg_cluster_size,
        notes=notes,
    )


def format_hard_mode(m: HardModeMetrics) -> str:
    lines = [
        f"ARI (Adjusted Rand Index): {m.ari:.3f}   (1=perfect, 0=random, <0=worse)",
        f"Partition Recall:          {m.partition_recall:.1%}   (TP / all true same-route pairs)",
        f"Partition Precision:       {m.partition_precision:.1%}   (TP / all predicted same-route pairs)",
        f"Partition F1:              {m.partition_f1:.1%}",
        f"Dates evaluated:           {m.n_dates}",
        f"Customers total:           {m.n_customers_total:,}",
        f"True routes total:         {m.n_true_routes_total:,}",
        f"Pred clusters total:       {m.n_pred_clusters_total:,}",
        f"Avg predicted cluster size: {m.avg_cluster_size:.1f}",
    ]
    if m.notes:
        lines.append("Notes:")
        for n in m.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


# Helper: greedy clustering from pair probabilities
def greedy_cluster_from_pairs(
    customers: list[str],
    pair_probs: dict[tuple[str, str], float],
    threshold: float = 0.5,
) -> dict[str, int]:
    """Build clusters by union-find on pairs above threshold.

    Simple O(E α(N)) greedy. For better clustering use spectral/agglomerative,
    but greedy is a reasonable baseline for the Siamese similarity signal.
    """
    parent: dict[str, str] = {c: c for c in customers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Sort pairs by probability desc, union top ones
    sorted_pairs = sorted(pair_probs.items(), key=lambda x: -x[1])
    for (a, b), p in sorted_pairs:
        if p >= threshold:
            union(a, b)

    # Assign cluster ids
    cid_to_cluster: dict[str, int] = {}
    next_id = 0
    for c in customers:
        root = find(c)
        if root not in {v for v in cid_to_cluster.values()}:
            # Find if any customer already mapped to this root
            existing = None
            for cc, rr in [(cc, find(cc)) for cc in cid_to_cluster]:
                if rr == root:
                    existing = cid_to_cluster[cc]
                    break
            if existing is not None:
                cid_to_cluster[c] = existing
            else:
                cid_to_cluster[c] = next_id
                next_id += 1
        else:
            # find the existing cluster for this root
            for cc, rr in [(cc, find(cc)) for cc in cid_to_cluster]:
                if rr == root:
                    cid_to_cluster[c] = cid_to_cluster[cc]
                    break
    return cid_to_cluster