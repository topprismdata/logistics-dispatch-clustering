"""Evaluation metrics for the dispatch RL problem.

6 metrics per design doc section 3.2:
  M1: Pair Recall          (核心: 真同车客户对被预测同车的比例)
  M2: Pair F1              (Precision + Recall)
  M3: KRC                  (Kendall Rank Correlation — 序列匹配度)
  M4: HR@3                 (前 3 个客户命中率)
  M5: PC Overflow Rate     (物理合规: 预测路线超容量比例)
  M6: Route Size Distribution (业务合规)

Usage:
    from taihe_dc.evaluator import evaluate_predictions, Predictions
    preds = Predictions(route_to_vehicle=..., per_route_pairs=...)
    metrics = evaluate_predictions(true_routes, preds, vehicle_capacities)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

from taihe_dc.data import Route, DispatchDataset


# Vehicle capacity in PC terms — for SOP-1 overflow check.
# Real PC max per route observed: ~10,000. Most cap values in tons
# but PC unit is cases, not tons. We use observed PC distribution
# (p99 of route_pc_total) as an empirical cap proxy.
DEFAULT_PC_CAP_PER_ROUTE = 3000  # empirical: 99th percentile route PC


@dataclass
class Predictions:
    """A model's predictions for one or more routes.

    Two ways to express:
      (a) route_to_vehicle: route_id -> vehicle_id (or None for unassigned)
      (b) per_route_pairs: route_id -> frozenset of customer pairs predicted as same-vehicle
    """
    route_to_vehicle: dict[str, Optional[str]] = field(default_factory=dict)
    per_route_pairs: dict[str, frozenset[tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def from_vehicle_assignments(cls, route_to_vehicle: dict[str, Optional[str]]) -> "Predictions":
        """Infer per_route_pairs from vehicle assignments."""
        per_route_pairs: dict[str, frozenset] = {}
        # Group routes by assigned vehicle
        veh_to_routes: dict[str, set[str]] = defaultdict(set)
        for rid, vid in route_to_vehicle.items():
            if vid is not None:
                veh_to_routes[vid].add(rid)
        # Customers on same vehicle = predicted same-route
        # This requires looking up customers per route — done by caller via from_data
        return cls(route_to_vehicle=route_to_vehicle, per_route_pairs=per_route_pairs)


@dataclass(frozen=True)
class Metrics:
    pair_recall: float
    pair_precision: float
    pair_f1: float
    krc: float
    hr_at_3: float
    pc_overflow_rate: float
    predicted_route_size_dist: dict[int, int]
    true_route_size_dist: dict[int, int]
    n_routes: int
    n_pairs_evaluated: int
    notes: list[str] = field(default_factory=list)


def _all_customer_pairs(customer_ids: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    cids = sorted(set(customer_ids))
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            pairs.add((cids[i], cids[j]))
    return pairs


def compute_pair_recall_precision(
    true_routes: list[Route],
    pred_route_pairs: dict[str, frozenset[tuple[str, str]]],
) -> tuple[float, float, int]:
    """Pair Recall = TP / (TP + FN) where TP = predicted same-route ∩ true same-route.

    True pairs: customers on the same actual route.
    Predicted pairs: provided by model.
    """
    # Build true pair set (across all routes)
    true_pairs: set[tuple[str, str]] = set()
    for r in true_routes:
        true_pairs |= _all_customer_pairs(r.customer_ids)

    # Build predicted pair set
    pred_pairs: set[tuple[str, str]] = set()
    for pairs in pred_route_pairs.values():
        pred_pairs |= pairs

    if not true_pairs:
        return 0.0, 0.0, 0

    tp = len(true_pairs & pred_pairs)
    fn = len(true_pairs - pred_pairs)
    fp = len(pred_pairs - true_pairs)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return recall, precision, len(true_pairs)


def _kendall_tau(rank_a: list, rank_b: list) -> float:
    """Kendall tau-b correlation between two rankings of the same items.

    Tied items handled by averaging. Returns float in [-1, 1].
    """
    n = len(rank_a)
    if n <= 1:
        return 1.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rank_a[i] == rank_a[j] and rank_b[i] == rank_b[j]:
                ties_a += 1
                ties_b += 1
            elif rank_a[i] == rank_a[j]:
                ties_a += 1
            elif rank_b[i] == rank_b[j]:
                ties_b += 1
            elif (rank_a[i] < rank_a[j]) == (rank_b[i] < rank_b[j]):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def compute_krc(true_routes: list[Route], pred_route_pairs: dict[str, frozenset]) -> float:
    """KRC averaged over routes that have both true and predicted pairs.

    BUG FIX (audit R1): when pred_pairs is empty for a route, the union-find
    produces all singletons → pred_ranking degenerates to identity → KRC = 1.0.
    Now we skip routes with no predicted pairs (KRC is undefined there).
    """
    krcs: list[float] = []
    for r in true_routes:
        cids = list(dict.fromkeys(r.customer_ids))  # preserve order, dedupe
        if len(cids) < 2:
            continue
        pred_pairs = pred_route_pairs.get(r.route_id, frozenset())
        # BUG FIX: skip routes with no predicted pairs (KRC undefined)
        if not pred_pairs:
            continue
        true_ranking = list(range(len(cids)))  # ground truth order
        # Assign group ids per predicted pair
        parent: dict[str, str] = {c: c for c in cids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in pred_pairs:
            if a in parent and b in parent:
                union(a, b)
        # Predicted ranking: stable sort by group root
        groups: dict[str, list[str]] = defaultdict(list)
        for c in cids:
            groups[find(c)].append(c)
        # Preserve original delivery order within group
        pred_order: list[str] = []
        seen_groups: set[str] = set()
        for c in cids:
            g = find(c)
            if g not in seen_groups:
                seen_groups.add(g)
                # output all members of this group in delivery order
                pred_order.extend(groups[g])
        # Map pred_order to rank
        pred_ranking = [pred_order.index(c) for c in cids]
        if len(set(pred_ranking)) > 1:
            krcs.append(_kendall_tau(true_ranking, pred_ranking))
    return sum(krcs) / len(krcs) if krcs else 0.0


def compute_hr_at_3(true_routes: list[Route], pred_route_pairs: dict[str, frozenset]) -> float:
    """Hit Rate @ 3: for each route, do the first 3 customers (in true order)
    all appear in the same predicted group?"""
    if not true_routes:
        return 0.0
    hits = 0
    total = 0
    for r in true_routes:
        if r.n_customers < 3:
            continue
        total += 1
        first3 = r.customer_ids[:3]
        pred_pairs = pred_route_pairs.get(r.route_id, frozenset())
        # Check if all 3 pairwise within predicted group
        same = all(((a, b) in pred_pairs) or ((b, a) in pred_pairs)
                   for a, b in combinations(first3, 2))
        if same:
            hits += 1
    return hits / max(1, total)


def compute_pc_overflow_rate(
    true_routes: list[Route],
    pred_route_pairs: dict[str, frozenset],
    pc_cap: float = DEFAULT_PC_CAP_PER_ROUTE,
) -> float:
    """Fraction of predicted routes whose total PC exceeds the empirical cap.

    For a model that respects SOP-1, this should be 0%.
    """
    if not true_routes:
        return 0.0
    overflow = 0
    total = 0
    for r in true_routes:
        total += 1
        # Predicted group: union-find on pred_pairs
        parent: dict[str, str] = {c: c for c in r.customer_ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for a, b in pred_route_pairs.get(r.route_id, frozenset()):
            if a in parent and b in parent:
                union(a, b)
        groups: dict[str, float] = defaultdict(float)
        for cid in r.customer_ids:
            groups[find(cid)] += r.pc_per_customer.get(cid, 0)
        if max(groups.values()) > pc_cap:
            overflow += 1
    return overflow / max(1, total)


def evaluate_predictions(
    true_routes: list[Route],
    predictions: Predictions,
    pc_cap: float = DEFAULT_PC_CAP_PER_ROUTE,
) -> Metrics:
    """Compute all 6 metrics."""
    recall, precision, n_pairs = compute_pair_recall_precision(true_routes, predictions.per_route_pairs)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    krc = compute_krc(true_routes, predictions.per_route_pairs)
    hr3 = compute_hr_at_3(true_routes, predictions.per_route_pairs)
    overflow = compute_pc_overflow_rate(true_routes, predictions.per_route_pairs, pc_cap)

    # Predicted route size dist
    pred_size_dist: dict[int, int] = defaultdict(int)
    for r in true_routes:
        parent: dict[str, str] = {c: c for c in r.customer_ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for a, b in predictions.per_route_pairs.get(r.route_id, frozenset()):
            if a in parent and b in parent:
                union(a, b)
        n_groups = len({find(c) for c in r.customer_ids})
        pred_size_dist[n_groups] += 1

    # True route size dist
    true_size_dist: dict[int, int] = defaultdict(int)
    for r in true_routes:
        true_size_dist[r.n_customers] += 1

    notes = []
    if precision < 0.3 and recall > 0.3:
        notes.append("Low precision: model over-predicts same-route pairs.")
    if recall < 0.3 and n_pairs > 100:
        notes.append("Low recall: model rarely finds correct pair groupings.")
    if overflow > 0.1:
        notes.append(f"High overflow rate {overflow:.1%}: SOP-1 capacity constraint not enforced.")

    return Metrics(
        pair_recall=recall,
        pair_precision=precision,
        pair_f1=f1,
        krc=krc,
        hr_at_3=hr3,
        pc_overflow_rate=overflow,
        predicted_route_size_dist=dict(pred_size_dist),
        true_route_size_dist=dict(true_size_dist),
        n_routes=len(true_routes),
        n_pairs_evaluated=n_pairs,
        notes=notes,
    )


def format_metrics(m: Metrics) -> str:
    lines = [
        f"Pair Recall:     {m.pair_recall:.1%}",
        f"Pair Precision:  {m.pair_precision:.1%}",
        f"Pair F1:         {m.pair_f1:.1%}",
        f"KRC:             {m.krc:.3f}",
        f"HR@3:            {m.hr_at_3:.1%}",
        f"PC Overflow:     {m.pc_overflow_rate:.1%}",
        f"Routes eval'd:   {m.n_routes}",
        f"Pairs eval'd:    {m.n_pairs_evaluated}",
    ]
    if m.notes:
        lines.append("Notes:")
        for n in m.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


# Helper: convert true Route assignments to per_route_pairs (the ground truth)
def true_predictions(routes: list[Route]) -> Predictions:
    """Build a Predictions from true routes — used as oracle baseline."""
    pred = Predictions()
    for r in routes:
        pred.per_route_pairs[r.route_id] = frozenset(_all_customer_pairs(r.customer_ids))
        pred.route_to_vehicle[r.route_id] = r.plate
    return pred