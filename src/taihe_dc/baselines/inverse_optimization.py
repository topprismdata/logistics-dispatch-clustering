"""Inverse Optimization for capacity-constrained dispatch.

Based on Amazon 2021 Last Mile Routing Challenge 2nd place ("Permission Denied")
paper: arXiv:2307.07357 "Inverse Optimization for Routing Problems" (TU Delft).

Core idea: learn the cost function C_θ that makes human dispatcher's historical
route groupings appear optimal, then use C_θ to plan new days.

Mathematical formulation:
  Forward:  y* = argmin_y C_θ(X, y)  s.t. capacity constraints
  Inverse:  learn θ from historical (X, y*) pairs
  Loss:     L(θ) = Σ_i [C_θ(X_i, y_i*) - min_y C_θ(X_i, y)]

For 郑东 DC (capacity-constrained GROUPING, no ordering):
  - Feature function φ(i,j) for customer pair
  - Cost C_θ(group) = Σ_{i,j in group} θ^T φ(i,j)
  - Hard constraints: PC > solo_threshold → solo; group PC ≤ cap

Simplified IO (no full bilevel optimization):
  - Learn θ via structured perceptron / logistic regression on pair features
  - Forward solver: greedy clustering with learned costs + capacity
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from taihe_dc.data import Route, DispatchDataset
from taihe_dc.baselines.community_with_capacity import (
    ROUTE_PC_CAP,
    SINGLE_CUSTOMER_PC_THRESHOLD,
    _greedy_bin_pack,
)
from taihe_dc.hard_mode import PredictedClusters, hard_mode_eval, format_hard_mode


# Feature dimensionality for pair cost
# φ(i,j) = [PMI_cooccur, pc_compat, freq_product, same_zone, capacity_fit]
N_FEATURES = 5
FEATURE_NAMES = ["PMI共现", "PC兼容性", "频次乘积", "同Zone", "容量适配"]


@dataclass
class PairFeatureExtractor:
    """Extract features for customer pairs from training data.

    Features (all pre-computed from train, applied to any pair):
      0. PMI co-occurrence: log P(i,j) / (P(i) * P(j))
      1. PC compatibility: 1 / (1 + |avg_pc_i - avg_pc_j| / 100)
      2. Frequency product: log(1 + freq_i) * log(1 + freq_j) / 100
      3. Same zone: 1.0 if same zone, 0.0 otherwise
      4. Capacity fit: 1 - max(0, (pc_i + pc_j) / cap) — closer to 0 = tighter fit
    """

    pmi: dict[tuple[str, str], float] = field(default_factory=dict)
    avg_pc: dict[str, float] = field(default_factory=dict)
    freq: dict[str, int] = field(default_factory=dict)
    zone: dict[str, str] = field(default_factory=dict)
    n_routes: int = 0

    @classmethod
    def fit(cls, train_routes: list[Route]) -> "PairFeatureExtractor":
        """Pre-compute all pair features from training routes."""
        # Co-occurrence counts
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        cust_count: dict[str, int] = defaultdict(int)
        cust_pc_sum: dict[str, float] = defaultdict(float)
        n_routes = len(train_routes)

        for r in train_routes:
            cids = sorted(set(r.customer_ids))
            for c in cids:
                cust_count[c] += 1
                pc = r.pc_per_customer.get(c, 0.0)
                cust_pc_sum[c] += pc
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pair_count[(cids[i], cids[j])] += 1

        # PMI
        pmi: dict[tuple[str, str], float] = {}
        for (a, b), cnt in pair_count.items():
            p_a = cust_count[a] / n_routes
            p_b = cust_count[b] / n_routes
            p_ab = cnt / n_routes
            if p_a > 0 and p_b > 0 and p_ab > 0:
                pmi[(a, b)] = math.log(p_ab / (p_a * p_b))
            else:
                pmi[(a, b)] = 0.0

        avg_pc = {c: cust_pc_sum[c] / max(1, cust_count[c]) for c in cust_pc_sum}

        # Zone extraction (from customer names — simple heuristic)
        import re
        ZONE_PATTERNS = [
            "郑东新区", "高新区", "经开区", "航空港区", "金水区", "二七区",
            "中原区", "管城回族区", "管城区", "惠济区", "上街区",
            "中牟县", "巩义", "荥阳", "新密", "新郑", "登封",
            "CBD", "龙子湖", "龙湖", "白沙",
        ]

        def extract_zone(name: str) -> str:
            if not name:
                return "?"
            for pat in ZONE_PATTERNS:
                if pat in name:
                    return pat
            if "郑州" in name:
                return "郑州其他"
            return "?"

        zone: dict[str, str] = {}
        # Need customer names from delivery rows
        for r in train_routes:
            for d in r.delivery_rows:
                if d.customer_id not in zone:
                    zone[d.customer_id] = extract_zone(d.customer_name)

        return cls(
            pmi=pmi,
            avg_pc=avg_pc,
            freq=dict(cust_count),
            zone=zone,
            n_routes=n_routes,
        )

    def extract(self, c1: str, c2: str, pc1: float, pc2: float) -> np.ndarray:
        """Extract 5-dim feature vector for a customer pair."""
        a, b = sorted([c1, c2])

        # 0. PMI co-occurrence
        pmi_val = self.pmi.get((a, b), -2.0)  # default: low (never seen together)

        # 1. PC compatibility
        pc_compat = 1.0 / (1.0 + abs(self.avg_pc.get(a, pc1) - self.avg_pc.get(b, pc2)) / 100.0)

        # 2. Frequency product
        f1 = math.log(1 + self.freq.get(a, 0))
        f2 = math.log(1 + self.freq.get(b, 0))
        freq_prod = f1 * f2 / 10.0

        # 3. Same zone
        z1 = self.zone.get(a, "?")
        z2 = self.zone.get(b, "?")
        same_zone = 1.0 if z1 == z2 and z1 != "?" else 0.0

        # 4. Capacity fit (for this specific day's PC)
        cap_fit = 1.0 - min(1.0, (pc1 + pc2) / ROUTE_PC_CAP)

        return np.array([pmi_val, pc_compat, freq_prod, same_zone, cap_fit])


@dataclass
class InverseOptimizationDispatcher:
    """Learn cost weights θ via IO, then dispatch with learned costs.

    Training:
      1. For each historical route, compute pair features for same-route pairs
         (positive) and sampled different-route pairs (negative)
      2. Learn θ via logistic regression (same-route P = sigmoid(θ^T φ))
      3. θ captures which features make humans group customers together

    Inference:
      1. For new day's customers, compute all pair costs using learned θ
      2. Greedy clustering: merge pairs with high P(same-route) respecting capacity
      3. SOP-1 hard constraint: PC > threshold → solo
    """

    extractor: PairFeatureExtractor = field(default_factory=PairFeatureExtractor)
    theta: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    route_pc_cap: float = ROUTE_PC_CAP
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD
    merge_threshold: float = 0.5  # P(same-route) threshold for merging

    @classmethod
    def fit(
        cls,
        train_routes: list[Route],
        epochs: int = 50,
        lr: float = 0.01,
        neg_ratio: float = 2.0,
    ) -> "InverseOptimizationDispatcher":
        """Train IO: learn θ from historical route groupings."""
        import random
        rng = random.Random(42)

        extractor = PairFeatureExtractor.fit(train_routes)

        # Build training pairs: positive (same route) + negative (different route)
        pos_pairs: list[tuple[str, str, float, float]] = []  # (c1, c2, pc1, pc2)
        neg_pairs: list[tuple[str, str, float, float]] = []

        # Customer → routes (for negative sampling)
        cust_routes: dict[str, set[str]] = defaultdict(set)
        for r in train_routes:
            for c in set(r.customer_ids):
                cust_routes[c].add(r.route_id)

        for r in train_routes:
            cids = list(set(r.customer_ids))
            # Positive: all same-route pairs
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pc_i = r.pc_per_customer.get(cids[i], 0.0)
                    pc_j = r.pc_per_customer.get(cids[j], 0.0)
                    pos_pairs.append((cids[i], cids[j], pc_i, pc_j))

            # Negative: sample different-route pairs
            for c1 in cids:
                n_neg = int(neg_ratio)
                for _ in range(n_neg):
                    c2 = rng.choice(list(cust_routes.keys()))
                    if c2 == c1:
                        continue
                    if cust_routes[c1] & cust_routes[c2]:
                        continue  # they do co-occur, skip
                    pc1 = extractor.avg_pc.get(c1, 0.0)
                    pc2 = extractor.avg_pc.get(c2, 0.0)
                    neg_pairs.append((c1, c2, pc1, pc2))

        print(f"  IO training pairs: {len(pos_pairs):,} positive, {len(neg_pairs):,} negative")

        # Extract features
        all_pairs = pos_pairs + neg_pairs
        labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.float32)

        X = np.array([
            extractor.extract(c1, c2, pc1, pc2)
            for c1, c2, pc1, pc2 in all_pairs
        ], dtype=np.float32)

        # Normalize features
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mean) / std

        # Logistic regression via SGD (simplified IO)
        theta = np.zeros(N_FEATURES, dtype=np.float32)
        bias = 0.0
        n = len(labels)

        for epoch in range(epochs):
            # Shuffle
            perm = rng.sample(range(n), n)
            total_loss = 0.0
            for idx in perm:
                x = X_norm[idx]
                y = labels[idx]
                z = np.dot(theta, x) + bias
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
                grad = (p - y) * x
                theta -= lr * grad
                bias -= lr * (p - y)
                total_loss += -(y * math.log(max(1e-8, p)) + (1 - y) * math.log(max(1e-8, 1 - p)))
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n
                acc = sum(1 for i in range(min(1000, n)) if (1 / (1 + math.exp(-np.dot(theta, X_norm[i]) - bias))) > 0.5 == labels[i]) / min(1000, n)
                print(f"    epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        # Store learned parameters
        dispatcher = cls(
            extractor=extractor,
            theta=theta,
        )
        dispatcher._mean = mean
        dispatcher._std = std
        dispatcher._bias = bias

        # Print learned feature importance
        print(f"\n  Learned cost weights θ (higher = stronger 'merge' signal):")
        for i, name in enumerate(FEATURE_NAMES):
            print(f"    {name:12s}: θ={theta[i]:+.4f}")
        print(f"    bias:        {bias:+.4f}")

        return dispatcher

    def pair_cost(self, c1: str, c2: str, pc1: float, pc2: float) -> float:
        """P(same-route) for a customer pair. Higher = should merge."""
        feat = self.extractor.extract(c1, c2, pc1, pc2)
        feat_norm = (feat - self._mean) / self._std
        z = np.dot(self.theta, feat_norm) + self._bias
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))

    def plan_day(self, target_date, customer_pcs: dict[str, float]) -> PredictedClusters:
        """Plan vehicles for one day using learned costs.

        Algorithm:
          1. SOP-1: PC > solo_threshold → solo
          2. For remaining customers, build complete graph with pair costs
          3. Greedy merge: repeatedly merge highest-cost pair if capacity allows
          4. Result = capacity-compliant clustering
        """
        # Separate SOP-1 solo customers
        solo = [(c, pc) for c, pc in customer_pcs.items() if pc > self.solo_threshold]
        group_custs = [(c, pc) for c, pc in customer_pcs.items() if pc <= self.solo_threshold]

        # Union-find for clustering
        parent: dict[str, str] = {c: c for c, _ in group_custs}
        group_pc: dict[str, float] = {c: pc for c, pc in group_custs}
        group_members: dict[str, list[str]] = {c: [c] for c, _ in group_custs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Compute all pair costs, sort descending
        pair_costs: list[tuple[str, str, float]] = []
        for i in range(len(group_custs)):
            for j in range(i + 1, len(group_custs)):
                c1, pc1 = group_custs[i]
                c2, pc2 = group_custs[j]
                cost = self.pair_cost(c1, c2, pc1, pc2)
                pair_costs.append((c1, c2, cost))
        pair_costs.sort(key=lambda x: -x[2])

        # Greedy merge: merge highest-cost pairs if capacity allows
        for c1, c2, cost in pair_costs:
            if cost < self.merge_threshold:
                break
            r1, r2 = find(c1), find(c2)
            if r1 == r2:
                continue
            merged_pc = group_pc[r1] + group_pc[r2]
            if merged_pc <= self.route_pc_cap:
                parent[r2] = r1
                group_pc[r1] = merged_pc
                group_members[r1].extend(group_members[r2])
                group_members[r2] = []

        # Build cluster assignments
        date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
        cluster_map: dict[str, int] = {}
        next_id = 0

        # Solo customers → each their own cluster
        for c, _ in solo:
            cluster_map[c] = next_id
            next_id += 1

        # Grouped customers → assign by union-find root
        root_to_cluster: dict[str, int] = {}
        for c, _ in group_custs:
            root = find(c)
            if root not in root_to_cluster:
                root_to_cluster[root] = next_id
                next_id += 1
            cluster_map[c] = root_to_cluster[root]

        return PredictedClusters(date_to_clusters={date_str: cluster_map})

    def evaluate_against_human(self, human_routes: list[Route]) -> HardModeMetrics:
        """Evaluate IO dispatcher on historical routes."""
        human_by_date: dict[str, list[Route]] = defaultdict(list)
        for r in human_routes:
            human_by_date[r.date.isoformat()].append(r)

        ai_date_to_clusters: dict[str, dict[str, int]] = {}
        for date_str, day_routes in human_by_date.items():
            cust_pcs = {}
            for r in day_routes:
                for c in r.customer_ids:
                    cust_pcs[c] = r.pc_per_customer.get(c, 0.0)
            preds = self.plan_day(date_str, cust_pcs)
            ai_date_to_clusters.update(preds.date_to_clusters)

        preds = PredictedClusters(date_to_clusters=ai_date_to_clusters)
        return hard_mode_eval(human_routes, preds)


def run_io_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    epochs: int = 50,
) -> "tuple[HardModeMetrics, HardModeMetrics, dict]":
    """End-to-end IO baseline."""
    print("Training Inverse Optimization dispatcher...")
    dispatcher = InverseOptimizationDispatcher.fit(train_routes, epochs=epochs)

    print("\nEvaluating...")
    val_m = dispatcher.evaluate_against_human(val_routes)
    test_m = dispatcher.evaluate_against_human(test_routes)

    info = {
        "theta": dispatcher.theta.tolist(),
        "feature_names": FEATURE_NAMES,
        "merge_threshold": dispatcher.merge_threshold,
    }
    return val_m, test_m, info