"""Amazon 2021 Last Mile Challenge — Zone-based routing baseline.

Two-stage approach (simplified 2nd place methodology):
  Stage 1: Learn zone transition probabilities from training routes (Markov)
  Stage 2: For test routes, order zones by learned transitions + order stops within zone

Evaluation: Sequence Deviation (SD) — lower is better.
  Top teams: 0.025-0.037
  Random: ~0.15
  This baseline target: < 0.10

Data: data/amazon2021/train_route_data.json (6112 routes)
      data/amazon2021/train_actual_sequences.json (ground truth orderings)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


def load_amazon_data(data_dir: str = "data/amazon2021"):
    """Load Amazon 2021 training data."""
    data_dir = Path(data_dir)
    with open(data_dir / "train_route_data.json") as f:
        route_data = json.load(f)
    with open(data_dir / "train_actual_sequences.json") as f:
        actual_sequences = json.load(f)
    return route_data, actual_sequences


def extract_zone_sequences(route_data: dict, actual_sequences: dict, max_routes: int = None):
    """Extract zone-level sequences from actual stop orderings.

    Returns: list of zone sequences (each = list of zone_ids in visit order)
    """
    zone_sequences = []
    route_ids = list(actual_sequences.keys())
    if max_routes:
        route_ids = route_ids[:max_routes]

    for rid in route_ids:
        seq_data = actual_sequences[rid]
        actual = seq_data.get("actual", []) if isinstance(seq_data, dict) else seq_data
        if not actual:
            continue

        rd = route_data.get(rid, {})
        stops = rd.get("stops", {})

        # Map stop_id → zone_id
        stop_to_zone = {}
        for sid, stop in stops.items():
            z = stop.get("zone_id")
            if isinstance(z, str) and z and z != "nan":
                stop_to_zone[sid] = z
            else:
                stop_to_zone[sid] = None

        # Build zone sequence (first occurrence of each zone)
        zone_seq = []
        seen = set()
        for sid in actual:
            z = stop_to_zone.get(sid)
            if z and z not in seen:
                zone_seq.append(z)
                seen.add(z)

        if zone_seq:
            zone_sequences.append((rid, zone_seq, stop_to_zone, actual))

    return zone_sequences


def learn_zone_transitions(zone_sequences: list, smoothing: float = 0.01):
    """Learn zone transition probabilities from training data.

    Returns:
      zone_prob: dict[zone_a] → dict[zone_b] → P(zone_b | zone_a)
      zone_first: Counter of first zones
    """
    zone_transitions = defaultdict(Counter)
    zone_first = Counter()
    zone_freq = Counter()

    for _, zone_seq, _, _ in zone_sequences:
        if not zone_seq:
            continue
        zone_first[zone_seq[0]] += 1
        for z in zone_seq:
            zone_freq[z] += 1
        for i in range(len(zone_seq) - 1):
            zone_transitions[zone_seq[i]][zone_seq[i + 1]] += 1

    # Convert to probabilities with smoothing
    zone_prob = {}
    all_zones = set(zone_freq.keys())
    n_zones = len(all_zones)

    for za in all_zones:
        total = sum(zone_transitions[za].values()) + smoothing * n_zones
        zone_prob[za] = {}
        for zb in all_zones:
            count = zone_transitions[za].get(zb, 0)
            zone_prob[za][zb] = (count + smoothing) / total

    return zone_prob, zone_first, zone_freq


def predict_zone_order(route_zones: list, zone_prob: dict, zone_first: Counter) -> list:
    """Predict zone ordering using greedy transition probability.

    Strategy: start from most likely first zone, greedily pick highest P(next).
    """
    if not route_zones:
        return []

    available = set(route_zones)

    # Pick start zone: most common first zone among available
    start = max(available, key=lambda z: zone_first.get(z, 0))
    ordered = [start]
    available.remove(start)

    while available:
        last = ordered[-1]
        # Pick next zone with highest transition probability
        best_zone = max(available, key=lambda z: zone_prob.get(last, {}).get(z, 0))
        ordered.append(best_zone)
        available.remove(best_zone)

    return ordered


def predict_stop_sequence(
    route_stops: dict,
    actual_order: list,
    zone_prob: dict,
    zone_first: Counter,
) -> list:
    """Predict full stop sequence: zone order + within-zone order.

    Within each zone: order stops by their lat (north→south) as simple heuristic.
    """
    # Map stops to zones
    stop_zones = {}
    for sid, stop in route_stops.items():
        z = stop.get("zone_id")
        if isinstance(z, str) and z and z != "nan":
            stop_zones[sid] = z
        else:
            stop_zones[sid] = "UNKNOWN"

    # Get unique zones in this route
    route_zones = list(set(stop_zones.values()))
    if "UNKNOWN" in route_zones:
        route_zones.remove("UNKNOWN")

    # Predict zone order
    zone_order = predict_zone_order(route_zones, zone_prob, zone_first)
    zone_rank = {z: i for i, z in enumerate(zone_order)}
    zone_rank["UNKNOWN"] = len(zone_order)  # unknown zones go last

    # Sort stops: by zone rank, then by lat (descending = north first)
    stop_list = list(route_stops.keys())
    stop_list.sort(key=lambda sid: (
        zone_rank.get(stop_zones[sid], 999),
        -(route_stops[sid].get("lat") or 0),  # north first
    ))

    return stop_list


def sequence_deviation(actual: list, predicted: list) -> float:
    """Amazon SD metric: normalized position difference.

    SD = (2 / N²) × Σ |pos_actual(i) - pos_predicted(i)|

    SD=0: perfect, SD→1: completely wrong
    """
    n = len(actual)
    if n < 2:
        return 0.0

    pos_actual = {sid: i for i, sid in enumerate(actual)}
    pos_predicted = {sid: i for i, sid in enumerate(predicted)}

    total_diff = 0.0
    count = 0
    for sid in actual:
        if sid in pos_predicted:
            total_diff += abs(pos_actual[sid] - pos_predicted[sid])
            count += 1

    if count == 0:
        return 1.0

    # Normalize: max possible diff for N items = N-1
    # Amazon formula: SD = (sum of |diff|) / (N * (N-1) / 2)
    return total_diff / (n * (n - 1) / 2) if n > 1 else 0.0


def run_amazon_baseline(train_frac: float = 0.8):
    """Run zone-based baseline on Amazon 2021 data.

    Split training data into train/test, learn zone transitions,
    predict test sequences, compute SD.
    """
    print("Loading Amazon 2021 data...")
    route_data, actual_sequences = load_amazon_data()
    print(f"  Routes: {len(route_data)}, Sequences: {len(actual_sequences)}")

    # Extract zone sequences
    print("Extracting zone sequences...")
    all_seqs = extract_zone_sequences(route_data, actual_sequences)
    print(f"  Valid sequences: {len(all_seqs)}")

    # Split train/test
    n = len(all_seqs)
    n_train = int(n * train_frac)
    train_seqs = all_seqs[:n_train]
    test_seqs = all_seqs[n_train:]
    print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}")

    # Learn zone transitions
    print("Learning zone transitions (Markov)...")
    zone_prob, zone_first, zone_freq = learn_zone_transitions(train_seqs)
    print(f"  Unique zones: {len(zone_freq)}")
    print(f"  Top first zones: {zone_first.most_common(5)}")

    # Evaluate on test
    print("\nEvaluating on test routes...")
    sds = []
    for rid, zone_seq, stop_to_zone, actual in test_seqs[:200]:  # first 200 for speed
        rd = route_data.get(rid, {})
        stops = rd.get("stops", {})
        predicted = predict_stop_sequence(stops, actual, zone_prob, zone_first)
        sd = sequence_deviation(actual, predicted)
        sds.append(sd)

    mean_sd = sum(sds) / len(sds)
    sorted_sds = sorted(sds)
    median_sd = sorted_sds[len(sorted_sds) // 2]

    print(f"\n{'═' * 60}")
    print(f"  Amazon 2021 Zone-Based Baseline Results")
    print(f"{'═' * 60}")
    print(f"  Routes evaluated: {len(sds)}")
    print(f"  SD (mean):   {mean_sd:.4f}")
    print(f"  SD (median): {median_sd:.4f}")
    print(f"  SD (p25):    {sorted_sds[len(sorted_sds)//4]:.4f}")
    print(f"  SD (p75):    {sorted_sds[3*len(sorted_sds)//4]:.4f}")
    print()
    print(f"  Reference:")
    print(f"    Random:     ~0.15")
    print(f"    Top 3:      0.025-0.037")
    print(f"    This baseline: {mean_sd:.4f}")
    print()

    return mean_sd, sds


if __name__ == "__main__":
    run_amazon_baseline()