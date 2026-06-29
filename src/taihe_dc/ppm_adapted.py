"""Adapt AWS PPM solution to Amazon 2021 data.

Uses PPM (Prediction by Partial Matching) from AWS official code (Apache 2.0).
Adapted to work directly with our JSON data files.

Pipeline:
  1. Build zone sequences from training routes (actual_sequences + route_data)
  2. Train PPM model (5th order Markov on zone sequences)
  3. For each eval route: PPM rollout → zone sequence → zone_based_tsp
  4. Compute SD
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Direct import from the AWS repo (it's inside our project)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "amazon_aws_sol"))

from aro.model.ppm import PPM
from aro.model.zone_utils import sort_zones, zone_based_tsp


def extract_zone_sequences(route_data, actual_sequences, max_routes=None):
    """Extract zone sequences from training data.

    Returns: list of zone sequences (each = list of unique zone_ids in visit order)
    """
    zone_seqs = []
    rids = list(actual_sequences.keys())
    if max_routes:
        rids = rids[:max_routes]

    for rid in rids:
        actual = actual_sequences[rid].get("actual", [])
        if not actual:
            continue

        stops = route_data.get(rid, {}).get("stops", {})
        seen = set()
        zone_seq = []
        for sid in actual:
            z = stops.get(sid, {}).get("zone_id")
            if isinstance(z, str) and z and z != "nan" and z not in seen:
                zone_seq.append(z)
                seen.add(z)

        if len(zone_seq) >= 2:
            zone_seqs.append(zone_seq)

    return zone_seqs


def train_ppm(route_data, actual_sequences, order=5, max_routes=None):
    """Train PPM model from historical zone sequences."""
    print(f"Extracting zone sequences from training data...")
    zone_seqs = extract_zone_sequences(route_data, actual_sequences, max_routes)
    print(f"  {len(zone_seqs)} zone sequences extracted")

    # Build vocab
    all_zones = set()
    for seq in zone_seqs:
        all_zones.update(seq)
    vocab_size = len(all_zones) + 1
    print(f"  {vocab_size} unique zones")

    # Train PPM
    print(f"Training PPM (order={order})...")
    ppm = PPM(order, vocab_size=vocab_size)

    for seq in zone_seqs:
        # Add station as first/last (cyclic)
        full_seq = ["stz"] + seq
        ppm.add_sequence(full_seq)

        # Also add hierarchical levels
        ppm.add_sequence(["stz"] + [z[0] for z in seq])  # major zone level
        ppm.add_sequence(["stz"] + [z.split(".")[0].split("-")[-1] for z in seq])  # sub-zone number
        ppm.add_sequence(["stz"] + [z.split(".")[-1] for z in seq])  # sub-sub-zone

    print(f"  PPM trained with {len(ppm.tables)} order tables")
    return ppm


def predict_route(ppm, stops, travel_times):
    """Predict stop sequence for one route using PPM + zone_based_tsp."""
    stop_ids = list(stops.keys())
    n = len(stop_ids)
    if n < 2:
        return stop_ids

    # Get unique zones in this route
    route_zones = []
    seen = set()
    for sid in stop_ids:
        z = stops[sid].get("zone_id")
        if isinstance(z, str) and z and z != "nan" and z not in seen:
            route_zones.append(z)
            seen.add(z)

    if not route_zones:
        return stop_ids

    # Build distance matrix from travel times
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, si in enumerate(stop_ids):
        for j, sj in enumerate(stop_ids):
            if i != j:
                matrix[i][j] = travel_times.get(si, {}).get(sj, 999999)

    # Run zone_based_tsp
    try:
        tour = zone_based_tsp(matrix, route_zones, ppm, "adapter")
        predicted = [stop_ids[i] for i in tour if i < n]
        # Make sure all stops are included
        if len(predicted) < n:
            missing = set(stop_ids) - set(predicted)
            predicted.extend(missing)
        return predicted
    except Exception as e:
        print(f"  zone_based_tsp failed: {e}, falling back to input order")
        return stop_ids


def compute_sd(actual_pos, predicted_list):
    """Compute Sequence Deviation."""
    n = len(actual_pos)
    if n < 2:
        return 0.0
    pp = {s: i for i, s in enumerate(predicted_list)}
    total = sum(abs(actual_pos[s] - pp.get(s, 0)) for s in actual_pos if s in pp)
    return total / (n * (n - 1) / 2)


def run_ppm_adapted(n_train=2000, n_eval=20):
    """Run adapted PPM solution end-to-end."""
    print("=" * 60)
    print("  AWS PPM Adapted — Amazon 2021")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    with open("data/amazon2021/train_route_data.json") as f:
        train_routes = json.load(f)
    with open("data/amazon2021/train_actual_sequences.json") as f:
        train_seq = json.load(f)

    # Train PPM
    ppm = train_ppm(train_routes, train_seq, order=5, max_routes=n_train)

    # Load eval data
    print("\nLoading eval data...")
    with open("data/amazon2021/eval_real_route_data.json") as f:
        eval_routes = json.load(f)
    with open("data/amazon2021/eval_real_actual.json") as f:
        eval_actual = json.load(f)

    # Use small travel times file
    tt_file = "data/amazon2021/eval_tt_small.json"
    print(f"Loading travel times from {tt_file}...")
    with open(tt_file) as f:
        eval_tt_small = json.load(f)

    # Evaluate
    print(f"\nEvaluating on {n_eval} routes...")
    sds = []
    t0 = time.time()

    for rid in list(eval_routes.keys())[:n_eval]:
        actual_raw = eval_actual.get(rid, {}).get("actual", {})
        actual_pos = actual_raw if isinstance(actual_raw, dict) else {s: i for i, s in enumerate(actual_raw)}

        stops = eval_routes[rid].get("stops", {})
        tt = eval_tt_small.get(rid, {})
        if not tt:
            print(f"  {rid[:30]}: no travel times, skipping")
            continue

        n = len(stops)
        if n < 5 or n > 200:
            continue

        predicted = predict_route(ppm, stops, tt)
        sd = compute_sd(actual_pos, predicted)
        sds.append(sd)
        print(f"  {rid[:30]}: {n} stops, SD={sd:.4f}")

    elapsed = time.time() - t0
    if sds:
        mean_sd = sum(sds) / len(sds)
        s_sorted = sorted(sds)
        print(f"\n{'=' * 60}")
        print(f"  PPM Adapted Results ({len(sds)} routes, {elapsed:.0f}s)")
        print(f"{'=' * 60}")
        print(f"  SD mean={mean_sd:.4f}")
        print(f"  SD median={s_sorted[len(s_sorted)//2]:.4f}")
        print(f"  SD p25={s_sorted[len(s_sorted)//4]:.4f}")
        print(f"\n  Reference: random≈0.50, AWS paper 0.038, top 0.025")
        return mean_sd
    else:
        print("No routes evaluated!")
        return None


if __name__ == "__main__":
    run_ppm_adapted(n_train=2000, n_eval=10)