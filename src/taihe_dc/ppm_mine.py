"""PPM reimplemented from scratch — clean, minimal version.

The AWS PPM has hierarchical structure, escape mechanism, log-probabilities, and
complex rollout. This is a simplified version that captures the CORE idea:

  - n-order Markov chain on zone sequences
  - Simple P(next | context) with backoff to lower order
  - Greedy rollout for zone sequence
  - Combined with OR-Tools TSP for stop ordering

Key question: can a clean PPM give SD better than 0.65?
"""

import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class SimplePPM:
    """Simple n-order PPM for zone sequence prediction.

    Stores counts of (context, next_symbol) pairs at multiple orders.
    For prediction: try highest order, fall back to lower order if context unseen.
    """

    def __init__(self, order: int = 5):
        self.order = order
        # tables[k] is a dict: context_tuple → Counter of next_symbol counts
        self.tables: list[dict[tuple, Counter]] = [{} for _ in range(order + 1)]

    def add_sequence(self, sequence: list[str]):
        """Add a sequence to the model (all order tables)."""
        seq = ["stz"] + sequence  # prepend station

        for k in range(self.order + 1):
            table = self.tables[k]
            for i in range(len(seq) - k):
                context = tuple(seq[i : i + k]) if k > 0 else ()
                next_sym = seq[i + k]
                if context not in table:
                    table[context] = Counter()
                table[context][next_sym] += 1

    def predict_next(self, context: list[str]) -> tuple[str, float]:
        """Predict next symbol with probability. Backoff to lower orders."""
        for k in range(min(len(context), self.order), -1, -1):
            ctx = tuple(context[-k:]) if k > 0 else ()
            table = self.tables[k]
            if ctx in table and len(table[ctx]) > 0:
                # Return most common
                sym, cnt = table[ctx].most_common(1)[0]
                total = sum(table[ctx].values())
                return sym, cnt / total
        return None, 0.0

    def rollout(self, zone_list: list[str]) -> list[str]:
        """Greedy zone sequence generation from PPM."""
        if not zone_list:
            return []

        all_zones = list(dict.fromkeys(zone_list))  # preserve first-occurrence order, dedup

        # Try all possible start zones, pick best rollout
        best_seq = None
        best_prob = -1.0
        for start in all_zones:
            seq = [start]
            rem = set(all_zones) - {start}
            cur_prob = 0.0
            while rem:
                next_sym, p = self.predict_next(seq)
                if next_sym is None or next_sym not in rem:
                    # Fallback: pick most common zone in remaining
                    next_sym = max(rem, key=lambda z: sum(self.tables[0].get((), Counter()).values()))
                    p = 0.01
                seq.append(next_sym)
                rem.remove(next_sym)
                cur_prob += math.log(max(p, 0.001))
            if cur_prob > best_prob:
                best_prob = cur_prob
                best_seq = seq

        return best_seq if best_seq else all_zones


def train_ppm_from_routes(route_data, actual_sequences, order=5, max_routes=None):
    """Train PPM from our training route JSON data."""
    print("Extracting zone sequences...")
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

    print(f"  {len(zone_seqs)} zone sequences")

    ppm = SimplePPM(order=order)
    for seq in zone_seqs:
        ppm.add_sequence(seq)
    return ppm


def solve_tsp_simple(cost_matrix, time_limit=2):
    """Simple OR-Tools TSP."""
    n = len(cost_matrix)
    if n < 2:
        return list(range(n))
    if n == 2:
        return [0, 1] if cost_matrix[0][1] < cost_matrix[1][0] else [1, 0]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_i, to_i):
        return cost_matrix[manager.IndexToNode(from_i)][manager.IndexToNode(to_i)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = time_limit
    solution = routing.SolveWithParameters(params)
    if not solution:
        return list(range(n))

    tour = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return tour


def predict_route_ppm(ppm, stops, travel_times, time_limit_per_tsp=1):
    """Predict stop order: PPM zone order + TSP within each zone."""
    stop_ids = list(stops.keys())
    n = len(stop_ids)
    if n < 2:
        return stop_ids

    # Get unique zones (first occurrence order)
    route_zones = []
    seen = set()
    for sid in stop_ids:
        z = stops[sid].get("zone_id")
        if isinstance(z, str) and z and z != "nan" and z not in seen:
            route_zones.append(z)
            seen.add(z)

    if not route_zones:
        return stop_ids

    # PPM rollout for zone order
    zone_order = ppm.rollout(route_zones)
    if not zone_order:
        return stop_ids

    # Group stops by zone (in rollout order)
    zone_to_stops = defaultdict(list)
    for sid in stop_ids:
        z = stops[sid].get("zone_id")
        if isinstance(z, str) and z and z != "nan":
            zone_to_stops[z].append(sid)

    # TSP within each zone + inter-zone chaining
    final_order = []
    for zi, zone in enumerate(zone_order):
        z_stops = zone_to_stops.get(zone, [])
        if not z_stops:
            continue
        if len(z_stops) == 1:
            final_order.append(z_stops[0])
            continue

        # Build sub-matrix for this zone's stops
        sub_n = len(z_stops)
        sub_matrix = [[0] * sub_n for _ in range(sub_n)]
        for i in range(sub_n):
            for j in range(sub_n):
                if i != j:
                    sub_matrix[i][j] = int(travel_times.get(z_stops[i], {}).get(z_stops[j], 999999))

        # Solve TSP for this zone
        sub_tour = solve_tsp_simple(sub_matrix, time_limit=time_limit_per_tsp)
        final_order.extend([z_stops[i] for i in sub_tour])

    # Make sure all stops included
    if set(final_order) != set(stop_ids):
        missing = set(stop_ids) - set(final_order)
        final_order.extend(missing)

    return final_order


def compute_sd(actual_pos, predicted_list):
    n = len(actual_pos)
    if n < 2:
        return 0.0
    pp = {s: i for i, s in enumerate(predicted_list)}
    total = sum(abs(actual_pos[s] - pp.get(s, 0)) for s in actual_pos if s in pp)
    return total / (n * (n - 1) / 2)


def run_my_ppm(n_train=2000, n_eval=10):
    """Run the from-scratch PPM implementation."""
    print("=" * 60)
    print("  PPM Reimplemented from Scratch")
    print("=" * 60)

    print("\nLoading training data...")
    with open("data/amazon2021/train_route_data.json") as f:
        train_routes = json.load(f)
    with open("data/amazon2021/train_actual_sequences.json") as f:
        train_seq = json.load(f)

    print("Training PPM...")
    t0 = time.time()
    ppm = train_ppm_from_routes(train_routes, train_seq, order=5, max_routes=n_train)
    print(f"  Trained in {time.time() - t0:.1f}s")

    print("\nLoading eval data...")
    with open("data/amazon2021/eval_real_route_data.json") as f:
        eval_routes = json.load(f)
    with open("data/amazon2021/eval_real_actual.json") as f:
        eval_actual = json.load(f)
    with open("data/amazon2021/eval_tt_small.json") as f:
        eval_tt = json.load(f)

    print(f"\nEvaluating on {n_eval} routes...")
    sds = []
    for rid in list(eval_routes.keys())[:n_eval]:
        actual_raw = eval_actual.get(rid, {}).get("actual", {})
        actual_pos = actual_raw if isinstance(actual_raw, dict) else {s: i for i, s in enumerate(actual_raw)}

        stops = eval_routes[rid].get("stops", {})
        tt = eval_tt.get(rid, {})
        if not tt:
            continue
        n = len(stops)
        if n < 5 or n > 200:
            continue

        predicted = predict_route_ppm(ppm, stops, tt)
        sd = compute_sd(actual_pos, predicted)
        sds.append(sd)
        print(f"  {rid[:30]}: {n} stops, SD={sd:.4f}")

    if sds:
        mean_sd = sum(sds) / len(sds)
        s_sorted = sorted(sds)
        print(f"\n{'=' * 60}")
        print(f"  My PPM Results ({len(sds)} routes)")
        print(f"{'=' * 60}")
        print(f"  SD mean={mean_sd:.4f}, median={s_sorted[len(s_sorted)//2]:.4f}")
        print(f"\n  Reference: random≈0.50, AWS paper 0.038")
        return mean_sd


if __name__ == "__main__":
    run_my_ppm(n_train=2000, n_eval=10)