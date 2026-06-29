"""Full Hierarchical PPM — target SD ≈ 0.038 (Amazon 2nd place).

Key improvements over my v1:
  1. Zone hierarchy: 3 levels from "C-17.3D":
     - Major: "C"
     - Sub-zone number: "17"
     - Sub-sub-zone: "3D"
     Train PPM at all 3 levels, combine queries with weights
  2. Multi-start rollout: try each zone as start, keep best
  3. Proper station handling ("stz" prefix)
  4. Post-processing: sort by lat within similar-cost segments
  5. Train on FULL 6112 routes
  6. Larger PPM order (5) + more data
"""

import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class HierarchicalPPM:
    """PPM with 3-level zone hierarchy matching Amazon 2nd place paper.

    Zone format: "C-17.3D"
      - Major zone: "C" (1 char before dash)
      - Sub-zone number: "17" (2 digits after dash, before dot)
      - Sub-sub-zone: "3D" (after dot)

    Each level has its own n-order Markov model.
    Query combines all levels with weights (0.25 each per paper).
    """

    def __init__(self, order: int = 5):
        self.order = order
        # tables[level] is list of n+1 dicts, one per Markov order
        # tables[level][k] is a dict: context_tuple → Counter of next_symbol
        self.tables: list[list[dict[tuple, Counter]]] = [
            [{} for _ in range(order + 1)] for _ in range(4)  # 4 levels
        ]
        self.order_n = order

    def _levels(self, zone: str) -> tuple[str, str, str, str]:
        """Decompose zone into 4 hierarchy levels."""
        if not isinstance(zone, str) or zone == "stz" or "-" not in zone:
            return ("stz", "stz", "stz", "stz")
        major = zone[0] if zone else "stz"
        after_dash = zone.split("-", 1)[1] if "-" in zone else ""
        # Sub-zone: characters before dot (e.g., "17" from "17.3D")
        sub_zone = after_dash.split(".")[0] if "." in after_dash else after_dash
        # Sub-sub: characters after dot
        sub_sub = after_dash.split(".")[1] if "." in after_dash else ""
        # Full zone (4th level for direct matching)
        return (zone, major, sub_zone, sub_sub)

    def add_sequence(self, sequence: list[str]):
        """Add a zone sequence to all 4 hierarchy levels."""
        seq = ["stz"] + sequence  # prepend station
        for level in range(4):
            decomposed = [self._levels(z)[level] for z in seq]
            tables = self.tables[level]
            for k in range(self.order_n + 1):
                table = tables[k]
                for i in range(len(decomposed) - k):
                    context = tuple(decomposed[i : i + k]) if k > 0 else ()
                    next_sym = decomposed[i + k]
                    if context not in table:
                        table[context] = Counter()
                    table[context][next_sym] += 1

    def predict_next(self, context: list[str], level: int) -> tuple[str, float]:
        """Predict next symbol for a specific hierarchy level."""
        for k in range(min(len(context), self.order_n), -1, -1):
            ctx = tuple(context[-k:]) if k > 0 else ()
            table = self.tables[level][k]
            if ctx in table and len(table[ctx]) > 0:
                sym, cnt = table[ctx].most_common(1)[0]
                total = sum(table[ctx].values())
                return sym, cnt / total
        return None, 0.0

    def predict_combined(self, context: list[str], cluster_weights=(0.25, 0.25, 0.25, 0.25)) -> tuple[str, float]:
        """Predict next zone combining all 4 hierarchy levels."""
        scores = {}  # symbol → total weighted score
        for level in range(4):
            sym, p = self.predict_next(context, level)
            if sym is not None:
                scores[sym] = scores.get(sym, 0) + cluster_weights[level] * p
        if not scores:
            return None, 0.0
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], best[1]

    def rollout(self, zone_list: list[str], cluster_weights=(0.25, 0.25, 0.25, 0.25)) -> list[str]:
        """Multi-start greedy rollout: try each zone as start, keep best."""
        all_zones = list(dict.fromkeys(zone_list))
        if len(all_zones) <= 1:
            return all_zones

        best_seq = None
        best_score = -1e18

        for start_zone in all_zones:
            seq = [start_zone]
            rem = set(all_zones) - {start_zone}
            cur_score = 0.0

            while rem:
                next_sym, p = self.predict_combined(seq, cluster_weights)
                if next_sym is None or next_sym not in rem:
                    # Fallback to order-0 most common
                    fallback = self.tables[0][0].get((), Counter())
                    if fallback:
                        candidates = [(s, c) for s, c in fallback.items() if s in rem]
                        if candidates:
                            next_sym = max(candidates, key=lambda x: x[1])[0]
                            p = 0.001
                        else:
                            next_sym = next(iter(rem))
                            p = 0.001
                    else:
                        next_sym = next(iter(rem))
                        p = 0.001
                seq.append(next_sym)
                rem.remove(next_sym)
                cur_score += math.log(max(p, 0.001))

            if cur_score > best_score:
                best_score = cur_score
                best_seq = seq

        return best_seq if best_seq else all_zones


def train_hierarchical_ppm(route_data, actual_sequences, order=5, max_routes=None):
    """Train hierarchical PPM from training data."""
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

    ppm = HierarchicalPPM(order=order)
    for seq in zone_seqs:
        ppm.add_sequence(seq)
    return ppm


def solve_tsp_orTools(cost_matrix, time_limit=2):
    """OR-Tools TSP solver."""
    n = len(cost_matrix)
    if n < 2:
        return list(range(n))
    if n == 2:
        return [0, 1] if cost_matrix[0][1] <= cost_matrix[1][0] else [1, 0]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_i, to_i):
        return cost_matrix[manager.IndexToNode(from_i)][manager.IndexToNode(to_i)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
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


def predict_route_full(ppm, stops, travel_times, time_limit_per_tsp=2):
    """Full prediction: hierarchical PPM zone order + per-zone TSP + post-processing."""
    stop_ids = list(stops.keys())
    n = len(stop_ids)
    if n < 2:
        return stop_ids

    # Get zones (first occurrence order)
    route_zones = []
    seen = set()
    for sid in stop_ids:
        z = stops[sid].get("zone_id")
        if isinstance(z, str) and z and z != "nan" and z not in seen:
            route_zones.append(z)
            seen.add(z)

    if not route_zones:
        return stop_ids

    # Hierarchical PPM rollout
    zone_order = ppm.rollout(route_zones)
    if not zone_order:
        return stop_ids

    # Group stops by zone
    zone_to_stops = defaultdict(list)
    for sid in stop_ids:
        z = stops[sid].get("zone_id")
        if isinstance(z, str) and z and z != "nan":
            zone_to_stops[z].append(sid)

    # TSP within each zone (in zone_order)
    final_order = []
    for zone in zone_order:
        z_stops = zone_to_stops.get(zone, [])
        if not z_stops:
            continue
        if len(z_stops) == 1:
            final_order.append(z_stops[0])
            continue

        # Build sub-matrix
        sub_n = len(z_stops)
        sub_matrix = [[0] * sub_n for _ in range(sub_n)]
        for i in range(sub_n):
            for j in range(sub_n):
                if i != j:
                    sub_matrix[i][j] = int(travel_times.get(z_stops[i], {}).get(z_stops[j], 999999))

        sub_tour = solve_tsp_orTools(sub_matrix, time_limit=time_limit_per_tsp)
        final_order.extend([z_stops[i] for i in sub_tour])

    # Make sure all stops included
    if set(final_order) != set(stop_ids):
        missing = set(stop_ids) - set(final_order)
        final_order.extend(missing)

    # Post-processing: sort by lat within close-distance segments
    # (paper: "if travel times similar, prefer stops with more packages" — we use lat as proxy)
    final_order = post_process_by_lat(final_order, stops, travel_times)

    return final_order


def post_process_by_lat(order, stops, travel_times, threshold=1.2):
    """Post-processing: for pairs with similar travel times, sort by latitude (north → south).

    Paper says: "drivers usually start with stops with more packages"
    Without package data, we use lat as proxy: sort descending (north → south)
    for segments where consecutive stops have similar travel times.
    """
    if len(order) < 2:
        return order

    result = list(order)
    for i in range(len(result) - 1):
        a, b = result[i], result[i + 1]
        if a not in travel_times or b not in travel_times:
            continue
        t = travel_times[a].get(b, 999999)
        # If travel time between a and b is similar to a and b's neighbors, sort by lat
        if t < threshold * 100:  # within reasonable distance
            # Check if next neighbor also has similar time
            if i + 2 < len(result):
                c = result[i + 2]
                if c in travel_times.get(b, {}):
                    t_bc = travel_times[b].get(c, 999999)
                    if t_bc < threshold * 100:
                        # Both transitions are short — try swapping b and a if lat(b) < lat(a)
                        if a in stops and b in stops:
                            lat_a = stops[a].get("lat", 0)
                            lat_b = stops[b].get("lat", 0)
                            if lat_b > lat_a:  # b is north of a, should come first
                                result[i], result[i + 1] = result[i + 1], result[i]

    return result


def compute_sd(actual_pos, predicted_list):
    n = len(actual_pos)
    if n < 2:
        return 0.0
    pp = {s: i for i, s in enumerate(predicted_list)}
    total = sum(abs(actual_pos[s] - pp.get(s, 0)) for s in actual_pos if s in pp)
    return total / (n * (n - 1) / 2)


def run_full(n_train=None, n_eval=20):
    """Run the full hierarchical PPM solution."""
    print("=" * 60)
    print("  Hierarchical PPM — Target SD=0.038")
    print("=" * 60)

    print("\nLoading training data...")
    with open("data/amazon2021/train_route_data.json") as f:
        train_routes = json.load(f)
    with open("data/amazon2021/train_actual_sequences.json") as f:
        train_seq = json.load(f)

    print("Training hierarchical PPM (FULL training data)...")
    t0 = time.time()
    ppm = train_hierarchical_ppm(train_routes, train_seq, order=5, max_routes=n_train)
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

        predicted = predict_route_full(ppm, stops, tt)
        sd = compute_sd(actual_pos, predicted)
        sds.append(sd)
        print(f"  {rid[:30]}: {n} stops, SD={sd:.4f}")

    if sds:
        mean_sd = sum(sds) / len(sds)
        s_sorted = sorted(sds)
        print(f"\n{'=' * 60}")
        print(f"  Full Hierarchical PPM Results ({len(sds)} routes)")
        print(f"{'=' * 60}")
        print(f"  SD mean={mean_sd:.4f}, median={s_sorted[len(s_sorted)//2]:.4f}")
        print(f"  SD p25={s_sorted[len(s_sorted)//4]:.4f}, p75={s_sorted[3*len(s_sorted)//4]:.4f}")
        print(f"\n  Reference: random≈0.50, my v1 PPM=0.5992, target=0.038")
        return mean_sd


if __name__ == "__main__":
    run_full(n_train=None, n_eval=15)