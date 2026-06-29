"""Auto-extract 8 SOPs from any 太古-style dispatch dataset.

This is the **new** version — it derives SOP thresholds from the data itself
(rather than hardcoding 太和DC's PC>500 / 9-12 customers). For each SOP we
compute the empirical distribution and choose the threshold that maximizes
the separation between "independent routes" vs "shared routes".

Reference: 太和DC SOP.md / SOP详解.md in iCloud Obsidian
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, median

from taihe_dc.data.schema import DispatchDataset, Route


@dataclass(frozen=True)
class SopReport:
    """Summary of all 8 SOPs auto-extracted from the dataset."""

    sop1_pc_density_threshold: float        # PC/customer/day threshold for "must be independent"
    sop1_separation: dict[str, float]       # {independent_above, shared_below}
    sop2_sweet_spot: tuple[int, int]        # (lower, upper) customer count for sweet spot
    sop2_route_size_distribution: dict[int, int]
    sop3_top_combinations: list[tuple[str, ...]]   # top customer-type combos (placeholder — types not in source data)
    sop4_member_stability: float            # co-occurrence rate (0..1)
    sop5_time_rhythm: dict                  # {weekday: avg_route_size, season: avg_size}
    sop6_driver_split: dict[str, float]     # {driver_category: avg_route_size}
    sop7_capacity_softbound: float          # 95th percentile PC
    sop8_vehicle_fixity: float              # fraction of routes always same vehicle
    sop8_route_to_vehicle_entropy: float    # avg entropy of route→vehicle mapping
    notes: list[str] = field(default_factory=list)


def _route_pc_per_customer(r: Route) -> float:
    if not r.pc_per_customer:
        return 0.0
    return r.route_pc_total / max(1, r.n_customers)


def _independent_vs_shared(routes: list[Route]) -> tuple[list[Route], list[Route]]:
    """Independent routes: only 1 customer (or 1 large + tiny). Shared: 2+."""
    indep: list[Route] = []
    shared: list[Route] = []
    for r in routes:
        if r.n_customers == 1:
            indep.append(r)
        else:
            shared.append(r)
    return indep, shared


def _find_pc_density_threshold(indep: list[Route], shared: list[Route]) -> tuple[float, dict[str, float]]:
    """Sweep thresholds, find the one that best separates indep vs shared."""
    if not indep or not shared:
        return 0.0, {"independent_above": 0.0, "shared_below": 0.0}

    indep_pcs = sorted(_route_pc_per_customer(r) for r in indep)
    shared_pcs = sorted(_route_pc_per_customer(r) for r in shared)

    # Sweep candidate thresholds around the median of all PC values
    all_pcs = indep_pcs + shared_pcs
    p25, p50, p75 = (
        all_pcs[int(0.25 * len(all_pcs))],
        all_pcs[int(0.50 * len(all_pcs))],
        all_pcs[int(0.75 * len(all_pcs))],
    )
    candidates = sorted({p25, p50, p75, mean(all_pcs)})

    best_thr = 0.0
    best_score = -1.0
    best_separation = {"independent_above": 0.0, "shared_below": 0.0}

    for thr in candidates:
        indep_above = sum(1 for p in indep_pcs if p >= thr) / max(1, len(indep_pcs))
        shared_below = sum(1 for p in shared_pcs if p < thr) / max(1, len(shared_pcs))
        score = indep_above + shared_below
        if score > best_score:
            best_score = score
            best_thr = thr
            best_separation = {"independent_above": round(indep_above, 3), "shared_below": round(shared_below, 3)}

    return round(best_thr, 2), best_separation


def _route_size_distribution(routes: list[Route]) -> dict[int, int]:
    bucket: dict[int, int] = {}
    for r in routes:
        bucket[r.n_customers] = bucket.get(r.n_customers, 0) + 1
    return dict(sorted(bucket.items()))


def _find_sweet_spot(distribution: dict[int, int]) -> tuple[int, int]:
    """Find the contiguous window with the highest density."""
    if not distribution:
        return (0, 0)
    sizes = sorted(distribution.keys())
    counts = [distribution[s] for s in sizes]
    # window of size 4
    best_sum = -1
    best_window = (sizes[0], sizes[0] + 3)
    for i in range(len(sizes) - 3):
        window_sum = sum(counts[i:i + 4])
        if window_sum > best_sum:
            best_sum = window_sum
            best_window = (sizes[i], sizes[i + 3])
    return best_window


def _member_stability(routes: list[Route]) -> float:
    """Co-occurrence rate of customers within the same route.

    For each customer, fraction of (customer, customer') pairs that always
    appear together. Higher = more stable. 太和DC = 100%.
    """
    pair_routes: dict[tuple[str, str], set[str]] = {}
    customer_routes: dict[str, set[str]] = {}

    for r in routes:
        for cid in set(r.customer_ids):
            customer_routes.setdefault(cid, set()).add(r.route_id)

    for r in routes:
        cids = sorted(set(r.customer_ids))
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                pair_routes.setdefault((cids[i], cids[j]), set()).add(r.route_id)

    if not pair_routes:
        return 0.0
    stabilities: list[float] = []
    for (c1, c2), shared in pair_routes.items():
        # Stability = (n shared routes) / min(n routes for c1, c2)
        denom = min(len(customer_routes.get(c1, set())), len(customer_routes.get(c2, set())))
        if denom > 0:
            stabilities.append(len(shared) / denom)
    return round(mean(stabilities), 3) if stabilities else 0.0


def _time_rhythm(routes: list[Route]) -> dict[str, dict]:
    """Per-weekday and per-month average route size."""
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_sizes: dict[int, list[int]] = {}
    month_sizes: dict[int, list[int]] = {}
    for r in routes:
        wd = r.date.weekday()
        weekday_sizes.setdefault(wd, []).append(r.n_customers)
        month_sizes.setdefault(r.date.month, []).append(r.n_customers)

    return {
        "weekday": {weekday_names[wd]: round(mean(sz), 1) for wd, sz in sorted(weekday_sizes.items())},
        "month": {f"M{m}": round(mean(sz), 1) for m, sz in sorted(month_sizes.items())},
        "busiest_weekday": weekday_names[max(weekday_sizes, key=lambda wd: mean(weekday_sizes[wd]))] if weekday_sizes else "—",
        "busiest_month": f"M{max(month_sizes, key=lambda m: mean(month_sizes[m]))}" if month_sizes else "—",
    }


def _driver_split(routes: list[Route]) -> dict[str, float]:
    """Bucket drivers by appearance frequency."""
    driver_count: dict[str, int] = {}
    for r in routes:
        driver_count[r.driver_name] = driver_count.get(r.driver_name, 0) + 1

    buckets = {"fixed (>=50 routes)": [], "swing (10-49)": [], "occasional (<10)": []}
    for driver, cnt in driver_count.items():
        if cnt >= 50:
            buckets["fixed (>=50 routes)"].append(driver)
        elif cnt >= 10:
            buckets["swing (10-49)"].append(driver)
        else:
            buckets["occasional (<10)"].append(driver)

    result: dict[str, float] = {}
    for label, drivers in buckets.items():
        if not drivers:
            result[label] = 0.0
            continue
        sizes: list[int] = []
        for r in routes:
            if r.driver_name in drivers:
                sizes.append(r.n_customers)
        result[label] = round(mean(sizes), 1) if sizes else 0.0
    return result


def _capacity_softbound(routes: list[Route]) -> float:
    """95th percentile of route total PC."""
    pcs = sorted(r.route_pc_total for r in routes)
    if not pcs:
        return 0.0
    return pcs[int(0.95 * len(pcs))]


def _vehicle_fixity(routes: list[Route]) -> tuple[float, float]:
    """How fixed is the vehicle↔(customer list) binding?

    Returns:
        (fixity_rate, entropy)
        fixity_rate = fraction of routes where customer-list is always on same plate
        entropy     = avg Shannon entropy of route-list→plate mapping (normalized)
    """
    list_to_plates: dict[frozenset[str], set[str]] = {}
    for r in routes:
        key = frozenset(r.customer_ids)
        list_to_plates.setdefault(key, set()).add(r.plate)

    n_total = len(list_to_plates)
    n_fixed = sum(1 for plates in list_to_plates.values() if len(plates) == 1)

    fixity_rate = n_fixed / max(1, n_total)

    entropies: list[float] = []
    for plates in list_to_plates.values():
        n = len(plates)
        if n <= 1:
            entropies.append(0.0)
        else:
            p = 1.0 / n
            entropies.append(-n * p * math.log2(p))
    max_entropy = math.log2(max(len({r.plate for r in routes}), 2))
    avg_entropy = mean(entropies) if entropies else 0.0
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0
    return round(fixity_rate, 3), round(normalized_entropy, 3)


def extract_sops(ds: DispatchDataset) -> SopReport:
    """Compute all 8 SOPs from the dataset."""
    routes = list(ds.routes)
    indep, shared = _independent_vs_shared(routes)

    pc_thr, sep = _find_pc_density_threshold(indep, shared)
    size_dist = _route_size_distribution(routes)
    sweet = _find_sweet_spot(size_dist)
    stability = _member_stability(routes)
    rhythm = _time_rhythm(routes)
    driver_split = _driver_split(routes)
    p95_pc = _capacity_softbound(routes)
    fixity, entropy = _vehicle_fixity(routes)

    notes: list[str] = []
    if pc_thr == 0.0:
        notes.append("Could not find meaningful PC threshold — dataset may have only one route type.")
    if stability > 0.95:
        notes.append(f"High member stability ({stability}) → route formation is essentially lookup, not optimization.")
    if fixity > 0.95:
        notes.append(f"High vehicle fixity ({fixity}) → vehicle-customer binding is strong (SOP-8 territory).")

    return SopReport(
        sop1_pc_density_threshold=pc_thr,
        sop1_separation=sep,
        sop2_sweet_spot=sweet,
        sop2_route_size_distribution=size_dist,
        sop3_top_combinations=[],  # customer types not in source data; would need enrichment
        sop4_member_stability=stability,
        sop5_time_rhythm=rhythm,
        sop6_driver_split=driver_split,
        sop7_capacity_softbound=p95_pc,
        sop8_vehicle_fixity=fixity,
        sop8_route_to_vehicle_entropy=entropy,
        notes=notes,
    )


def format_report(rpt: SopReport) -> str:
    """Human-readable multi-line report."""
    lines = []
    lines.append("═══ AUTO-EXTRACTED SOPs ═══")
    lines.append("")
    lines.append(f"SOP-1 PC density threshold: PC > {rpt.sop1_pc_density_threshold:.1f} → must be independent")
    lines.append(f"       separation: independent_above={rpt.sop1_separation['independent_above']:.1%}, shared_below={rpt.sop1_separation['shared_below']:.1%}")
    lines.append("")
    lines.append(f"SOP-2 sweet spot: {rpt.sop2_sweet_spot[0]}-{rpt.sop2_sweet_spot[1]} customers per route")
    lines.append(f"       top sizes: {sorted(rpt.sop2_route_size_distribution.items(), key=lambda x: -x[1])[:5]}")
    lines.append("")
    lines.append(f"SOP-4 member stability: {rpt.sop4_member_stability:.1%}")
    lines.append("")
    lines.append(f"SOP-5 time rhythm:")
    lines.append(f"       weekday avg size: {rpt.sop5_time_rhythm['weekday']}")
    lines.append(f"       month avg size:   {rpt.sop5_time_rhythm['month']}")
    lines.append(f"       busiest weekday:  {rpt.sop5_time_rhythm['busiest_weekday']}")
    lines.append(f"       busiest month:    {rpt.sop5_time_rhythm['busiest_month']}")
    lines.append("")
    lines.append(f"SOP-6 driver split: {rpt.sop6_driver_split}")
    lines.append("")
    lines.append(f"SOP-7 capacity softbound: 95% routes have PC ≤ {rpt.sop7_capacity_softbound:.0f}")
    lines.append("")
    lines.append(f"SOP-8 vehicle fixity: {rpt.sop8_vehicle_fixity:.1%} of customer-lists are on a single vehicle")
    lines.append(f"       route-to-vehicle entropy (normalized): {rpt.sop8_route_to_vehicle_entropy:.3f}")
    lines.append("")
    if rpt.notes:
        lines.append("⚠ Notes:")
        for n in rpt.notes:
            lines.append(f"   - {n}")
    return "\n".join(lines)