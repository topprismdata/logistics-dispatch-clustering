"""End-to-end dispatch pipeline.

Given a day's customer orders → output route assignments.

Pipeline:
  1. Load historical routes → build Louvain community partition
  2. For each new day's customers:
     a. Assign each customer to its community (or singleton if unseen)
     b. SOP-1: customers with PC > 260 → solo routes
     c. Group remaining by community
     d. Time-window split (unload_time > 2h apart)
     e. Capacity bin packing (PC <= 3000 per route)
  3. Output: list of routes, each with customers + total PC + suggested vehicle type

Usage:
    from taihe_dc.pipeline import DispatchPipeline
    pipe = DispatchPipeline.fit(train_routes)
    routes = pipe.dispatch(test_routes_for_date)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from taihe_dc.data import Route, DispatchDataset, load_dataset
from taihe_dc.baselines.community_louvain import build_cooccurrence_graph, detect_communities
from taihe_dc.baselines.community_with_capacity import (
    ROUTE_PC_CAP,
    SINGLE_CUSTOMER_PC_THRESHOLD,
    _greedy_bin_pack,
)


DEFAULT_TIME_WINDOW_HOURS = 2.0


@dataclass
class PredictedRoute:
    """One predicted route (cluster of customers assigned together).

    Note: vehicle type is NOT predicted. Per user instruction, only PC
    (capacity unit) matters. PC IS the capacity metric — no ton conversion.
    """
    customers: tuple[str, ...]
    pc_total: float
    pc_per_customer: dict[str, float]
    has_sop1_solo: bool                  # true if forced solo by PC > threshold
    n_customers: int

    def summary(self) -> str:
        return (f"Route({self.n_customers} cust, PC={self.pc_total:.0f}, "
                f"solo={self.has_sop1_solo})")


@dataclass
class DayDispatchResult:
    """Dispatch result for one date."""
    date: date
    routes: tuple[PredictedRoute, ...]
    n_customers_input: int
    n_unseen_customers: int              # customers not in train vocab
    n_solo_routes: int                   # SOP-1 forced solo
    n_capacity_split: int                # routes from bin packing


def suggest_vehicle_type(pc_total: float) -> str:
    """DEPRECATED. Per user: PC is the capacity unit, don't predict vehicle type.
    Kept for backward compat but returns empty string.
    """
    return ""


@dataclass
class DispatchPipeline:
    """End-to-end dispatch pipeline. Fit on train, predict on new days."""

    partition: dict[str, int] = field(default_factory=dict)
    min_weight: int = 2
    resolution: float = 1.0
    route_pc_cap: float = ROUTE_PC_CAP
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD
    time_window_hours: float = DEFAULT_TIME_WINDOW_HOURS
    # Stats from training
    n_communities: int = 0
    n_train_customers: int = 0

    @classmethod
    def fit(
        cls,
        train_routes: list[Route],
        min_weight: int = 2,
        resolution: float = 1.0,
    ) -> "DispatchPipeline":
        """Train: build co-occurrence graph + detect communities."""
        G = build_cooccurrence_graph(train_routes, min_weight=min_weight, use_pmi=True)
        partition = detect_communities(G, resolution=resolution)
        train_customers = set()
        for r in train_routes:
            train_customers.update(r.customer_ids)
        return cls(
            partition=partition,
            min_weight=min_weight,
            resolution=resolution,
            n_communities=len(set(partition.values())),
            n_train_customers=len(train_customers),
        )

    def dispatch_day(
        self,
        date_str: str,
        customer_pcs: dict[str, float],
        customer_unload_times: Optional[dict[str, datetime]] = None,
    ) -> DayDispatchResult:
        """Dispatch customers for one date.

        Args:
            date_str: ISO date string (e.g. "2026-05-15")
            customer_pcs: customer_id → PC for that day
            customer_unload_times: optional customer_id → unload_time
        """
        if customer_unload_times is None:
            customer_unload_times = {}

        # Group by community
        comm_to_custs: dict[int, list[tuple[str, float, Optional[datetime]]]] = defaultdict(list)
        n_unseen = 0
        for c, pc in customer_pcs.items():
            comm_id = self.partition.get(c)
            if comm_id is None:
                comm_id = -(hash((date_str, c)) % (10**9))
                n_unseen += 1
            t = customer_unload_times.get(c)
            comm_to_custs[comm_id].append((c, pc, t))

        routes: list[PredictedRoute] = []
        n_solo = 0
        n_split = 0

        for comm_id, custs in comm_to_custs.items():
            # SOP-1: solo for big PC
            solo = [(c, pc, t) for c, pc, t in custs if pc > self.solo_threshold]
            group = [(c, pc, t) for c, pc, t in custs if pc <= self.solo_threshold]

            for c, pc, _ in solo:
                routes.append(PredictedRoute(
                    customers=(c,),
                    pc_total=pc,
                    pc_per_customer={c: pc},
                    has_sop1_solo=True,
                    n_customers=1,
                ))
                n_solo += 1

            if not group:
                continue

            # Time-window split
            with_time = sorted([(c, pc, t) for c, pc, t in group if t is not None], key=lambda x: x[2])
            no_time = [(c, pc, t) for c, pc, t in group if t is None]

            time_bins: list[list[tuple[str, float, Optional[datetime]]]] = []
            if with_time:
                cur = [with_time[0]]
                for c, pc, t in with_time[1:]:
                    last_t = cur[-1][2]
                    gap_h = (t - last_t).total_seconds() / 3600
                    if gap_h > self.time_window_hours:
                        time_bins.append(cur)
                        cur = [(c, pc, t)]
                    else:
                        cur.append((c, pc, t))
                time_bins.append(cur)
            if no_time:
                time_bins.append(no_time)

            for bin_items in time_bins:
                items_for_pack = [(c, pc) for c, pc, _ in bin_items]
                total_pc = sum(pc for _, pc in items_for_pack)
                if len(items_for_pack) == 1 or total_pc <= self.route_pc_cap:
                    routes.append(self._make_route(items_for_pack, solo=False))
                else:
                    bins = _greedy_bin_pack(items_for_pack, self.route_pc_cap)
                    for bin_cids in bins:
                        bin_items_dict = {c: dict(items_for_pack)[c] for c in bin_cids}
                        routes.append(self._make_route(list(bin_items_dict.items()), solo=False))
                    n_split += 1

        return DayDispatchResult(
            date=date.fromisoformat(date_str),
            routes=tuple(routes),
            n_customers_input=len(customer_pcs),
            n_unseen_customers=n_unseen,
            n_solo_routes=n_solo,
            n_capacity_split=n_split,
        )

    def _make_route(self, items: list[tuple[str, float]], solo: bool) -> PredictedRoute:
        cids = tuple(c for c, _ in items)
        pc_map = {c: pc for c, pc in items}
        total = sum(pc for _, pc in items)
        return PredictedRoute(
            customers=cids,
            pc_total=total,
            pc_per_customer=pc_map,
            has_sop1_solo=solo,
            n_customers=len(cids),
        )

    def dispatch_from_routes(self, routes: list[Route]) -> dict[str, DayDispatchResult]:
        """Convenience: dispatch from Route objects (uses their customer PCs + unload times).

        Returns date_str → DayDispatchResult.
        """
        by_date: dict[str, dict[str, float]] = defaultdict(dict)
        by_date_unload: dict[str, dict[str, datetime]] = defaultdict(dict)
        for r in routes:
            ds_str = r.date.isoformat()
            for c in r.customer_ids:
                by_date[ds_str][c] = r.pc_per_customer.get(c, 0.0)
            for d in r.delivery_rows:
                if d.unload_time:
                    by_date_unload[ds_str].setdefault(d.customer_id, d.unload_time)

        results: dict[str, DayDispatchResult] = {}
        for ds_str, cust_pcs in by_date.items():
            results[ds_str] = self.dispatch_day(
                ds_str, cust_pcs, by_date_unload.get(ds_str, {})
            )
        return results


def demo():
    """Run a demo on the real dataset."""
    import warnings
    warnings.filterwarnings("ignore")

    ds = load_dataset("data/raw/全流程报表2026.1.1-5.31.xlsx")
    from taihe_dc.split import chronological_split
    splits = chronological_split(ds, train_frac=0.70, val_frac=0.10)

    print("=== Fitting dispatch pipeline on train (Jan-Apr 16) ===")
    pipe = DispatchPipeline.fit(list(splits.train.routes))
    print(f"  Communities: {pipe.n_communities}")
    print(f"  Train customers: {pipe.n_train_customers}")
    print()

    # Dispatch one test date
    test_date = splits.test.routes[0].date.isoformat()
    test_routes_for_date = [r for r in splits.test.routes if r.date.isoformat() == test_date]
    cust_pcs = {}
    cust_unloads = {}
    for r in test_routes_for_date:
        for c in r.customer_ids:
            cust_pcs[c] = r.pc_per_customer.get(c, 0.0)
        for d in r.delivery_rows:
            if d.unload_time:
                cust_unloads.setdefault(c, d.unload_time)

    print(f"=== Dispatch for {test_date} ({len(cust_pcs)} customers, {len(test_routes_for_date)} true routes) ===")
    result = pipe.dispatch_day(test_date, cust_pcs, cust_unloads)
    print(f"  Predicted routes: {len(result.routes)}")
    print(f"  Unseen customers: {result.n_unseen_customers}")
    print(f"  SOP-1 solo routes: {result.n_solo_routes}")
    print(f"  Capacity splits: {result.n_capacity_split}")
    print()
    print("  Sample routes:")
    for r in result.routes[:10]:
        custs_str = ", ".join(c[:6] for c in r.customers[:4])
        if r.n_customers > 4:
            custs_str += f" ... ({r.n_customers} total)"
        print(f"    {r.summary()} | customers: {custs_str}")


if __name__ == "__main__":
    demo()