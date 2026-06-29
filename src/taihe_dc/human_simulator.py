"""Human Dispatcher Simulator.

Goal: AI simulates human daily vehicle planning, with load capacity (PC)
as the only hard constraint.

Input:  daily customer orders (customer_id → PC)
Output: vehicle plan (groups of customers, each capacity-compliant)

Method:
  1. Learn human grouping patterns from history (Louvain community detection)
  2. For each new day, assign customers to their learned communities
  3. Enforce capacity: PC > solo_threshold → solo route; bin pack rest
  4. NO time window, NO vehicle type, NO ordering — just capacity

Evaluation:
  - ARI vs human's actual plan (how well AI mimics human)
  - Capacity compliance (0% overload = perfect)
  - Route count match (AI plan size vs human plan size)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from taihe_dc.data import Route
from taihe_dc.baselines.community_louvain import build_cooccurrence_graph, detect_communities
from taihe_dc.baselines.community_with_capacity import (
    ROUTE_PC_CAP,
    SINGLE_CUSTOMER_PC_THRESHOLD,
    _greedy_bin_pack,
)
from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters, HardModeMetrics, format_hard_mode


@dataclass
class VehiclePlan:
    """AI's vehicle plan for one day — simulates human dispatcher output."""
    date: date
    routes: tuple["PlannedRoute", ...]
    n_customers: int
    total_pc: float
    n_overload_routes: int                    # routes exceeding ROUTE_PC_CAP
    max_route_pc: float

    @property
    def capacity_compliance(self) -> float:
        """Fraction of routes that respect capacity (PC <= cap)."""
        if not self.routes:
            return 1.0
        ok = sum(1 for r in self.routes if r.pc_total <= ROUTE_PC_CAP)
        return ok / len(self.routes)


@dataclass
class PlannedRoute:
    """One planned vehicle route."""
    customers: tuple[str, ...]
    pc_total: float
    pc_per_customer: dict[str, float]
    is_solo: bool                             # SOP-1 forced solo (PC > threshold)
    is_overload: bool                         # exceeds capacity even as solo


    def __repr__(self) -> str:
        tag = " [SOLO]" if self.is_solo else ""
        ovf = " [OVERLOAD!]" if self.is_overload else ""
        return f"Route({len(self.customers)}cust, PC={self.pc_total:.0f}{tag}{ovf})"


@dataclass
class HumanSimulator:
    """AI that simulates human daily vehicle planning.

    Trains on historical routes to learn community structure,
    then plans new days respecting capacity.
    """

    partition: dict[str, int] = field(default_factory=dict)
    route_pc_cap: float = ROUTE_PC_CAP
    solo_threshold: float = SINGLE_CUSTOMER_PC_THRESHOLD
    n_communities: int = 0

    @classmethod
    def fit(cls, train_routes: list[Route], min_weight: int = 2) -> "HumanSimulator":
        """Learn human grouping patterns from history."""
        G = build_cooccurrence_graph(train_routes, min_weight=min_weight, use_pmi=True)
        partition = detect_communities(G, resolution=1.0)
        return cls(
            partition=partition,
            n_communities=len(set(partition.values())),
        )

    def plan_day(self, target_date: date, customer_pcs: dict[str, float]) -> VehiclePlan:
        """Simulate human dispatcher: plan vehicles for one day.

        Only constraint: load capacity (PC).
        """
        # Group by community (learned from human history)
        comm_groups: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for c, pc in customer_pcs.items():
            comm_id = self.partition.get(c)
            if comm_id is None:
                comm_id = -(hash((str(target_date), c)) % (10**9))
            comm_groups[comm_id].append((c, pc))

        planned: list[PlannedRoute] = []
        n_overload = 0
        max_pc = 0.0

        for comm_id, members in comm_groups.items():
            # SOP-1: big PC customers go solo
            solo = [(c, pc) for c, pc in members if pc > self.solo_threshold]
            group = [(c, pc) for c, pc in members if pc <= self.solo_threshold]

            for c, pc in solo:
                overload = pc > self.route_pc_cap
                if overload:
                    n_overload += 1
                max_pc = max(max_pc, pc)
                planned.append(PlannedRoute(
                    customers=(c,), pc_total=pc,
                    pc_per_customer={c: pc}, is_solo=True, is_overload=overload,
                ))

            if not group:
                continue

            total_pc = sum(pc for _, pc in group)
            if total_pc <= self.route_pc_cap:
                max_pc = max(max_pc, total_pc)
                overload = False
                planned.append(self._make_route(group, solo=False, overload=False))
            else:
                # Bin pack by capacity
                bins = _greedy_bin_pack(group, self.route_pc_cap)
                for bin_cids in bins:
                    bin_items = [(c, dict(group)[c]) for c in bin_cids]
                    bin_pc = sum(pc for _, pc in bin_items)
                    overload = bin_pc > self.route_pc_cap
                    if overload:
                        n_overload += 1
                    max_pc = max(max_pc, bin_pc)
                    planned.append(self._make_route(bin_items, solo=False, overload=overload))

        total = sum(r.pc_total for r in planned)
        return VehiclePlan(
            date=target_date,
            routes=tuple(planned),
            n_customers=len(customer_pcs),
            total_pc=total,
            n_overload_routes=n_overload,
            max_route_pc=max_pc,
        )

    def _make_route(self, items, solo: bool, overload: bool) -> PlannedRoute:
        cids = tuple(c for c, _ in items)
        pc_map = {c: pc for c, pc in items}
        total = sum(pc for _, pc in items)
        return PlannedRoute(
            customers=cids, pc_total=total, pc_per_customer=pc_map,
            is_solo=solo, is_overload=overload,
        )

    def evaluate_against_human(
        self,
        human_routes: list[Route],
    ) -> dict:
        """Compare AI plans vs human plans across all dates.

        Returns:
          - hard_mode: ARI/F1 (how well AI mimics human grouping)
          - capacity: AI + human compliance rates
          - route_count_match: AI vs human route counts per day
        """
        # Group human routes by date
        human_by_date: dict[str, list[Route]] = defaultdict(list)
        for r in human_routes:
            human_by_date[r.date.isoformat()].append(r)

        # AI plans
        ai_date_to_clusters: dict[str, dict[str, int]] = {}
        ai_plans: dict[str, VehiclePlan] = {}
        for date_str, day_routes in human_by_date.items():
            cust_pcs = {}
            for r in day_routes:
                for c in r.customer_ids:
                    cust_pcs[c] = r.pc_per_customer.get(c, 0.0)
            plan = self.plan_day(date.fromisoformat(date_str), cust_pcs)
            ai_plans[date_str] = plan

            # Convert to cluster format for ARI
            cluster_map: dict[str, int] = {}
            for i, route in enumerate(plan.routes):
                for c in route.customers:
                    cluster_map[c] = i
            ai_date_to_clusters[date_str] = cluster_map

        # Hard mode eval
        preds = PredictedClusters(date_to_clusters=ai_date_to_clusters)
        hard = hard_mode_eval(human_routes, preds)

        # Capacity stats
        ai_compliance = sum(p.capacity_compliance for p in ai_plans.values()) / max(1, len(ai_plans))
        ai_avg_overload = sum(p.n_overload_routes for p in ai_plans.values()) / max(1, len(ai_plans))
        ai_route_counts = [len(p.routes) for p in ai_plans.values()]
        human_route_counts = [len(rs) for rs in human_by_date.values()]

        return {
            "hard_mode": hard,
            "ai_capacity_compliance": ai_compliance,
            "ai_avg_overload_routes_per_day": ai_avg_overload,
            "ai_avg_routes_per_day": sum(ai_route_counts) / max(1, len(ai_route_counts)),
            "human_avg_routes_per_day": sum(human_route_counts) / max(1, len(human_route_counts)),
            "n_dates": len(ai_plans),
        }


def demo():
    """Full demo: AI simulates human, evaluated on test set."""
    import warnings
    warnings.filterwarnings("ignore")

    from taihe_dc.data import load_dataset
    from taihe_dc.split import chronological_split

    ds = load_dataset("data/raw/全流程报表2026.1.1-5.31.xlsx")
    splits = chronological_split(ds, train_frac=0.70, val_frac=0.10)
    train, test = list(splits.train.routes), list(splits.test.routes)

    print("═" * 60)
    print("  Human Dispatcher Simulator")
    print("  Goal: AI simulates human daily vehicle planning")
    print("  Constraint: load capacity (PC) only")
    print("═" * 60)
    print()

    # Train
    sim = HumanSimulator.fit(train)
    print(f"Trained on {len(train):,} historical routes")
    print(f"Learned {sim.n_communities} customer communities")
    print(f"Capacity: solo if PC > {sim.solo_threshold:.0f}, route cap = {sim.route_pc_cap:.0f}")
    print()

    # Evaluate on test
    print(f"Evaluating on test ({len(test)} routes, {splits.test.date_range[0]} → {splits.test.date_range[1]})...")
    results = sim.evaluate_against_human(test)

    hard = results["hard_mode"]
    print()
    print("─── AI vs Human Similarity ───")
    print(f"  ARI (clustering match):    {hard.ari:.3f}")
    print(f"  Partition F1:              {hard.partition_f1:.1%}")
    print(f"  Recall:                    {hard.partition_recall:.1%}")
    print(f"  Precision:                 {hard.partition_precision:.1%}")
    print()
    print("─── Capacity Compliance ───")
    print(f"  AI capacity compliance:    {results['ai_capacity_compliance']:.1%}")
    print(f"  AI avg overload routes/day: {results['ai_avg_overload_routes_per_day']:.1f}")
    print()
    print("─── Route Count Match ───")
    print(f"  AI avg routes/day:         {results['ai_avg_routes_per_day']:.1f}")
    print(f"  Human avg routes/day:      {results['human_avg_routes_per_day']:.1f}")
    print()

    # Show one day side-by-side
    test_date = splits.test.routes[0].date
    day_human = [r for r in test if r.date == test_date]
    cust_pcs = {}
    for r in day_human:
        for c in r.customer_ids:
            cust_pcs[c] = r.pc_per_customer.get(c, 0.0)
    ai_plan = sim.plan_day(test_date, cust_pcs)

    print(f"─── Sample Day: {test_date} ───")
    print(f"  Customers: {len(cust_pcs)}")
    print(f"  Human routes: {len(day_human)}")
    print(f"  AI routes:    {len(ai_plan.routes)}")
    print(f"  AI capacity compliance: {ai_plan.capacity_compliance:.1%}")
    print()
    print("  AI planned routes:")
    for r in ai_plan.routes[:8]:
        custs = ", ".join(c[:6] for c in r.customers[:3])
        if len(r.customers) > 3:
            custs += f" +{len(r.customers)-3}"
        print(f"    {r} | {custs}")
    if len(ai_plan.routes) > 8:
        print(f"    ... ({len(ai_plan.routes) - 8} more)")


if __name__ == "__main__":
    demo()