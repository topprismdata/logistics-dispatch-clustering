"""Time-based train/val/test split for the dispatch problem.

Critical rule (NotebookLM insight):
  NEVER use random split for logistics data — strong temporal dependencies.
  Use Out-of-Time (OOT) split.

Strategy:
  - Training: first 70% of dates
  - Validation: next 10%
  - Test: last 20%

Per-route assignment to splits: based on route.date.
Each split contains all routes from its date range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from taihe_dc.data import DispatchDataset, Route


@dataclass(frozen=True)
class Split:
    """Train/val/test split result."""

    name: str                            # "train" / "val" / "test"
    start_date: date
    end_date: date
    routes: tuple[Route, ...]

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def n_customers(self) -> int:
        seen: set[str] = set()
        for r in self.routes:
            for cid in r.customer_ids:
                seen.add(cid)
        return len(seen)


@dataclass(frozen=True)
class DataSplits:
    """Bundle of train/val/test."""

    train: Split
    val: Split
    test: Split

    def summary(self) -> str:
        lines = []
        for s in [self.train, self.val, self.test]:
            lines.append(f"{s.name:5s} {s.start_date} → {s.end_date}: {s.n_routes:5d} routes, {s.n_customers:5d} customers")
        return "\n".join(lines)


def chronological_split(
    ds: DispatchDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> DataSplits:
    """Split by route.date in chronological order.

    train_frac + val_frac should be < 1.0; remainder is test.
    """
    if not ds.routes:
        empty = Split("train", date(1970, 1, 1), date(1970, 1, 1), ())
        return DataSplits(train=empty, val=empty, test=empty)

    dates_sorted = sorted({r.date for r in ds.routes})
    n_dates = len(dates_sorted)
    train_end_idx = max(1, int(n_dates * train_frac))
    val_end_idx = max(train_end_idx + 1, int(n_dates * (train_frac + val_frac)))

    train_dates = set(dates_sorted[:train_end_idx])
    val_dates = set(dates_sorted[train_end_idx:val_end_idx])
    test_dates = set(dates_sorted[val_end_idx:])

    def _filter(dates: set[date]) -> tuple[Route, ...]:
        return tuple(r for r in ds.routes if r.date in dates)

    return DataSplits(
        train=Split("train", dates_sorted[0], dates_sorted[train_end_idx - 1], _filter(train_dates)),
        val=Split("val", dates_sorted[train_end_idx], dates_sorted[val_end_idx - 1], _filter(val_dates)),
        test=Split("test", dates_sorted[val_end_idx], dates_sorted[-1], _filter(test_dates)),
    )


def split_dates(start: date, end: date, train_frac: float, val_frac: float) -> tuple[tuple[date, date], tuple[date, date], tuple[date, date]]:
    """Helper: explicit date ranges for reproducibility."""
    total_days = (end - start).days + 1
    train_days = int(total_days * train_frac)
    val_days = int(total_days * val_frac)
    train_range = (start, start + timedelta(days=train_days - 1))
    val_range = (train_range[1] + timedelta(days=1), train_range[1] + timedelta(days=val_days))
    test_range = (val_range[1] + timedelta(days=1), end)
    return train_range, val_range, test_range