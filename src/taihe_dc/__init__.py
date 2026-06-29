"""taihe-dc-rl: Auto-dispatch RL for 太古DC-style FMCG distribution centers.

This package implements the 8-SOP framework extracted from 太和DC historical
route data, applied to any FMCG distribution center with similar structure.

Public API:
    from taihe_dc import data, models, env
    routes = data.load_routes("data/raw/全流程报表2026.1.1-5.31.xlsx")
    sop_report = data.extract_sops(routes)
"""

from __future__ import annotations

__version__ = "0.1.0"