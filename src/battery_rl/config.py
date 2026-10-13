"""Single source of truth: paths, market data, battery spec, and the time split."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

HUB = "HB_HUBAVG"                 # ERCOT hub average real-time settlement point price
YEARS = [2022, 2023, 2024, 2025]  # 2021 (Winter Storm Uri, $9,000 cap for days) left out deliberately
SPLIT = {"train": ("2022-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"),
         "test": ("2025-01-01", "2025-12-31")}

# A typical ERCOT utility-scale battery, per MW: 1 MW / 2 MWh, ~86% round trip.
BATTERY = {
    "power_mw": 1.0,
    "energy_mwh": 2.0,
    "eta_charge": 0.86 ** 0.5,
    "eta_discharge": 0.86 ** 0.5,
    "soc_init": 0.5,              # fraction of energy_mwh
    "degradation_usd_per_mwh": 10.0,   # wear cost per MWh discharged
}
DT_H = 1.0                        # hourly decisions on hourly-mean prices

SEED = 42
