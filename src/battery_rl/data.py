"""ERCOT real-time prices via gridstatus (ERCOT's public MIS archive, no API key).

Each year is fetched once (~75 s), reduced to the hub-average 15-minute price and saved as
data/rtm_<year>.parquet. `hourly()` averages to hourly prices on a UTC index, so DST days don't
produce duplicate or missing hours.
"""
from __future__ import annotations

import pandas as pd

from . import config


def fetch(years=None):
    import gridstatus
    ercot = gridstatus.Ercot()
    config.DATA.mkdir(parents=True, exist_ok=True)
    for y in years or config.YEARS:
        out = config.DATA / f"rtm_{y}.parquet"
        if out.exists():
            continue
        df = ercot.get_rtm_spp(year=y)
        df = df[df["Location"] == config.HUB][["Interval Start", "SPP"]]
        df = df.rename(columns={"Interval Start": "start", "SPP": "price"})
        df["start"] = pd.to_datetime(df["start"], utc=True)
        df.to_parquet(out, index=False)
        print(f"{y}: {len(df)} intervals -> {out.name}")


def hourly() -> pd.Series:
    """Hourly mean hub price ($/MWh), UTC index, all fetched years."""
    parts = [pd.read_parquet(config.DATA / f"rtm_{y}.parquet") for y in config.YEARS]
    df = pd.concat(parts).drop_duplicates("start").set_index("start").sort_index()
    s = df["price"].resample("1h").mean()
    return s.interpolate(limit=3).rename("price")


def split(prices: pd.Series, name: str) -> pd.Series:
    """Calendar split in Texas local time (a 'day' is a local day)."""
    a, b = config.SPLIT[name]
    local = prices.index.tz_convert("America/Chicago")
    m = (local >= pd.Timestamp(a, tz="America/Chicago")) & (local < pd.Timestamp(b, tz="America/Chicago") + pd.Timedelta(days=1))
    return prices[m]
