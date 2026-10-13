"""Non-learning policies: the ceiling, two honest baselines, and the information each one uses.

| policy | knows at hour t |
|---|---|
| perfect_foresight | every price in the period (an upper bound, not a strategy) |
| daily_oracle | today's 24 prices at midnight (roughly what a day-ahead market price gives) |
| mpc_persistence | yesterday's prices as today's forecast, re-planned each midnight |
| mpc_weekly | median of the same hour over the last 7 days, re-planned each midnight |
| time_of_day | the cheapest and dearest hours of the day, learned on the training years |

Everything is executed through battery.simulate, so all policies face the same physics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import battery, config

B = config.BATTERY


def local_days(prices: pd.Series):
    """Positions of each local calendar day (23-25 hours around DST)."""
    local = prices.index.tz_convert("America/Chicago")
    key = pd.Series(np.arange(len(prices)), index=local.date)
    return [g.to_numpy() for _, g in key.groupby(level=0)]


def perfect_foresight(prices: pd.Series):
    return battery.simulate(prices.to_numpy(), battery.lp_schedule(prices.to_numpy()))


def _daily(prices: pd.Series, forecast):
    """Re-plan at each local midnight against forecast(day_index) and run the day for real."""
    p = prices.to_numpy()
    days = local_days(prices)
    soc = B["soc_init"] * B["energy_mwh"]
    setpoints = np.zeros(len(p))
    for k, idx in enumerate(days):
        f = forecast(k, days, p)
        if f is None:
            continue
        plan = battery.lp_schedule(f, soc0=soc, soc_end=B["soc_init"] * B["energy_mwh"])
        setpoints[idx] = plan
        # advance the true SoC through the day (the plan may be clipped by reality)
        for t, sp in zip(idx, plan):
            soc, _, _ = battery.step(soc, sp, p[t])
    return battery.simulate(p, setpoints)


def _fit_len(x, n):
    x = np.asarray(x, float)
    return np.resize(x, n) if len(x) != n else x


def daily_oracle(prices):
    return _daily(prices, lambda k, days, p: p[days[k]])


def mpc_persistence(prices):
    return _daily(prices, lambda k, days, p: None if k == 0 else _fit_len(p[days[k - 1]], len(days[k])))


def mpc_weekly(prices):
    def f(k, days, p):
        if k < 7:
            return None
        hist = np.stack([_fit_len(p[days[j]], 24) for j in range(k - 7, k)])
        return _fit_len(np.median(hist, axis=0), len(days[k]))
    return _daily(prices, f)


class TimeOfDay:
    """Charge in the n cheapest local hours, discharge in the n dearest (means over training years)."""

    def __init__(self, train_prices: pd.Series, n=2):
        hour = train_prices.index.tz_convert("America/Chicago").hour
        mean = train_prices.groupby(hour).mean()
        self.charge = set(mean.nsmallest(n).index)
        self.discharge = set(mean.nlargest(n).index)

    def __call__(self, prices: pd.Series):
        hour = prices.index.tz_convert("America/Chicago").hour
        sp = np.where(np.isin(hour, list(self.discharge)), B["power_mw"],
                      np.where(np.isin(hour, list(self.charge)), -B["power_mw"], 0.0))
        return battery.simulate(prices.to_numpy(), sp)
