"""The battery: one simulator every policy is scored with, and the LP that schedules it.

Sign convention: a power setpoint p > 0 discharges (sells) p MW for an hour, p < 0 charges (buys).
Setpoints are clipped to what the state of charge allows, so no policy can cheat physics.

Profit per hour = price * (MWh sold - MWh bought) - degradation * MWh discharged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags, hstack, identity, lil_matrix

from . import config

B = config.BATTERY


def step(soc_mwh: float, p_mw: float, price: float, b=B, dt=config.DT_H):
    """Apply one setpoint. Returns (new_soc_mwh, delivered_mw, profit_usd)."""
    p = float(np.clip(p_mw, -b["power_mw"], b["power_mw"]))
    if p > 0:   # discharge: limited by energy stored
        p = min(p, soc_mwh * b["eta_discharge"] / dt)
        soc = soc_mwh - p * dt / b["eta_discharge"]
        profit = price * p * dt - b["degradation_usd_per_mwh"] * p * dt
    else:       # charge: limited by headroom
        p = max(p, -(b["energy_mwh"] - soc_mwh) / (b["eta_charge"] * dt))
        soc = soc_mwh - p * dt * b["eta_charge"]
        profit = price * p * dt
    return min(max(soc, 0.0), b["energy_mwh"]), p, profit


def simulate(prices, setpoints, b=B, soc0=None):
    """Run setpoints through the battery; returns dict of arrays and totals."""
    soc = b["soc_init"] * b["energy_mwh"] if soc0 is None else soc0
    n = len(prices)
    socs, ps, profits = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        soc, ps[t], profits[t] = step(soc, setpoints[t], prices[t], b)
        socs[t] = soc
    discharged = ps.clip(min=0).sum() * config.DT_H
    return {"soc": socs, "p": ps, "profit": profits, "total": float(profits.sum()),
            "cycles": float(discharged / b["energy_mwh"])}


def lp_schedule(prices, b=B, soc0=None, soc_end=None):
    """Profit-maximising schedule for known prices (HiGHS LP).

    Variables per hour: charge c, discharge d (MW, >= 0) and SoC s. Simultaneous charge and discharge
    is never optimal with losses and positive wear cost, except at negative prices where it burns
    energy; the simulator clips that, so it can only lower the LP's realised profit, not inflate it.
    """
    prices = np.asarray(prices, float)
    n, dt = len(prices), config.DT_H
    soc0 = b["soc_init"] * b["energy_mwh"] if soc0 is None else soc0
    # minimise  sum( price*c - (price - wear)*d ) * dt
    cost = np.concatenate([prices * dt, -(prices - b["degradation_usd_per_mwh"]) * dt, np.zeros(n)])
    # s_t - s_{t-1} - eta_c*c_t*dt + d_t*dt/eta_d = 0   (s_{-1} = soc0)
    I = identity(n, format="csr")
    shift = diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    A_eq = hstack([-b["eta_charge"] * dt * I, dt / b["eta_discharge"] * I, I - shift]).tocsr()
    b_eq = np.zeros(n)
    b_eq[0] = soc0
    if soc_end is not None:            # end at the same SoC so days are comparable
        row = lil_matrix((1, 3 * n))
        row[0, 3 * n - 1] = 1.0
        from scipy.sparse import vstack
        A_eq = vstack([A_eq, row.tocsr()]).tocsr()
        b_eq = np.append(b_eq, soc_end)
    bounds = [(0, b["power_mw"])] * (2 * n) + [(0, b["energy_mwh"])] * n
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c, d = res.x[:n], res.x[n:2 * n]
    return d - c
