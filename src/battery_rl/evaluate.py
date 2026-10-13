"""Score every policy on one split; compare against the perfect-foresight ceiling.

Battery revenue in ERCOT is lumpy: a handful of scarcity hours can carry a year. So beyond the
total we report the share of profit from the top 1% of hours, monthly profits, and a paired
monthly block bootstrap for "policy A minus policy B" (hours inside a month are not independent).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, data, policies


def monthly(prices, result):
    local = prices.index.tz_convert("America/Chicago").tz_localize(None)
    return pd.Series(result["profit"], index=local).groupby(local.to_period("M")).sum()


def paired_month_bootstrap(a: pd.Series, b: pd.Series, n_boot=5000, seed=config.SEED):
    """90% interval for the annual profit difference a - b, resampling whole months."""
    d = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), (n_boot, len(d)))].sum(1)
    return [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))], float((draws > 0).mean())


def top_share(result, q=0.99):
    pr = np.asarray(result["profit"])
    total = pr.sum()
    if total <= 0:
        return float("nan")
    cut = np.quantile(pr, q)
    return float(pr[pr >= cut].sum() / total)


def run_all(split="test", rl_runs=()):
    prices = data.split(data.hourly(), split)
    train = data.split(data.hourly(), "train")
    results = {
        "perfect_foresight": policies.perfect_foresight(prices),
        "daily_oracle": policies.daily_oracle(prices),
        "mpc_weekly": policies.mpc_weekly(prices),
        "mpc_persistence": policies.mpc_persistence(prices),
        "time_of_day": policies.TimeOfDay(train)(prices),
        "idle": {"profit": np.zeros(len(prices)), "total": 0.0, "cycles": 0.0},
    }
    for run in rl_runs:
        from . import rl
        model, _ = rl.load(run, prices)
        results[run] = rl.run_policy(model, prices)
    ceiling = results["perfect_foresight"]["total"]
    months = {k: monthly(prices, r) for k, r in results.items()}
    best_baseline = max(("mpc_weekly", "mpc_persistence", "time_of_day"), key=lambda k: results[k]["total"])
    rows = {}
    for k, r in results.items():
        row = {"profit_usd": r["total"], "pct_of_perfect": r["total"] / ceiling, "cycles": r["cycles"],
               "top1pct_hours_share": top_share(r) if k != "idle" else float("nan"),
               "monthly": {str(m): float(v) for m, v in months[k].items()}}
        if k in rl_runs:
            ci, p_better = paired_month_bootstrap(months[k], months[best_baseline])
            row["vs_best_baseline"] = {"baseline": best_baseline, "diff_ci90": ci, "p_better": p_better}
        rows[k] = row
    out = {"split": split, "hours": len(prices), "start": str(prices.index[0]), "end": str(prices.index[-1]),
           "price_mean": float(prices.mean()), "price_max": float(prices.max()), "battery": config.BATTERY,
           "policies": rows}
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / f"{split}.json").write_text(json.dumps(out, indent=1))
    print(table(out))
    return out


def table(out):
    lines = [f"{out['split']}: {out['hours']} hours, mean ${out['price_mean']:.1f}/MWh, max ${out['price_max']:,.0f}",
             "| Policy | Profit ($/MW-yr) | % of perfect foresight | Cycles | Top-1% hours' share |", "|---|---|---|---|---|"]
    for k, r in sorted(out["policies"].items(), key=lambda kv: -kv[1]["profit_usd"]):
        lines.append(f"| {k} | {r['profit_usd']:,.0f} | {r['pct_of_perfect']:.0%} | {r['cycles']:.0f} | {r['top1pct_hours_share']:.0%} |")
    return "\n".join(lines)
