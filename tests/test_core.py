import numpy as np
import pandas as pd
import pytest

from battery_rl import battery, config, env, evaluate, policies

B = config.BATTERY


def test_step_respects_limits():
    soc, p, _ = battery.step(0.0, 1.0, 50.0)                 # empty: cannot discharge
    assert p == 0 and soc == 0
    soc, p, _ = battery.step(B["energy_mwh"], -1.0, 50.0)    # full: cannot charge
    assert p == 0 and soc == B["energy_mwh"]
    soc, p, _ = battery.step(B["energy_mwh"], 5.0, 50.0)    # clipped to power rating
    assert p == pytest.approx(B["power_mw"])


def test_round_trip_efficiency():
    soc, _, buy = battery.step(0.0, -1.0, 0.0)
    _, p, _ = battery.step(soc, 1.0, 0.0)
    assert p == pytest.approx(0.86, rel=1e-6)                # 1 MWh in -> 0.86 MWh out


def test_lp_buys_low_sells_high():
    prices = np.array([10, 10, 100, 100, 10, 10], float)
    sp = battery.lp_schedule(prices, soc0=0.0, soc_end=0.0)
    assert (sp[:2] < 0).all() and (sp[2:4] > 0).all()
    r = battery.simulate(prices, sp, soc0=0.0)
    assert r["total"] > 0


def test_lp_idles_when_spread_too_small():
    prices = np.array([50, 50, 52, 52], float)               # spread < losses + wear
    sp = battery.lp_schedule(prices, soc0=0.0, soc_end=0.0)
    assert np.allclose(sp, 0, atol=1e-6)


def _series(values, start="2024-06-01"):
    idx = pd.date_range(start, periods=len(values), freq="1h", tz="America/Chicago").tz_convert("UTC")
    return pd.Series(np.asarray(values, float), index=idx)


def test_perfect_foresight_beats_everyone():
    rng = np.random.default_rng(0)
    hours = np.arange(24 * 30)
    prices = _series(30 + 20 * np.sin(2 * np.pi * hours / 24) + rng.gamma(1, 5, len(hours)))
    pf = policies.perfect_foresight(prices)["total"]
    for pol in (policies.daily_oracle, policies.mpc_weekly, policies.mpc_persistence):
        assert pol(prices)["total"] <= pf + 1e-6


def test_features_use_only_the_past():
    hours = np.arange(24 * 10)
    a = _series(np.sin(hours))
    b = a.copy()
    b.iloc[100:] += 1000                                       # change the future from hour 100
    fa, fb = env.features(a), env.features(b)
    assert np.allclose(fa[:101], fb[:101])                     # row 100 only sees prices < 100
    assert not np.allclose(fa[101], fb[101])


def test_env_episode_runs():
    e = env.BatteryEnv(_series(np.full(24 * 20, 30.0)), episode_hours=48)
    obs, _ = e.reset(seed=0)
    total = 0
    for _ in range(48):
        obs, r, done, _, _ = e.step(2)                          # idle
        total += r
    assert done and total == 0


def test_month_bootstrap_sign():
    a = pd.Series(np.full(12, 10.0))
    ci, p = evaluate.paired_month_bootstrap(a, a - 1)
    assert ci[0] > 0 and p == 1.0
