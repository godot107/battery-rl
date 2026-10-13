"""Gymnasium environment: one hour per step, the agent sees only the past.

At hour t the agent knows prices up to t-1 (the real-time price for hour t settles after the
fact), the local hour/weekday/month, and its state of charge. It picks one of five setpoints
(full/half charge, idle, half/full discharge). Reward is the hour's profit in $100s.

Prices are heavy-tailed (2023 had hours above $5,000), so observations use a signed log
transform; rewards are left in dollars (scaled) so the agent optimises the real objective.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from . import battery, config

ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * config.BATTERY["power_mw"]
LAGS = 24
REWARD_SCALE = 100.0


def squash(p):
    return np.sign(p) * np.log1p(np.abs(p) / 25.0)


def features(prices: pd.Series) -> np.ndarray:
    """Per-hour context known *before* the hour starts: (T, LAGS + 7). Row t uses prices < t."""
    p = squash(prices.to_numpy())
    T = len(p)
    pad = np.concatenate([np.full(LAGS + 24 * 7, p[0]), p])   # warm-up uses the first price, never later ones
    lags = np.stack([pad[LAGS + 24 * 7 - k: LAGS + 24 * 7 - k + T] for k in range(1, LAGS + 1)], axis=1)
    same_hour = np.stack([pad[LAGS + 24 * 7 - 24 * d: LAGS + 24 * 7 - 24 * d + T] for d in range(1, 8)], axis=1)
    local = prices.index.tz_convert("America/Chicago")
    h, m = local.hour.to_numpy(), local.month.to_numpy() - 1
    cal = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), (local.dayofweek >= 5).astype(float),
                    np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)], axis=1)
    return np.concatenate([lags, same_hour.mean(1, keepdims=True), same_hour.std(1, keepdims=True), cal], axis=1).astype(np.float32)


class BatteryEnv(gym.Env):
    def __init__(self, prices: pd.Series, episode_hours=24 * 14, random_start=True, seed=config.SEED):
        super().__init__()
        self.price = prices.to_numpy(np.float64)
        self.feat = features(prices)
        self.episode_hours = min(episode_hours, len(self.price)) if episode_hours else len(self.price)
        self.random_start = random_start
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.feat.shape[1] + 1,), np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self._rng = np.random.default_rng(seed)

    def _obs(self):
        soc = self.soc / config.BATTERY["energy_mwh"]
        return np.append(self.feat[self.t], soc).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        hi = len(self.price) - self.episode_hours
        self.t0 = int(self._rng.integers(0, hi + 1)) if self.random_start and hi > 0 else 0
        self.t = self.t0
        self.soc = config.BATTERY["soc_init"] * config.BATTERY["energy_mwh"]
        return self._obs(), {}

    def step(self, action):
        self.soc, p, profit = battery.step(self.soc, ACTIONS[int(action)], self.price[self.t])
        self.t += 1
        done = self.t >= self.t0 + self.episode_hours
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, profit / REWARD_SCALE, done, False, {"p": p, "profit": profit}
