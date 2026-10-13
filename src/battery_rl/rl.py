"""PPO agent (stable-baselines3), model selection on the validation year, safetensors weights.

SB3's own `save()` writes a zip that contains pickled objects, so we don't use it: the policy's
state_dict goes to safetensors next to a JSON config, and loading rebuilds the policy from that
config and loads the tensors strictly.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from . import battery, config, env

HP = {"total_timesteps": 1_000_000, "n_steps": 2048, "n_envs": 8, "batch_size": 512, "learning_rate": 3e-4,
      "gamma": 0.995, "ent_coef": 0.01, "net_arch": [128, 128], "episode_hours": 24 * 14, "eval_every": 100_000}


def _ppo(vec_env, hp, seed):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", vec_env, n_steps=hp["n_steps"], batch_size=hp["batch_size"],
               learning_rate=hp["learning_rate"], gamma=hp["gamma"], ent_coef=hp["ent_coef"],
               policy_kwargs={"net_arch": hp["net_arch"]}, seed=seed, device="cpu", verbose=0)


def run_policy(policy, prices):
    """Deterministic pass over a whole period through the shared simulator."""
    e = env.BatteryEnv(prices, episode_hours=None, random_start=False)
    obs, _ = e.reset()
    setpoints = np.empty(len(prices))
    for t in range(len(prices)):
        with torch.no_grad():
            a, _ = policy.predict(obs, deterministic=True)
        setpoints[t] = env.ACTIONS[int(a)]
        obs, _, done, _, _ = e.step(a)
        if done:
            break
    return battery.simulate(prices.to_numpy(), setpoints)


def train(train_prices, val_prices, run="ppo", seed=config.SEED, **overrides):
    from stable_baselines3.common.env_util import make_vec_env
    hp = {**HP, **overrides}
    torch.manual_seed(seed)
    vec = make_vec_env(lambda: env.BatteryEnv(train_prices, hp["episode_hours"]), n_envs=hp["n_envs"], seed=seed)
    model = _ppo(vec, hp, seed)
    best, history, t0 = -np.inf, [], time.time()
    done = 0
    while done < hp["total_timesteps"]:
        model.learn(hp["eval_every"], reset_num_timesteps=False)
        done += hp["eval_every"]
        val = run_policy(model, val_prices)
        history.append({"timesteps": done, "val_profit": val["total"], "val_cycles": val["cycles"],
                        "seconds": time.time() - t0})
        print(f"[{run}] {done:>9,d} steps  val profit ${val['total']:,.0f}  cycles {val['cycles']:.0f}  ({time.time() - t0:.0f}s)")
        if val["total"] > best:                     # keep the checkpoint that does best on val
            best = val["total"]
            save(model, run, {"hp": hp, "seed": seed, "selected_at": done, "val_profit": best})
    meta = json.loads((config.MODELS / run / "config.json").read_text())
    meta["history"] = history
    (config.MODELS / run / "config.json").write_text(json.dumps(meta, indent=1))
    return meta


def save(model, run, meta):
    d = config.MODELS / run
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.policy.state_dict().items()}, d / "policy.safetensors")
    (d / "config.json").write_text(json.dumps(meta, indent=1))


def load(run, prices):
    """Rebuild PPO from config.json, load policy tensors strictly (no pickle)."""
    from stable_baselines3.common.env_util import make_vec_env
    d = config.MODELS / run
    meta = json.loads((d / "config.json").read_text())
    vec = make_vec_env(lambda: env.BatteryEnv(prices, meta["hp"]["episode_hours"]), n_envs=1)
    model = _ppo(vec, meta["hp"], meta["seed"])
    model.policy.load_state_dict(load_file(d / "policy.safetensors"), strict=True)
    return model, meta
