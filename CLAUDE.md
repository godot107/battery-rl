# battery-rl

Trello #53 Reinforcement Learning. **Question: can a reinforcement-learning agent run a grid battery on
ERCOT real-time prices better than the forecast-and-optimise approach operators actually use?**
"No" is a valid answer.

## The problem

- **Battery:** 1 MW / 2 MWh, 86% round trip, $10/MWh wear cost on discharge (`config.BATTERY`).
- **Action (hourly):** one of 5 setpoints, from full charge to full discharge. The simulator clips what
  the state of charge can't deliver.
- **Reward:** that hour's profit = price x MWh sold - price x MWh bought - $10 x MWh discharged.
- **Observation:** prices before this hour only (24 lags, the same hour's 7-day mean and std), calendar,
  and the state of charge.
- **Prices:** ERCOT hub average (HB_HUBAVG) real-time, 15-minute prices averaged to hourly, via
  gridstatus (public MIS archive, no key). Split: train 2022-23, val 2024, test 2025. 2021 is excluded
  (Winter Storm Uri). Data goes in `data/` (gitignored).

## Policies, all run through `battery.simulate`

| Policy | Knows |
|---|---|
| `perfect_foresight` | every price in the period (LP, the ceiling) |
| `daily_oracle` | today's prices at midnight (roughly day-ahead) |
| `mpc_weekly` / `mpc_persistence` | a forecast (7-day same-hour median / yesterday), re-planned each midnight |
| `time_of_day` | the cheapest/dearest 2 hours, learned on train |
| `ppo` | PPO (stable-baselines3), checkpoint chosen on val profit |

## Run

```bash
source .venv/bin/activate          # own venv, CPU torch
pytest -q                          # 8 tests
python -m battery_rl fetch         # ~75 s per year
python -m battery_rl baselines --split val
python -m battery_rl train --run ppo-s42 --seed 42 --timesteps 2000000   # ~6 min, 1 thread
python -m battery_rl eval ppo-s42 --split val          # test only once, at the end
```

## Key decisions

- **No look-ahead:** feature row t uses prices < t (unit test). The feature warm-up once averaged the
  first week, which leaked; it now repeats the first price.
- **No pickle:** SB3's `save()` zip pickles objects, so the policy state_dict goes to safetensors plus
  `config.json`, and loading is strict.
- **Rewards in dollars (scaled 1/100), not clipped:** clipping would change the objective. Observations are
  signed-log squashed because prices reach $5,000.
- **One thread per run:** `rl.train` sets `torch.set_num_threads(1)`. Three seeds in parallel with default
  threads oversubscribed 12 cores and ran ~5x slower. Use `PYTHONUNBUFFERED=1` when logging to a file.
- **Test touched once (2025-10-13 run):** PPO 41-50% of perfect foresight vs forecast+LP 46%; see README.
