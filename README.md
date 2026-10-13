# Can reinforcement learning run a grid battery better than forecast + optimise?

A 1 MW / 2 MWh battery trades ERCOT real-time energy (hub average, hourly). Each hour it charges, idles or
discharges; profit = sales - purchases - $10/MWh wear, with 86% round-trip efficiency. A PPO agent that sees
only past prices is compared with what an operator would actually run: forecast tomorrow, solve a linear
program, re-plan every midnight.

## Result (2025 test year, evaluated once; trained 2022-23, checkpoint chosen on 2024)

| Policy | Profit ($/MW-yr) | % of perfect foresight | Cycles |
|---|---|---|---|
| Perfect foresight (LP, the ceiling) | 37,200 | 100% | 408 |
| Daily oracle (knows today's prices at midnight) | 34,162 | 92% | 441 |
| PPO, seed 42 | 18,475 | 50% | 299 |
| PPO, seed 43 | 17,877 | 48% | 294 |
| **Forecast + LP (7-day same-hour median)** | **17,257** | **46%** | 392 |
| PPO, seed 44 | 15,114 | 41% | 282 |
| Forecast + LP (yesterday's prices) | 13,798 | 37% | 440 |
| Time-of-day rule (learned on 2022-23) | 1,925 | 5% | 314 |

**PPO vs the best forecast baseline, paired monthly block bootstrap (90% CI of the annual difference):**
seed 42 -$2.0k to +$4.0k (P(better) 0.76); seed 43 -$2.4k to +$3.1k (0.66); seed 44 -$5.3k to +$1.0k (0.13).

**Reading:**
- **RL ties a simple forecast + optimiser.** It doesn't beat it.
- **It isn't stable across seeds.** The spread between seeds (about $3k) is as large as the effect being measured.
- **It cycles about 25% less for similar profit**, which matters once battery wear costs more than $10/MWh.
- **The money is in information.** Knowing the day's prices at midnight is worth 92% of the ceiling, and good forecasting is worth 46%. Closing that gap is a *forecasting* problem; a different decision rule does little.
- **The time-of-day rule collapsed in 2025** (5%), while it got 24% in 2024. Solar moved the cheap and expensive hours, so fixed rules learned on 2022-23 go stale.

## Method

- **Prices:** ERCOT HB_HUBAVG real-time settlement prices via gridstatus, from ERCOT's public archive with no key.
  2021 is excluded (Winter Storm Uri). Split: train 2022-23, validation 2024, test 2025.
- **One simulator** (`battery.simulate`) scores every policy; setpoints the battery can't physically deliver are clipped.
- **No look-ahead:** the agent's features at hour t use prices before t only, and a unit test enforces it.
- **PPO** (stable-baselines3):
  - 5 discrete setpoints, 14-day episodes, 2M steps.
  - The checkpoint is chosen by validation profit, 3 seeds, about 6 minutes each on CPU.
  - Weights are saved as safetensors, never as SB3's pickled zip.

## Run

See `CLAUDE.md`.
