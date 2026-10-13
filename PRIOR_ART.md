# Prior art

Verify citations before quoting them in the blog.

- **Wang & Zhang (2018), *Energy storage arbitrage in real-time markets via reinforcement learning*
  (IEEE PES GM).** Q-learning on NYISO real-time prices; compared mainly with a simple threshold rule.
- **Cao et al. (2020), *Deep reinforcement learning-based energy storage arbitrage with accurate lithium-ion
  battery degradation model* (IEEE Trans. Smart Grid).** Noisy-net DQN with a degradation model on UK prices.
- **The industry standard is not RL:** operators schedule with optimisation (LP/MILP) against price
  forecasts, re-planned on a rolling horizon. ERCOT batteries also earn most revenue from ancillary
  services, which this project leaves out (energy arbitrage only).

**Common weakness:** papers often report RL against perfect foresight and a naive rule, but not against
forecast + optimisation (MPC), the thing a real desk would run. Some evaluate on the training period.

**Our angle:**
1. **Strong, honest baselines:** perfect foresight (ceiling), a daily oracle (known day-ahead prices),
   two forecast-driven MPCs, and a time-of-day rule. RL has to beat the MPCs to matter.
2. **Strict information sets:** the agent sees only past prices; a unit test proves features don't use
   the future.
3. **Time split:** train 2022-2023, choose the checkpoint on 2024, test once on 2025.
4. **Lumpy revenue made visible:** the share of profit from the top 1% of hours, and a paired monthly block
   bootstrap for "RL minus best baseline".
5. **Several seeds:** RL results vary a lot by seed; report the spread, not the best one.
