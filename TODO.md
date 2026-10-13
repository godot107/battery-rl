# TODO

## Done (2026-10-13)
- [x] Data 2022-2025, simulator, LP ceiling, daily oracle, two MPCs, time-of-day rule
- [x] PPO, 3 seeds x 2M steps; test 2025 evaluated once: RL ties forecast+LP (see README)

## Next
- [ ] Figures: monthly profit by policy; a week of prices with each policy's SoC; cumulative profit
- [ ] Better forecast for MPC (gradient boosting on lags + calendar + ERCOT load/solar forecasts):
      the bigger lever according to the results
- [ ] Day-ahead prices (`get_dam_spp`): a realistic "daily oracle" (DAM known the day before) and the
      DAM/RT spread as a feature
- [ ] RL with uncertainty: add a forecast as an observation (RL on top of MPC, not instead of it)
- [ ] Sensitivity: wear cost $0-$40/MWh, 4-hour battery, 15-minute decisions
- [ ] Blog draft, textbook-kb review (RL / optimisation books), push to GitHub **private**

Note: once these changes are made, the 2025 test number is no longer untouched. Report new variants on
2024 (val), or hold out 2026 when the data exists.
