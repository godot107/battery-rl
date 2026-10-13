"""python -m battery_rl <command>  (see CLAUDE.md)"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(prog="battery_rl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="ERCOT hub-average real-time prices, 2022-2025 (~75 s per year)")
    b = sub.add_parser("baselines", help="perfect foresight, daily oracle, MPC, time-of-day")
    b.add_argument("--split", default="val", choices=["train", "val", "test"])
    t = sub.add_parser("train", help="PPO on train years, checkpoint chosen on the validation year")
    t.add_argument("--run", default="ppo")
    t.add_argument("--timesteps", type=int)
    t.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("eval", help="all baselines + the given RL runs on a split")
    e.add_argument("runs", nargs="*")
    e.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = ap.parse_args()

    if args.cmd == "fetch":
        from .data import fetch
        fetch()
    elif args.cmd == "baselines":
        from .evaluate import run_all
        run_all(args.split)
    elif args.cmd == "train":
        from . import data, rl
        p = data.hourly()
        kw = {"total_timesteps": args.timesteps} if args.timesteps else {}
        rl.train(data.split(p, "train"), data.split(p, "val"), args.run, args.seed, **kw)
    elif args.cmd == "eval":
        from .evaluate import run_all
        run_all(args.split, tuple(args.runs))


if __name__ == "__main__":
    main()
