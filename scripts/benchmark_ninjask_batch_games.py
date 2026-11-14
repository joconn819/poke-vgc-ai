#!/usr/bin/env python3
"""Measure Ninjask Batch throughput with random legal choices.

This isolates the battle/observation/action loop. It does not include neural
policy inference, PPO updates, data writes, or model-feature conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import ninjask


def run_trial(
    *,
    dex: ninjask.Dex,
    packed_teams: list[str],
    game_count: int,
    seed: int,
) -> dict:
    team_rng = random.Random(seed)
    pairs = [
        (team_rng.randrange(len(packed_teams)), team_rng.randrange(len(packed_teams)))
        for _ in range(game_count)
    ]
    batch = ninjask.Batch(dex, game_count)
    batch.reset_many(
        list(range(game_count)),
        [packed_teams[a] for a, _ in pairs],
        [packed_teams[b] for _, b in pairs],
        [seed + index for index in range(game_count)],
    )
    rngs = [random.Random(seed * 17 + index) for index in range(game_count)]
    decisions = 0
    decision_cycles = 0
    choice_refusals = 0
    started = time.perf_counter()
    while True:
        observation = batch.observe()
        count = observation["n"]
        if count == 0:
            break
        who = np.frombuffer(observation["who"], dtype=np.uint32).reshape(count, 2)
        offsets = np.frombuffer(observation["offsets"], dtype=np.uint32)
        picks = []
        for row, (game, _side) in enumerate(who):
            available = int(offsets[row + 1] - offsets[row])
            picks.append(rngs[int(game)].randrange(available))
        statuses = np.frombuffer(batch.step(picks), dtype=np.uint8)
        choice_refusals += sum(bool(status & 1) for status in statuses)
        decisions += count
        decision_cycles += 1
    elapsed = time.perf_counter() - started
    winners = [batch.winner(index) for index in range(game_count)]
    turns = [batch.turn(index) for index in range(game_count)]
    return {
        "games": game_count,
        "elapsed_seconds": elapsed,
        "games_per_second": game_count / elapsed,
        "decisions": decisions,
        "decision_cycles": decision_cycles,
        "choice_refusals": choice_refusals,
        "p1_wins": winners.count("p1"),
        "p2_wins": winners.count("p2"),
        "draws": winners.count(None),
        "mean_turns": statistics.mean(turns) if turns else 0.0,
        "max_turns": max(turns, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-teams", type=Path, required=True)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.games < 1 or args.trials < 1:
        parser.error("games and trials must be positive")
    packed_teams = json.loads(args.packed_teams.read_text())
    dex = ninjask.Dex.load(str(args.dex))
    trials = [
        run_trial(
            dex=dex,
            packed_teams=packed_teams,
            game_count=args.games,
            seed=args.seed + index * args.games,
        )
        for index in range(args.trials)
    ]
    rates = [trial["games_per_second"] for trial in trials]
    print(json.dumps({
        "backend": "Ninjask Batch, CPU",
        "policy": "uniform random legal choice index from the batched action rows",
        "torch_or_jax_inference_included": False,
        "training_updates_included": False,
        "data_writes_included": False,
        "python_cpu_count": os.cpu_count(),
        "trials": trials,
        "median_games_per_second": statistics.median(rates),
        "min_games_per_second": min(rates),
        "max_games_per_second": max(rates),
        "scope_limit": "Microbenchmark of Ninjask observe/step with random policies; not end-to-end training throughput or strength.",
    }, indent=2))


if __name__ == "__main__":
    main()
