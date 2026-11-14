#!/usr/bin/env python3
"""Benchmark Batch rollouts using incremental, player-visible protocol views.

Each view is reconstructed only from that player's redacted log and request.
The engine projection is never passed to the tracker or policy-side adapter.
The script checks the tracker's fixed numeric encodings against Batch.observe
and also parses Tracker.view JSON to approximate the richer feature-adapter
cost. It intentionally uses random legal action rows, not PPO inference.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import ninjask


def run_trial(
    dex: ninjask.Dex,
    packed_teams: list[str],
    game_count: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    team1: list[str] = []
    team2: list[str] = []
    for _ in range(game_count):
        first = rng.randrange(len(packed_teams))
        second = rng.randrange(len(packed_teams) - 1)
        second += second >= first
        team1.append(packed_teams[first])
        team2.append(packed_teams[second])

    batch = ninjask.Batch(dex, game_count)
    batch.reset_many(
        list(range(game_count)),
        team1,
        team2,
        [seed + index for index in range(game_count)],
    )
    trackers = {
        (game, side): ninjask.Tracker(dex, side)
        for game in range(game_count)
        for side in ("p1", "p2")
    }
    cursors = {(game, side): 0 for game in range(game_count) for side in ("p1", "p2")}
    action_rngs = [random.Random(seed * 17 + index) for index in range(game_count)]
    decisions = 0
    parsed_views = 0
    view_json_bytes = 0
    encoding_mismatches = 0
    cycles = 0
    started = time.perf_counter()

    while True:
        observation = batch.observe()
        count = observation["n"]
        if not count:
            break
        who = np.frombuffer(observation["who"], dtype=np.uint32).reshape(count, 2)
        batch_obs = np.frombuffer(observation["obs"], dtype=np.float32).reshape(
            count, ninjask.OBS_LEN
        )
        batch_ids = np.frombuffer(observation["ids"], dtype=np.int16).reshape(
            count, ninjask.IDS_LEN
        )
        offsets = np.frombuffer(observation["offsets"], dtype=np.uint32)
        battles: dict[int, ninjask.Battle] = {}
        picks: list[int] = []

        for row, (game_value, side_value) in enumerate(who):
            game = int(game_value)
            side = "p1" if int(side_value) == 0 else "p2"
            battle = battles.get(game)
            if battle is None:
                # Battle is used only as a source of side-redacted log/request
                # methods. Its projection is never inspected or passed on.
                battle = batch.battle(game)
                battles[game] = battle
            key = (game, side)
            tracker = trackers[key]
            tracker.feed(battle.log(side, cursors[key]))
            cursors[key] = battle.log_len()
            request = battle.request(side)
            if request:
                tracker.feed_request(request)

            encoded_obs = np.frombuffer(tracker.encode(), dtype=np.float32)
            encoded_ids = np.frombuffer(tracker.encode_ids(), dtype=np.int16)
            if not np.array_equal(encoded_obs, batch_obs[row]) or not np.array_equal(
                encoded_ids, batch_ids[row]
            ):
                encoding_mismatches += 1
            raw_view = tracker.view()
            view_json_bytes += len(raw_view)
            json.loads(raw_view)
            parsed_views += 1

            available = int(offsets[row + 1] - offsets[row])
            picks.append(action_rngs[game].randrange(available))

        batch.step(picks)
        decisions += count
        cycles += 1

    elapsed = time.perf_counter() - started
    if encoding_mismatches:
        raise RuntimeError(f"player tracker disagreed with Batch buffers {encoding_mismatches} times")
    return {
        "games": game_count,
        "elapsed_seconds": elapsed,
        "games_per_second": game_count / elapsed,
        "decision_rows": decisions,
        "tracker_views_parsed": parsed_views,
        "view_json_megabytes": view_json_bytes / 1_000_000,
        "encoding_mismatches": encoding_mismatches,
        "decision_cycles": cycles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-teams", type=Path, required=True)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.games < 1 or args.trials < 1:
        parser.error("games and trials must be positive")
    teams = json.loads(args.packed_teams.read_text())
    dex = ninjask.Dex.load(str(args.dex))
    trials = [
        run_trial(dex, teams, args.games, args.seed + i * args.games)
        for i in range(args.trials)
    ]
    rates = [float(trial["games_per_second"]) for trial in trials]
    print(json.dumps({
        "backend": "Ninjask 0.4.0 Batch on CPU",
        "view_source": "incremental side-redacted Battle.log plus that player's request",
        "engine_projection_passed_to_tracker": False,
        "richer_json_view_parsed": True,
        "current_policy_inference_included": False,
        "ppo_update_included": False,
        "setup_reset_included": False,
        "trials": trials,
        "median_games_per_second": statistics.median(rates),
        "min_games_per_second": min(rates),
        "max_games_per_second": max(rates),
        "scope_limit": "Player-view rollout and JSON conversion benchmark only; not PPO throughput or a strength result.",
    }, indent=2))


if __name__ == "__main__":
    main()
