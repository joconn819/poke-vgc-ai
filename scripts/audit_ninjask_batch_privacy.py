#!/usr/bin/env python3
"""Audit Ninjask Batch's encoded p1 view under hidden opponent-set changes.

For paired battles, opponent team-preview order zero brings positions 1-4, so
only the hidden fifth and sixth sets change and remain unselected. At every
decision, compare p1's numeric observations, categorical IDs, and legal
choice rows. P2's own observation is intentionally allowed to differ because
the mutated sets belong to P2.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import ninjask
from ninjask.refsim import seed_from_int


def mutate_unselected_bench(packed: str) -> str:
    members = packed.split("]")
    for index in range(4, len(members)):
        if not members[index]:
            continue
        fields = members[index].split("|")
        if len(fields) < 7:
            raise ValueError("Unexpected packed team record")
        fields[2] = "sitrusberry" if fields[2] != "sitrusberry" else "focussash"
        fields[3] = "intimidate" if fields[3] != "intimidate" else "pressure"
        fields[4] = "protect,helpinghand,facade,rockslide"
        fields[5] = "Timid" if fields[5] != "Timid" else "Adamant"
        fields[6] = "0,32,0,0,0,32"
        members[index] = "|".join(fields)
    return "]".join(members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-teams", type=Path, required=True)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=902400)
    parser.add_argument("--turn-limit", type=int, default=12)
    args = parser.parse_args()

    packed = json.loads(args.packed_teams.read_text())
    dex = ninjask.Dex.load(str(args.dex))
    team_count = len(packed)
    game_count = team_count * 2
    teams1: list[str] = []
    teams2: list[str] = []
    seeds: list[int] = []
    for index, foe in enumerate(packed):
        own = packed[(index + 1) % team_count]
        seed = args.seed + index
        teams1.extend((own, own))
        teams2.extend((foe, mutate_unselected_bench(foe)))
        seeds.extend((seed, seed))

    batch = ninjask.Batch(dex, game_count)
    batch.turn_limit = args.turn_limit
    batch.reset_many(
        list(range(game_count)), teams1, teams2, seeds
    )
    paired_slots = [(2 * i, 2 * i + 1) for i in range(team_count)]
    rngs = [[random.Random(19142 + i * 2 + side) for side in (0, 1)] for i in range(team_count)]
    matched_p1_states = 0
    matched_action_rows = 0
    decisions = 0
    paired_refusals = 0
    issues: list[dict] = []
    cycles = 0

    while True:
        observation = batch.observe()
        count = observation["n"]
        if count == 0:
            break
        who = np.frombuffer(observation["who"], dtype=np.uint32).reshape(count, 2)
        preview = np.frombuffer(observation["preview"], dtype=np.uint8)
        obs = np.frombuffer(observation["obs"], dtype=np.float32).reshape(count, ninjask.OBS_LEN)
        ids = np.frombuffer(observation["ids"], dtype=np.int16).reshape(count, ninjask.IDS_LEN)
        offsets = np.frombuffer(observation["offsets"], dtype=np.uint32)
        actions = np.frombuffer(observation["actions"], dtype=np.int8).reshape(-1, ninjask.ACTION_WIDTH)
        rows = {(int(game), int(side)): row for row, (game, side) in enumerate(who)}
        picks = [0] * count

        for pair_index, (baseline_slot, variant_slot) in enumerate(paired_slots):
            for side in (0, 1):
                baseline_row = rows.get((baseline_slot, side))
                variant_row = rows.get((variant_slot, side))
                if (baseline_row is None) != (variant_row is None):
                    issues.append({"pair": pair_index, "side": side, "cycle": cycles, "reason": "decision presence differs"})
                    continue
                if baseline_row is None:
                    continue

                start_a, end_a = int(offsets[baseline_row]), int(offsets[baseline_row + 1])
                start_b, end_b = int(offsets[variant_row]), int(offsets[variant_row + 1])
                rows_a = actions[start_a:end_a]
                rows_b = actions[start_b:end_b]
                if len(rows_a) != len(rows_b):
                    issues.append({"pair": pair_index, "side": side, "cycle": cycles, "reason": "legal choice count differs", "counts": [len(rows_a), len(rows_b)]})
                    continue
                if not np.array_equal(rows_a, rows_b):
                    issues.append({"pair": pair_index, "side": side, "cycle": cycles, "reason": "legal choice rows differ"})
                    continue
                matched_action_rows += 1

                # Only p1 is the observer of the mutated opposing sets. P2
                # legitimately knows its own full sets and may encode them.
                if side == 0:
                    equal = (
                        np.array_equal(obs[baseline_row], obs[variant_row])
                        and np.array_equal(ids[baseline_row], ids[variant_row])
                        and preview[baseline_row] == preview[variant_row]
                    )
                    if not equal:
                        issues.append({"pair": pair_index, "cycle": cycles, "reason": "p1 observation or IDs differ"})
                        continue
                    matched_p1_states += 1

                # Team-preview row zero is team 1234 (bring the unchanged
                # first four); battle decisions use the same seeded random
                # choice index in both paired worlds.
                pick = 0 if preview[baseline_row] else rngs[pair_index][side].randrange(len(rows_a))
                picks[baseline_row] = pick
                picks[variant_row] = pick

        if issues:
            break
        status = np.frombuffer(batch.step(picks), dtype=np.uint8)
        decisions += count
        for pair_index, (baseline_slot, variant_slot) in enumerate(paired_slots):
            for side in (0, 1):
                baseline_row = rows.get((baseline_slot, side))
                variant_row = rows.get((variant_slot, side))
                if baseline_row is None or variant_row is None:
                    continue
                refused_a = bool(status[baseline_row] & 1)
                refused_b = bool(status[variant_row] & 1)
                if refused_a != refused_b:
                    issues.append({"pair": pair_index, "side": side, "cycle": cycles, "reason": "choice-refusal flag differs"})
                paired_refusals += int(refused_a)
        cycles += 1

    report = {
        "status": "pass" if not issues else "fail",
        "opponent_teams": team_count,
        "paired_battles": game_count,
        "turn_limit": args.turn_limit,
        "matched_p1_observation_states": matched_p1_states,
        "matched_action_lists": matched_action_rows,
        "decision_rows": decisions,
        "decision_cycles": cycles,
        "paired_choice_refusals": paired_refusals,
        "team_preview": "index 0 brings positions 1-4 so altered positions 5-6 stay unselected",
        "mutated_fields": ["moves", "held item", "ability", "nature", "EVs"],
        "p1_observation_or_action_divergences": issues,
        "scope_limit": "Batch numeric-view smoke test on sampled trajectories; it does not certify every active hidden attribute, battle mechanic, or hidden-world assumption.",
    }
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
