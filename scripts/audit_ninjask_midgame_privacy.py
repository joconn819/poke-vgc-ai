#!/usr/bin/env python3
"""Check that unrevealed bench sets do not affect a player's live view.

For each opponent team, the selected four are held fixed while only the fifth
and sixth sets are changed. Both worlds then play from the same seed and the
same player-request-constrained choices. This is a sampled invariance audit,
not a proof for all battle events or mechanics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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


def player_view(dex: ninjask.Dex, battle: ninjask.Battle) -> str:
    tracker = ninjask.Tracker(dex, "p1")
    tracker.feed(battle.log("p1", 0))
    request = battle.request("p1")
    if request:
        tracker.feed_request(request)
    return tracker.view()


def allowed_by_request(choice: str, request: dict) -> bool:
    active = request.get("active", [])
    for slot, action in enumerate(choice.split(", ")):
        fields = action.split()
        if not fields or slot >= len(active):
            continue
        view = active[slot]
        if fields[0] == "switch" and (view.get("trapped") or view.get("maybeTrapped")):
            return False
        if fields[0] == "move":
            moves = view.get("moves", [])
            move_index = int(fields[1]) - 1
            if 0 <= move_index < len(moves) and moves[move_index].get("disabled"):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--packed-teams", type=Path, required=True)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=902400)
    parser.add_argument("--choice-cycles", type=int, default=12)
    args = parser.parse_args()

    raw_teams = json.loads(args.teams.read_text())["teams"]
    packed = json.loads(args.packed_teams.read_text())
    if len(raw_teams) != len(packed):
        raise ValueError("Raw and packed team pools differ in length")
    dex = ninjask.Dex.load(str(args.dex))

    matched_states = 0
    fully_checked_games = 0
    choice_rejections: list[dict] = []
    view_divergences: list[dict] = []
    for game, foe in enumerate(packed):
        own = packed[(game + 1) % len(packed)]
        seed = seed_from_int(args.seed + game)
        baseline = ninjask.Battle.reference(
            dex, own, foe, seed, names=("Observer", "Opponent")
        )
        hidden_variant = ninjask.Battle.reference(
            dex,
            own,
            mutate_unselected_bench(foe),
            seed,
            names=("Observer", "Opponent"),
        )
        for battle in (baseline, hidden_variant):
            battle.choose("p1", "team 1234")
            battle.choose("p2", "team 1234")

        rng = random.Random(19142 + game)
        cycles = 0
        while not baseline.ended and cycles < args.choice_cycles:
            if baseline.log("p1", 0) != hidden_variant.log("p1", 0):
                view_divergences.append({"game": game, "turn": baseline.turn, "reason": "player log differs"})
                break
            if player_view(dex, baseline) != player_view(dex, hidden_variant):
                view_divergences.append({"game": game, "turn": baseline.turn, "reason": "Tracker.view differs"})
                break
            matched_states += 1

            choices: list[str] = []
            for side in ("p1", "p2"):
                if not baseline.needs_choice(side):
                    choices.append("")
                    continue
                request = json.loads(baseline.request(side))
                legal = [
                    action
                    for action in baseline.legal_choices(side)
                    if allowed_by_request(action, request)
                ]
                if not legal:
                    choice_rejections.append({"game": game, "turn": baseline.turn, "side": side, "reason": "no filtered choices"})
                    break
                choices.append(rng.choice(legal))
            if choice_rejections and choice_rejections[-1]["game"] == game:
                break
            try:
                for battle in (baseline, hidden_variant):
                    if choices[0]:
                        battle.choose("p1", choices[0])
                    if choices[1]:
                        battle.choose("p2", choices[1])
            except Exception as exc:
                choice_rejections.append({"game": game, "turn": baseline.turn, "choices": choices, "reason": str(exc)})
                break
            cycles += 1

        if (
            not view_divergences
            or view_divergences[-1].get("game") != game
        ) and (
            not choice_rejections
            or choice_rejections[-1].get("game") != game
        ):
            fully_checked_games += 1

    report = {
        "status": "pass" if not choice_rejections and not view_divergences else "fail",
        "opponent_teams": len(packed),
        "fully_checked_games": fully_checked_games,
        "matched_player_view_states": matched_states,
        "choice_cycles_per_game_max": args.choice_cycles,
        "mutated_hidden_members": "fifth and sixth team members; selected four and all public species preserved",
        "mutated_fields": ["moves", "held item", "ability", "nature", "EVs"],
        "choice_rejections": choice_rejections,
        "view_divergences": view_divergences,
        "scope_limit": "Sampled random-choice trajectories only; does not cover every hidden attribute, mechanic, or midgame event.",
    }
    print(json.dumps(report, indent=2))
    if choice_rejections or view_divergences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
