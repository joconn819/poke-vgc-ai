#!/usr/bin/env python3
"""Check that Ninjask's player-view Tracker hides unrevealed opponent moves.

This is a focused information-boundary smoke audit, not a complete privacy proof.
Run in an environment with ``ninjask==0.4.0`` and a Dex JSON for the target format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ninjask
from ninjask.refsim import seed_from_int


def mutate_moves(packed: str) -> str:
    """Change all hidden move sets while preserving species, items, and abilities."""
    changed = []
    for pokemon in packed.split("]"):
        if not pokemon:
            continue
        fields = pokemon.split("|")
        if len(fields) < 5:
            raise ValueError("Unexpected packed team record")
        fields[4] = "protect,helpinghand,facade,rockslide"
        changed.append("|".join(fields))
    return "]".join(changed)


def player_view(dex: ninjask.Dex, own: str, foe: str, seed: int) -> str:
    battle = ninjask.Battle.reference(
        dex,
        own,
        foe,
        seed_from_int(seed),
        names=("Observer", "Opponent"),
    )
    battle.choose("p1", "team 1234")
    battle.choose("p2", "team 1234")
    tracker = ninjask.Tracker(dex, "p1")
    tracker.feed(battle.log("p1", 0))
    tracker.feed_request(battle.request("p1"))
    return tracker.view()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=Path, required=True, help="JSON team pool")
    parser.add_argument("--packed-teams", type=Path, required=True, help="Packed team strings")
    parser.add_argument("--dex", type=Path, required=True, help="Ninjask format Dex JSON")
    parser.add_argument("--seed", type=int, default=721621)
    args = parser.parse_args()

    packed = json.loads(args.packed_teams.read_text())
    teams = json.loads(args.teams.read_text())["teams"]
    if len(packed) != len(teams):
        raise ValueError("Packed teams and source teams have different lengths")
    dex = ninjask.Dex.load(str(args.dex))

    checked = 0
    for opponent_index, foe in enumerate(packed):
        changed_foe = mutate_moves(foe)
        if changed_foe == foe:
            raise AssertionError(f"Team {opponent_index} was not mutated")
        original = player_view(dex, packed[0], foe, args.seed + opponent_index)
        changed = player_view(dex, packed[0], changed_foe, args.seed + opponent_index)
        if original != changed:
            raise AssertionError(
                f"Observer view changed when unrevealed moves changed, opponent team {opponent_index}"
            )
        checked += 1

    print(
        json.dumps(
            {
                "status": "pass",
                "phase": "after team preview and lead selection, before turn-one choices",
                "observer_team_index": 0,
                "opponent_teams_checked": checked,
                "hidden_fields_mutated": ["opponent_moves"],
                "public_species_and_selected_leads_preserved": True,
                "scope_limit": "Smoke test for move secrecy in Tracker.view; not a full audit of every field, event, or hidden attribute.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
