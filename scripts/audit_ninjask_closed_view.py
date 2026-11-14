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


def mutate_hidden_fields(packed: str) -> str:
    """Change hidden set fields while preserving public species and gender."""
    changed = []
    for pokemon in packed.split("]"):
        if not pokemon:
            continue
        fields = pokemon.split("|")
        if len(fields) < 7:
            raise ValueError("Unexpected packed team record")
        fields[2] = "sitrusberry" if fields[2] != "sitrusberry" else "focussash"
        fields[3] = "intimidate" if fields[3] != "intimidate" else "pressure"
        fields[4] = "protect,helpinghand,facade,rockslide"
        fields[5] = "Timid" if fields[5] != "Timid" else "Adamant"
        fields[6] = "0,32,0,0,0,32"
        changed.append("|".join(fields))
    return "]".join(changed)


def player_view(
    dex: ninjask.Dex,
    own: str,
    foe: str,
    seed: int,
    *,
    select_leads: bool,
) -> str:
    battle = ninjask.Battle.reference(
        dex,
        own,
        foe,
        seed_from_int(seed),
        names=("Observer", "Opponent"),
    )
    if select_leads:
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

    preselection_checked = 0
    postselection_move_checked = 0
    for opponent_index, foe in enumerate(packed):
        changed_foe = mutate_hidden_fields(foe)
        if changed_foe == foe:
            raise AssertionError(f"Team {opponent_index} was not mutated")
        original = player_view(
            dex, packed[0], foe, args.seed + opponent_index, select_leads=False
        )
        changed = player_view(
            dex, packed[0], changed_foe, args.seed + opponent_index, select_leads=False
        )
        if original != changed:
            raise AssertionError(
                f"Observer preview changed when hidden opponent sets changed, team {opponent_index}"
            )
        preselection_checked += 1

        # Items and abilities may activate and become public at battle start,
        # so only mutate moves in the post-selection comparison.
        move_variant = []
        for pokemon in foe.split("]"):
            if not pokemon:
                continue
            fields = pokemon.split("|")
            fields[4] = "protect,helpinghand,facade,rockslide"
            move_variant.append("|".join(fields))
        original = player_view(
            dex, packed[0], foe, args.seed + opponent_index, select_leads=True
        )
        changed = player_view(
            dex,
            packed[0],
            "]".join(move_variant),
            args.seed + opponent_index,
            select_leads=True,
        )
        if original != changed:
            raise AssertionError(
                f"Observer battle view changed when opponent moves changed, team {opponent_index}"
            )
        postselection_move_checked += 1

    sample_battle = ninjask.Battle.reference(
        dex, packed[0], packed[1], seed_from_int(args.seed)
    )
    projection = json.loads(sample_battle.projection())
    opponent_member = projection["sides"][1]["pokemon"][0]
    projection_contains_hidden_sets = bool(
        opponent_member.get("moves")
        and opponent_member.get("item")
        and opponent_member.get("ability")
        and opponent_member.get("storedStats")
    )
    if not projection_contains_hidden_sets:
        raise AssertionError("Expected full battle projection to contain hidden opponent set data")

    print(
        json.dumps(
            {
                "status": "pass",
                "preselection_phase": "team preview, before lead selection",
                "preselection_opponent_teams_checked": preselection_checked,
                "preselection_hidden_fields_mutated": ["item", "ability", "moves", "nature", "EVs"],
                "postselection_phase": "after lead selection, before turn-one choices",
                "postselection_opponent_teams_checked": postselection_move_checked,
                "postselection_hidden_fields_mutated": ["opponent_moves"],
                "observer_team_index": 0,
                "public_species_and_selected_leads_preserved": True,
                "projection_contains_hidden_opponent_sets": projection_contains_hidden_sets,
                "scope_limit": "Tracker.view invariance smoke tests only; not every hidden field, event, or midgame observation is certified.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
