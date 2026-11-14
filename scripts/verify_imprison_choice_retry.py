#!/usr/bin/env python3
"""Exercise Showdown's rejected-move retry path with an Imprison matchup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.env.headless_showdown import HeadlessShowdownProcess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "training/inputs/round595_development_team_pool.json"
DEFAULT_SHOWDOWN = Path("/home/justin/p/poke_learning/pokemon-showdown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--showdown-path", type=Path, default=DEFAULT_SHOWDOWN)
    args = parser.parse_args()

    pool = json.loads(args.team_pool.read_text(encoding="utf-8"))["teams"]
    # Team 22 has Starmie with Expanding Force; team 19 has Indeedee with
    # Imprison. Put those Pokemon in active slot zero for a deterministic case.
    starmie_team, imprison_team = pool[22], pool[19]
    retries: list[dict[str, object]] = []

    with HeadlessShowdownProcess(
        starmie_team,
        imprison_team,
        format_id="gen9championsdoubles2026mcclosed",
        seed=998877,
        showdown_path=args.showdown_path,
    ) as match:
        match.advance("team 2,1,3,4", "team 3,1,4,5")
        match.advance("move 4, move 4", "move 4, move 4")

        def retry_handler(state, rejected_sides, previous_choices):
            request_moves = state["requests"][0]["active"][0]["moves"]
            retries.append({
                "sides": rejected_sides,
                "disabled_moves": [move["id"] for move in request_moves if move.get("disabled")],
                "accepted_side_choice_preserved": previous_choices[1] == "move 4, move 4",
            })
            return "move 4, move 4", previous_choices[1]

        result = match.advance(
            "move 2, move 4",
            "move 4, move 4",
            on_choice_retry=retry_handler,
        )
        assert retries, "Showdown did not reject the Imprison-disabled move"
        assert retries[0]["sides"] == [0]
        assert "expandingforce" in retries[0]["disabled_moves"]
        assert retries[0]["accepted_side_choice_preserved"]
        assert match.action_adjustments >= 1
        assert result["turn"] == 3

    print(json.dumps({
        "result": "passed",
        "retry": retries[0],
        "completed_turn": result["turn"],
        "closed_team_sheet": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
