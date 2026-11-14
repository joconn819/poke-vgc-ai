#!/usr/bin/env python3
"""Differentially replay random legal-choice matches against local Showdown.

Requires ninjask==0.4.0, Node dependencies in the local Showdown checkout, and
the development-team pool plus its Showdown-packed equivalent as input files.
Actions are enumerated from each player's Showdown request, not from the full
simulator's omniscient legal-choice list. If a hidden trap or Imprison refuses
an action, the refreshed request is used and the player retries.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import ninjask
from ninjask.choices import joint_choices
from ninjask.refsim import seed_from_int


def choose_with_request_refresh(
    battle: ninjask.Battle,
    side: str,
    choice: str,
    rng: random.Random,
) -> tuple[str, int]:
    """Submit a player-view action, retrying only after request refresh."""
    retries = 0
    for _ in range(6):
        previous_request = battle.request(side)
        try:
            battle.choose(side, choice)
            return choice, retries
        except ValueError as exc:
            if not str(exc).startswith("[Unavailable choice]"):
                raise
            refreshed_request = battle.request(side)
            if refreshed_request is None or refreshed_request == previous_request:
                raise
            choices = joint_choices(json.loads(refreshed_request), game_type="doubles")
            if not choices:
                raise
            choice = rng.choice(choices)
            retries += 1
    raise RuntimeError("exceeded six refreshed-request retries")


def normalized_ninjask_log(battle: ninjask.Battle) -> list[str]:
    return [
        line
        for line in battle.log("p1", 0)
        if line not in ("|", "")
        and not line.startswith(("|gen|", "|rule|", "|t:|", "|-hint|"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--packed-teams", type=Path, required=True)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--showdown", type=Path, required=True)
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=720160)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    raw_teams = json.loads(args.teams.read_text())["teams"]
    packed_teams = json.loads(args.packed_teams.read_text())
    if len(raw_teams) != len(packed_teams):
        raise ValueError("Raw and packed team pools differ in length")
    dex = ninjask.Dex.load(str(args.dex))
    bridge = Path(__file__).with_name("ninjask_showdown_bridge.js")
    node = subprocess.Popen(
        ["node", str(bridge), str(args.showdown)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    matched = 0
    hidden_choice_retries = 0
    choice_rejections: list[dict] = []
    divergences: list[dict] = []

    try:
        for game in range(args.games):
            p1_index = game % len(raw_teams)
            p2_index = (game + 1) % len(raw_teams)
            seed = seed_from_int(args.seed + game)
            rng = random.Random(game + 772)
            battle = ninjask.Battle.reference(
                dex,
                packed_teams[p1_index],
                packed_teams[p2_index],
                seed,
                names=("Player 1", "Player 2"),
            )
            battle.choose("p1", "team 1234")
            battle.choose("p2", "team 1234")
            choices = [["team 1234", "team 1234"]]
            failure = None
            guard = 0
            while not battle.ended and guard < 150:
                guard += 1
                pair = []
                for side in ("p1", "p2"):
                    if not battle.needs_choice(side):
                        pair.append("")
                        continue
                    request = json.loads(battle.request(side))
                    options = joint_choices(request, game_type="doubles")
                    if not options:
                        failure = {"game": game, "turn": battle.turn, "side": side, "reason": "no request-derived choices"}
                        break
                    pair.append(rng.choice(options))
                if failure:
                    break
                try:
                    for side_index, side in enumerate(("p1", "p2")):
                        if pair[side_index]:
                            pair[side_index], retries = choose_with_request_refresh(
                                battle, side, pair[side_index], rng
                            )
                            hidden_choice_retries += retries
                except Exception as exc:
                    failure = {"game": game, "turn": battle.turn, "choices": pair, "reason": str(exc)}
                    break
                choices.append(pair)

            if failure:
                choice_rejections.append(failure)
                continue
            if not battle.ended:
                divergences.append({"game": game, "reason": "Ninjask match did not terminate within 150 choice cycles"})
                continue

            payload = {
                "format": args.format,
                "team1": raw_teams[p1_index],
                "team2": raw_teams[p2_index],
                "seed": seed,
                "choices": choices,
            }
            assert node.stdin is not None and node.stdout is not None
            node.stdin.write(json.dumps(payload) + "\n")
            node.stdin.flush()
            showdown = json.loads(node.stdout.readline())
            nlog = normalized_ninjask_log(battle)
            slog = showdown["log"]
            mismatch = next((idx for idx, (a, b) in enumerate(zip(nlog, slog)) if a != b), None)
            if mismatch is None and len(nlog) != len(slog):
                mismatch = min(len(nlog), len(slog))
            ninjask_winner = (
                "Player 1" if battle.winner_side == "p1" else
                "Player 2" if battle.winner_side == "p2" else None
            )
            if (
                not showdown["error"]
                and mismatch is None
                and len(nlog) == len(slog)
                and battle.turn == showdown["turn"]
                and battle.ended == showdown["ended"]
                and ninjask_winner == showdown["winner"]
            ):
                matched += 1
            else:
                index = 0 if mismatch is None else mismatch
                divergences.append({
                    "game": game,
                    "turns": [battle.turn, showdown["turn"]],
                    "winners": [ninjask_winner, showdown["winner"]],
                    "first_event_index": mismatch,
                    "ninjask_event": nlog[index:index + 2],
                    "showdown_event": slog[index:index + 2],
                    "showdown_error": showdown["error"],
                })
    finally:
        if node.stdin:
            node.stdin.close()
        node.wait(timeout=15)

    report = {
        "requested_games": args.games,
        "matched_games": matched,
        "request_refresh_retries": hidden_choice_retries,
        "choice_rejections": choice_rejections,
        "state_or_log_divergences": divergences,
        "choice_generation": "ninjask.choices.joint_choices from each acting player's own request; retry from refreshed request after hidden trap/Imprison refusal",
        "scope_limit": "Random legal-choice sample only; does not certify all mechanics or action combinations.",
    }
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n")
    print(rendered)
    if matched != args.games or choice_rejections or divergences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
