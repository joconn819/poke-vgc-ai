#!/usr/bin/env python3
"""Challenge a local Showdown bot once and wait for the complete VGC match."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from scripts.run_lan_bot import load_team, pack_showdown_team, team_to_showdown_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", required=True, help="local Showdown bot username")
    parser.add_argument("--username", default="VGCTestOpp2")
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--showdown-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=202610291)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if len(args.username) > 18:
        parser.error("Showdown usernames must be 18 characters or shorter")
    if args.port < 1 or args.timeout <= 0:
        parser.error("port and timeout must be positive")

    team = pack_showdown_team(team_to_showdown_text(
        load_team(args.seed, args.format, args.showdown_path.resolve())
    ))
    server = ServerConfiguration(
        websocket_url=f"ws://{args.host}:{args.port}/showdown/websocket",
        authentication_url=f"http://{args.host}:{args.port}/action.php?",
    )
    challenger = RandomPlayer(
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.format,
        server_configuration=server,
        team=team,
        accept_open_team_sheet=False,
        start_timer_on_battle_start=True,
    )

    async def challenge() -> None:
        try:
            await challenger.send_challenges(args.bot, 1)
            deadline = asyncio.get_running_loop().time() + args.timeout
            while asyncio.get_running_loop().time() < deadline:
                battles = list(challenger.battles.values())
                if battles and battles[-1].finished:
                    battle = battles[-1]
                    print(json.dumps({
                        "finished": True,
                        "challenger_won": bool(battle.won),
                        "turn": int(battle.turn),
                        "battle_tag": battle.battle_tag,
                        "format": args.format,
                    }, sort_keys=True))
                    return
                await asyncio.sleep(1)
            raise TimeoutError(
                f"local battle did not finish in {args.timeout}s; "
                f"active battle count: {len(challenger.battles)}"
            )
        finally:
            try:
                await challenger.ps_client._stop_listening()
            except Exception:
                pass

    asyncio.run(challenge())


if __name__ == "__main__":
    main()
