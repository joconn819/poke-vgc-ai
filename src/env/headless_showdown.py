"""Headless local Showdown match bridge with poke-env-compatible observations.

The worker owns the authoritative Showdown ``Battle``. Python receives only the
normal battle log plus each side's own request, then reconstructs the same
player-view objects used by the local-server adapter.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import MoveSet
from poke_env.battle.pokemon import Pokemon
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.data.normalize import to_id_str
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from src.env.showdown_env import ShowdownDoublesEnv
from src.env.public_battle_history import PublicBattleHistory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPOSITORY_ROOT / "scripts" / "showdown_headless_worker.js"


class HeadlessShowdownWorker:
    """Persistent Node process that can host multiple sequential matches."""

    def __init__(
        self,
        showdown_path: str | Path,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        root = Path(showdown_path).resolve()
        self.process = subprocess.Popen(
            ["node", str(WORKER_PATH)],
            cwd=REPOSITORY_ROOT,
            env={**os.environ, "POKEMON_SHOWDOWN_PATH": str(root)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def start_match(
        self,
        team1: Sequence[dict[str, Any]],
        team2: Sequence[dict[str, Any]],
        *,
        format_id: str,
        seed: int,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.send({
            "type": "start_match",
            "format": format_id,
            "seed": int(seed),
            "team1": list(team1),
            "team2": list(team2),
            "snapshot": snapshot,
        })

    def generate_team(
        self,
        seed: int,
        format_id: str,
        archetype: str = "random",
    ) -> list[dict[str, Any]]:
        self.send({
            "type": "generate_team",
            "seed": int(seed),
            "format": format_id,
            "archetype": archetype,
        })
        payload = self.receive()
        if payload.get("type") != "team" or not isinstance(payload.get("team"), list):
            raise RuntimeError(f"invalid Showdown team-generation response: {payload}")
        return payload["team"]

    def send(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("headless worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def receive(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("headless worker stdout is unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout_seconds)
        if not ready:
            status = self.process.poll()
            diagnostic = ""
            if self.process.stderr is not None:
                descriptor = self.process.stderr.fileno()
                os.set_blocking(descriptor, False)
                try:
                    diagnostic = os.read(descriptor, 65536).decode(errors="replace").strip()
                except BlockingIOError:
                    pass
            raise TimeoutError(
                f"headless Showdown worker exceeded {self.timeout_seconds:.1f}s "
                f"(exit={status}); stderr={diagnostic}"
            )
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"headless Showdown worker exited unexpectedly: {stderr.strip()}")
        payload = json.loads(line)
        if payload.get("type") not in {"state", "team", "search_worlds", "random_seed_set", "snapshot", "match_aborted"}:
            raise RuntimeError(f"unexpected headless worker payload: {payload}")
        return payload

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)

    def __enter__(self) -> "HeadlessShowdownWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HeadlessShowdownProcess:
    """One interactive match hosted by the installed local Showdown engine."""

    def __init__(
        self,
        team1: Sequence[dict[str, Any]],
        team2: Sequence[dict[str, Any]],
        *,
        format_id: str,
        seed: int,
        showdown_path: str | Path,
        timeout_seconds: float = 10.0,
        worker: HeadlessShowdownWorker | None = None,
        snapshot: dict[str, Any] | None = None,
        parser_snapshot: dict[str, Any] | None = None,
        refresh_request_move_slots_for_sides: Sequence[int] = (),
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._worker = worker or HeadlessShowdownWorker(
            showdown_path, timeout_seconds=timeout_seconds
        )
        self._owns_worker = worker is None
        self._process = self._worker.process
        self._battles = deepcopy(parser_snapshot["battles"]) if parser_snapshot else [
            DoubleBattle(
                battle_tag=f"headless-{seed}-{side}",
                username=f"Headless P{side}",
                logger=logging.getLogger(f"headless-showdown-p{side}"),
                save_replays=False,
                gen=9,
            )
            for side in (1, 2)
        ]
        self._public_history = (
            deepcopy(parser_snapshot["public_history"])
            if parser_snapshot else PublicBattleHistory()
        )
        self._refresh_request_move_slots_for_sides = frozenset(
            int(side) for side in refresh_request_move_slots_for_sides
        )
        if any(side not in {0, 1} for side in self._refresh_request_move_slots_for_sides):
            raise ValueError("request move refresh sides must be 0 or 1")
        self._last_state: dict[str, Any] | None = None
        self._aborted = False
        self.action_adjustments = 0
        self._worker.start_match(team1, team2, format_id=format_id, seed=seed, snapshot=snapshot)
        self._last_state = self._receive()
        if self._last_state.get("type") != "state":
            raise RuntimeError(f"expected initial battle state, got {self._last_state}")
        self._apply_state(self._last_state)

    @property
    def battles(self) -> tuple[DoubleBattle, DoubleBattle]:
        return self._battles[0], self._battles[1]

    @property
    def public_history(self) -> PublicBattleHistory:
        """Return the shared log-derived tracker, containing public facts only."""
        return self._public_history

    @property
    def state(self) -> dict[str, Any]:
        if self._last_state is None:
            raise RuntimeError("headless match has not produced a state")
        return self._last_state

    @property
    def ended(self) -> bool:
        return self._aborted or bool(self.state.get("ended"))

    def observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return both player-view observations and legal native action masks."""
        if self.ended:
            raise RuntimeError("terminal headless match has no action observations")
        observations = tuple(
            ShowdownDoublesEnv.embed_battle(None, battle) for battle in self._battles
        )
        masks = tuple(
            np.asarray(ShowdownDoublesEnv.get_action_mask(battle), dtype=np.int8)
            for battle in self._battles
        )
        return observations[0], masks[0], observations[1], masks[1]

    def public_mechanics(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return each player's public event summary in its own roster order."""
        return tuple(
            self._public_history.for_battle(battle) for battle in self._battles
        )  # type: ignore[return-value]

    def prepare_search_worlds(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ask the worker for closed-sheet current-state simulator snapshots."""
        self._worker.send({"type": "search_worlds", **request})
        response = self._worker.receive()
        if response.get("type") != "search_worlds":
            raise RuntimeError(f"expected sanitized search worlds, got {response}")
        if response.get("error"):
            raise RuntimeError(
                f"current-state search-world preparation failed: {response['error']}\n"
                f"{response.get('stack', '')}"
            )
        return response

    def abort_match(self) -> None:
        """Discard a live disposable battle while keeping a shared worker alive."""
        if self.ended:
            return
        self._send({"type": "abort_match"})
        response = self._worker.receive()
        if response.get("type") != "match_aborted":
            raise RuntimeError(f"expected aborted-match acknowledgement, got {response}")
        self._aborted = True

    def choices_from_native_actions(
        self,
        actions: Sequence[Sequence[int]],
        *,
        forced_targetless_positions: dict[int, Sequence[int]] | None = None,
    ) -> tuple[str, str]:
        """Translate native masked actions back to strict Showdown choices."""
        if len(actions) != 2:
            raise ValueError("two native action pairs are required")
        result = []
        requests = self.state["requests"]
        for side_index, (action, battle) in enumerate(zip(actions, self._battles)):
            if requests[side_index] is None or requests[side_index].get("wait"):
                result.append("")
                continue
            targetless_slots = set(
                (forced_targetless_positions or {}).get(side_index, ())
            )
            order = ShowdownDoublesEnv.action_to_order(
                np.asarray(action, dtype=np.int64),
                battle,
                fake=bool(targetless_slots),
                strict=True,
            )
            if not isinstance(order, DoubleBattleOrder):
                raise ValueError(f"expected a doubles order, received {order!r}")
            single_orders = [order.first_order, order.second_order]
            active_requests = requests[side_index].get("active") or []
            for position, single_order in enumerate(single_orders):
                move_slots = (
                    active_requests[position].get("moves", [])
                    if position < len(active_requests) else []
                )
                if (
                    len(move_slots) == 1
                    and move_slots[0].get("pp") is None
                    and getattr(getattr(single_order, "order", None), "id", None)
                    == move_slots[0].get("id")
                    and single_order.move_target == 0
                ):
                    targetless_slots.add(position)
            for position, single_order in enumerate(single_orders):
                if position in targetless_slots:
                    move_slots = (
                        active_requests[position].get("moves", [])
                        if position < len(active_requests) else []
                    )
                    if (
                        not hasattr(single_order.order, "id")
                        or single_order.move_target != 0
                        or len(move_slots) != 1
                        or move_slots[0].get("id") != single_order.order.id
                        or move_slots[0].get("pp") is not None
                        or (single_order.mega and not active_requests[position].get("canMegaEvo"))
                        or (
                            single_order.terastallize
                            and not active_requests[position].get("canTerastallize")
                        )
                    ):
                        raise ValueError(
                            "targetless search order does not match a forced move request"
                        )
                elif str(single_order) not in {
                    str(candidate) for candidate in battle.valid_orders[position]
                }:
                    raise ValueError(
                        f"native action is not in the live legal-order list: {single_order}"
                    )
            parts = [
                self._single_order_to_choice(
                    order.first_order, requests[side_index], 0, battle,
                    forced_targetless=0 in targetless_slots,
                ),
                self._single_order_to_choice(
                    order.second_order, requests[side_index], 1, battle,
                    forced_targetless=1 in targetless_slots,
                ),
            ]
            result.append(", ".join(parts))
        return result[0], result[1]

    def _single_order_to_choice(
        self,
        order: SingleBattleOrder,
        request: dict[str, Any],
        position: int,
        battle: DoubleBattle,
        *,
        forced_targetless: bool = False,
    ) -> str:
        if isinstance(order, (PassBattleOrder, DefaultBattleOrder)):
            return "pass"
        if isinstance(order, ForfeitBattleOrder):
            return "forfeit"
        if isinstance(order.order, Pokemon):
            target_species = order.order.species
            pokemon = request["side"]["pokemon"]
            for index, member in enumerate(pokemon):
                if to_id_str(member.get("details", "").split(",", 1)[0]) == to_id_str(target_species):
                    return f"switch {index + 1}"
            # Species strings can differ only in cosmetic forme spelling; use
            # the battle's stable team order as a second exact source.
            own_team = list(battle.team.values())
            for index, member in enumerate(own_team):
                if member is order.order:
                    return f"switch {index + 1}"
            raise ValueError(f"switch target {target_species!r} is absent from own request")

        if not hasattr(order.order, "id"):
            raise ValueError(f"unsupported native order {order!r}")
        active_requests = request.get("active") or []
        if position >= len(active_requests):
            raise ValueError("move order has no corresponding active request")
        move_id = order.order.id
        move_slots = active_requests[position].get("moves", [])
        slot_index = next(
            (index for index, move in enumerate(move_slots) if move.get("id") == move_id),
            None,
        )
        if slot_index is None:
            raise ValueError(f"move {move_id!r} is absent from the player's request")
        command = f"move {slot_index + 1}"
        if forced_targetless:
            if (
                order.move_target != 0
                or len(move_slots) != 1
                or move_slots[0].get("pp") is not None
                or (order.mega and not active_requests[position].get("canMegaEvo"))
                or (order.terastallize and not active_requests[position].get("canTerastallize"))
            ):
                raise ValueError("targetless search move is not a forced move request")
            if order.terastallize:
                command += " terastallize"
            elif order.mega:
                command += " mega"
            return command
        move_target_type = str(move_slots[slot_index].get("target", ""))
        if os.environ.get("POKE_HEADLESS_DEBUG"):
            print(
                "native_move_choice",
                {"position": position, "move": move_id, "request_target": move_target_type,
                 "order_target": order.move_target,
                 "legal_orders": [str(candidate) for candidate in battle.valid_orders[position]
                                  if getattr(getattr(candidate, "order", None), "id", None) == move_id]},
                file=sys.stderr,
                flush=True,
            )
        # poke-env's factorized doubles action space carries target variants
        # for every move. Showdown accepts a target location only for moves
        # whose target type permits player selection; passing poke-env's
        # default sentinel to self/field/spread moves is rejected by the local
        # engine (for example, "You can't choose a target for Solar Beam").
        choosable_targets = {
            "normal", "any", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe"
        }
        if move_target_type in choosable_targets:
            target_loc = int(order.move_target or 0)
            self_loc = -(position + 1)
            invalid_selected_target = target_loc == 0
            if move_target_type == "adjacentAllyOrSelf":
                invalid_selected_target |= target_loc > 0
            elif move_target_type == "adjacentAlly":
                invalid_selected_target |= target_loc >= 0 or target_loc == self_loc
            elif move_target_type == "adjacentFoe":
                invalid_selected_target |= target_loc <= 0
            elif move_target_type == "any":
                invalid_selected_target |= target_loc == self_loc
            elif move_target_type == "normal":
                invalid_selected_target |= target_loc == self_loc
            if invalid_selected_target:
                # The server request can expose a targetable move as a single
                # poke-env order with targetLoc=0, which Showdown rejects in
                # doubles. Preserve a valid deterministic default: self for
                # ally-or-self moves, the partner for ally moves, and the
                # opposite foe for other selectable target types.
                if move_target_type == "adjacentAllyOrSelf":
                    target_loc = self_loc
                elif move_target_type == "adjacentAlly":
                    target_loc = -(2 - position)
                else:
                    target_loc = 2 - position
                self.action_adjustments += 1
            command += f" {target_loc:+d}"
        if order.terastallize:
            command += " terastallize"
        elif order.mega:
            command += " mega"
        elif order.z_move:
            command += " zmove"
        elif order.dynamax:
            command += " dynamax"
        return command

    def advance(self, choice1: str, choice2: str) -> dict[str, Any]:
        """Submit one simultaneous choice pair and return the next public state."""
        if self.ended:
            raise RuntimeError("cannot advance a completed headless match")
        self._send({"type": "choices", "choices": [choice1, choice2]})
        self._last_state = self._receive()
        self._apply_state(self._last_state)
        return self._last_state

    def reseed_random(self, seed: int) -> None:
        """Set Showdown's future battle PRNG stream without changing current state."""
        if self.ended:
            raise RuntimeError("cannot reseed a completed headless match")
        self._send({"type": "reseed_random", "seed": int(seed)})
        response = self._worker.receive()
        if response.get("type") != "random_seed_set" or response.get("seed") != int(seed):
            raise RuntimeError(f"Showdown did not confirm the requested battle RNG seed: {response}")

    def snapshot(self) -> dict[str, Any]:
        """Serialize the live authoritative Showdown battle for exact branching."""
        if self.ended:
            raise RuntimeError("cannot snapshot a completed headless match")
        self._send({"type": "snapshot"})
        response = self._worker.receive()
        if response.get("type") != "snapshot" or not isinstance(response.get("snapshot"), dict):
            raise RuntimeError(f"Showdown did not return a battle snapshot: {response}")
        return response["snapshot"]

    def parser_snapshot(self) -> dict[str, Any]:
        """Copy player-view parsers and public history for a resumed branch."""
        return {
            "battles": deepcopy(self._battles),
            "public_history": deepcopy(self._public_history),
        }

    def _send(self, payload: dict[str, Any]) -> None:
        self._worker.send(payload)

    def _receive(self) -> dict[str, Any]:
        payload = self._worker.receive()
        if payload.get("type") != "state":
            raise RuntimeError(f"expected battle state from Showdown worker, got {payload}")
        return payload

    def _apply_state(self, state: dict[str, Any]) -> None:
        # Public history is built from the shared protocol branch only. Each
        # parser receives its own recipient-specific projection, never the raw
        # omniscient Battle.log that contains both sides' split messages.
        self._public_history.consume(state.get("log", []))
        player_logs = state.get("player_logs")
        if not isinstance(player_logs, list) or len(player_logs) != 2:
            raise ValueError("headless state must contain two player-specific logs")
        for battle, player_log in zip(self._battles, player_logs):
            for line in player_log:
                parts = str(line).split("|")
                # The direct simulator includes public room timestamps in its
                # log buffer; the websocket client strips them before parsing.
                if len(parts) > 1 and parts[1] in {"t:", "win", "tie"}:
                    continue
                battle.parse_message(parts)
        requests = state.get("requests", [None, None])
        refresh_sides = self._refresh_request_move_slots_for_sides
        for side_index, (battle, request) in enumerate(zip(self._battles, requests)):
            if request is not None:
                if side_index in refresh_sides:
                    # A hypothetical hidden-world branch can change a player's
                    # own sampled moves while retaining the original parser
                    # objects. Refresh every active request, because a sampled
                    # bench Pokemon may switch in after the branch starts.
                    for slot, active_request in enumerate(request.get("active") or ()):
                        if (not active_request or not active_request.get("moves")
                                or slot >= len(battle.active_pokemon)):
                            continue
                        pokemon = battle.active_pokemon[slot]
                        if pokemon is not None:
                            # ``Pokemon.moves`` can be a synthesized copy when
                            # Mimic/Transform overrides are present. Replace
                            # the container so stale source-set entries cannot
                            # survive the restored world's current request.
                            pokemon._moves = MoveSet({})
                battle.parse_request(request, strict_battle_tracking=False)

    def close(self) -> None:
        if self._owns_worker:
            self._worker.close()

    def __enter__(self) -> "HeadlessShowdownProcess":
        return self

    def __exit__(self, *_: object) -> None:
        if not self.ended:
            self.abort_match()
        self.close()
