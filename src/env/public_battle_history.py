"""Track mechanics facts that are reconstructible from public Showdown logs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from poke_env.data.normalize import to_id_str


STALLING_MOVES = frozenset({
    "protect", "detect", "endure", "spikyshield", "banefulbunker",
    "kingsshield", "obstruct", "silktrap", "burningbulwark", "maxguard",
    "matblock", "quickguard", "wideguard",
})

_SUCCESS_MESSAGES = frozenset({
    "protect", "detect", "endure", "spiky shield", "baneful bunker",
    "king's shield", "obstruct", "silk trap", "burning bulwark", "max guard",
    "mat block", "quick guard", "wide guard",
})


def _actor(value: str) -> tuple[str, str, str] | None:
    """Parse a public protocol ident into (side, active slot, nickname)."""
    if ":" not in value:
        return None
    slot, nickname = value.split(":", 1)
    slot = slot.strip().lower()
    if len(slot) < 2 or slot[:2] not in {"p1", "p2"}:
        return None
    return slot[:2], slot, to_id_str(nickname.strip())


class PublicBattleHistory:
    """Incremental, hidden-set-free summary of public battle events.

    The tracker records moves only when a ``|move|`` line reveals them. Protect
    streaks advance only on Showdown's public successful-protection event and
    reset on a non-stalling move, a failed stalling move, a switch, or a full
    turn without another successful stall move.
    """

    def __init__(self) -> None:
        self.revealed_moves: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.recent_moves: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.last_move: dict[tuple[str, str], str] = {}
        self.protect_streak: dict[tuple[str, str], int] = defaultdict(int)
        self.last_successful_stall_turn: dict[tuple[str, str], int] = {}
        self.switch_turn: dict[tuple[str, str], int] = {}
        self.move_actions_since_switch: dict[tuple[str, str], int] = {}
        self.species_by_pokemon: dict[tuple[str, str], str] = {}
        self.turn_number = 0
        self.active_by_slot: dict[str, tuple[str, str]] = {}
        self.pending_stall: dict[tuple[str, str], str] = {}

    def consume(self, lines: Iterable[str]) -> None:
        for line in lines:
            parts = [part.strip() for part in str(line).split("|")]
            if len(parts) < 3:
                continue
            event = parts[1]
            if event == "turn" and len(parts) >= 3:
                try:
                    self.turn_number = int(parts[2])
                except ValueError:
                    continue
                for key, successful_turn in tuple(self.last_successful_stall_turn.items()):
                    if self.turn_number - successful_turn > 1:
                        self.protect_streak[key] = 0
                        self.last_successful_stall_turn.pop(key, None)
                for key in tuple(self.pending_stall):
                    self.protect_streak[key] = 0
                    self.last_successful_stall_turn.pop(key, None)
                    self.pending_stall.pop(key, None)
                # Initial switches are announced before turn 1. Treat those
                # active Pokémon as entering on turn 1 for Fake Out timing.
                if self.turn_number == 1:
                    for key, entered_on in tuple(self.switch_turn.items()):
                        if entered_on == 0:
                            self.switch_turn[key] = 1
                continue
            if event in {"switch", "drag", "replace"} and len(parts) >= 3:
                actor = _actor(parts[2])
                if actor is not None:
                    side, slot, nickname = actor
                    previous = self.active_by_slot.get(slot)
                    if previous is not None:
                        self.protect_streak[previous] = 0
                        self.last_successful_stall_turn.pop(previous, None)
                        self.pending_stall.pop(previous, None)
                    key = (side, nickname)
                    self.active_by_slot[slot] = key
                    self.protect_streak[key] = 0
                    self.last_successful_stall_turn.pop(key, None)
                    self.switch_turn[key] = self.turn_number
                    self.move_actions_since_switch[key] = 0
                    if len(parts) >= 4:
                        species = parts[3].split(",", 1)[0].strip()
                        if species:
                            self.species_by_pokemon[key] = species
                    self.pending_stall.pop(key, None)
                continue

            if event == "move" and len(parts) >= 4:
                actor = _actor(parts[2])
                if actor is None:
                    continue
                side, slot, nickname = actor
                key = (side, nickname)
                move = to_id_str(parts[3])
                self.revealed_moves[key].add(move)
                recent = self.recent_moves[key]
                recent.append(move)
                del recent[:-2]
                self.last_move[key] = move
                # Fake Out is restricted by Showdown's activeMoveActions, not
                # merely by elapsed turns. runMove increments the counter even
                # for external moves such as Dancer and Instruct, so their
                # public move events count just like ordinary move actions.
                if key in self.move_actions_since_switch:
                    self.move_actions_since_switch[key] += 1
                if move in STALLING_MOVES:
                    self.pending_stall[key] = move
                else:
                    self.protect_streak[key] = 0
                    self.last_successful_stall_turn.pop(key, None)
                    self.pending_stall.pop(key, None)
                self.active_by_slot[slot] = key
                continue

            if event == "-singleturn" and len(parts) >= 4:
                actor = _actor(parts[2])
                message = parts[3].lower().removeprefix("move: ").strip()
                if actor is not None and message in _SUCCESS_MESSAGES:
                    key = (actor[0], actor[2])
                    if key in self.pending_stall:
                        self.protect_streak[key] = min(
                            self.protect_streak[key] + 1, 6
                        )
                        self.last_successful_stall_turn[key] = self.turn_number
                        self.pending_stall.pop(key, None)
                continue

            if event == "cant" and len(parts) >= 3:
                actor = _actor(parts[2])
                if actor is not None:
                    key = (actor[0], actor[2])
                    if key in self.move_actions_since_switch:
                        self.move_actions_since_switch[key] += 1
                continue

            if event in {"-fail", "-block"} and len(parts) >= 3:
                actor = _actor(parts[2])
                if actor is not None:
                    key = (actor[0], actor[2])
                    if key in self.pending_stall:
                        self.protect_streak[key] = 0
                        self.last_successful_stall_turn.pop(key, None)
                        self.pending_stall.pop(key, None)

    def for_battle(self, battle: Any) -> dict[str, list[dict[str, Any]]]:
        """Project the tracked public history into the battle's roster order."""
        player_role = str(getattr(battle, "player_role", "p1")).lower()
        own_side = "p2" if player_role.startswith("p2") else "p1"
        opponent_side = "p1" if own_side == "p2" else "p2"

        def rows(team: Any, side: str) -> list[dict[str, Any]]:
            members = team.values() if isinstance(team, dict) else team or ()
            result = []
            for pokemon in members:
                nickname = to_id_str(str(getattr(pokemon, "name", "")))
                key = (side, nickname)
                result.append({
                    "revealed_moves": sorted(self.revealed_moves.get(key, ())),
                    "recent_moves": list(reversed(self.recent_moves.get(key, ()))),
                    "last_move": self.last_move.get(key),
                    "protect_streak": int(self.protect_streak.get(key, 0)),
                    "turns_since_switch": (
                        max(self.turn_number - self.switch_turn[key], 0)
                        if key in self.switch_turn else None
                    ),
                    "move_actions_since_switch": self.move_actions_since_switch.get(key),
                })
            return result[:6]

        own = rows(getattr(battle, "team", {}), own_side)
        preview = getattr(battle, "teampreview_opponent_team", None)
        opponent_team = preview or getattr(battle, "opponent_team", {})
        opponent = rows(opponent_team, opponent_side)
        empty = {
            "revealed_moves": [], "recent_moves": [], "last_move": None, "protect_streak": 0,
            "turns_since_switch": None,
            "move_actions_since_switch": None,
        }
        own = (own + [dict(empty) for _ in range(6)])[:6]
        opponent = (opponent + [dict(empty) for _ in range(6)])[:6]
        return {"own": own, "opponent": opponent}

    def for_species_rosters(
        self,
        own_species: Iterable[str],
        opponent_species: Iterable[str],
        *,
        own_side: str = "p1",
    ) -> dict[str, list[dict[str, Any]]]:
        """Project public history onto rosters without hidden nicknames.

        Search workers know the acting player's full roster and only public
        preview species for the opponent. Switch-event details associate a
        public species with its protocol ident, so this projection never needs
        the hidden nickname or set for an unrevealed opponent member.
        """
        own_side = "p2" if str(own_side).lower().startswith("p2") else "p1"
        opponent_side = "p1" if own_side == "p2" else "p2"

        def rows(species_list: Iterable[str], side: str) -> list[dict[str, Any]]:
            by_species = {
                to_id_str(species): key
                for key, species in self.species_by_pokemon.items()
                if key[0] == side
            }
            result = []
            for species in list(species_list)[:6]:
                key = by_species.get(to_id_str(str(species)))
                age = (
                    max(self.turn_number - self.switch_turn[key], 0)
                    if key is not None and key in self.switch_turn else None
                )
                result.append({
                    "revealed_moves": sorted(self.revealed_moves.get(key, ())) if key else [],
                    "recent_moves": list(reversed(self.recent_moves.get(key, ()))) if key else [],
                    "last_move": self.last_move.get(key) if key else None,
                    "protect_streak": int(self.protect_streak.get(key, 0)) if key else 0,
                    "turns_since_switch": age,
                    "move_actions_since_switch": (
                        self.move_actions_since_switch.get(key) if key else None
                    ),
                })
            empty = {
                "revealed_moves": [], "recent_moves": [], "last_move": None, "protect_streak": 0,
                "turns_since_switch": None,
                "move_actions_since_switch": None,
            }
            return (result + [dict(empty) for _ in range(6)])[:6]

        return {
            "own": rows(own_species, own_side),
            "opponent": rows(opponent_species, opponent_side),
        }
