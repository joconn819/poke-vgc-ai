"""Player-view opponent tracking and sampled hidden-set hypotheses."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import subprocess
import select
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from src.jsonl_io import open_jsonl_binary



@dataclass
class RevealedPokemon:
    """Publicly observed facts for one preview-revealed opposing species."""

    species: str
    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    active_ability: str | None = None
    hp_fraction: float | None = None
    status: str | None = None
    tera_type: str | None = None


_PUBLIC_STATUSES = frozenset({"brn", "frz", "par", "psn", "slp", "tox"})
_PUBLIC_CONDITION = re.compile(r"^(\d+)\s*/\s*(\d+)[a-z]?(?:\s+([a-z]+))?$", re.IGNORECASE)


def _apply_public_condition(pokemon: RevealedPokemon, condition: str) -> None:
    """Store only the shared HP fraction/status encoded in Showdown protocol."""
    tokens = str(condition).strip().lower().split()
    if "fnt" in tokens:
        pokemon.hp_fraction = 0.0
        return
    match = _PUBLIC_CONDITION.match(str(condition).strip())
    if not match:
        return
    current, maximum = int(match.group(1)), int(match.group(2))
    if maximum > 0:
        pokemon.hp_fraction = min(1.0, max(0.0, current / maximum))
    status = match.group(3)
    pokemon.status = status if status in _PUBLIC_STATUSES else None


class ShowdownHiddenTeamSampler:
    """Persistent local set-builder process for repeated search decisions."""

    def __init__(
        self, *, format_id: str, showdown_path: str | Path,
        timeout_seconds: float = 15.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("hidden-team sampling timeout must be positive")
        self.format_id = format_id
        self.showdown_path = Path(showdown_path).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.script = Path(__file__).resolve().parents[2] / "scripts" / "showdown_sample_hidden_teams.js"
        self.process = self._start()

    def _start(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["node", str(self.script), self.format_id],
            cwd=self.script.parents[1],
            env={
                **os.environ,
                "POKEMON_SHOWDOWN_PATH": str(self.showdown_path),
                "POKE_SEARCH_PERSISTENT": "1",
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def sample(
        self, request: dict[str, Any], *, timeout_seconds: float | None = None
    ) -> list[list[dict[str, Any]]]:
        if self.process.poll() is not None:
            self.process = self._start()
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("persistent hidden-team sampler streams are unavailable")
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
            raise TimeoutError(
                f"persistent hidden-team sampling exceeded {timeout:g} seconds"
            )
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"hidden-team sampler exited with status {self.process.poll()}")
        payload = json.loads(line)
        if "error" in payload:
            raise RuntimeError(f"hidden-team sampling failed: {payload['error']}\n{payload.get('stack', '')}")
        return payload["teams"]

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)


class ClosedSheetBeliefTracker:
    """Accumulate only information observable to the current player."""

    def __init__(self, preview_species: Iterable[str] = ()) -> None:
        self.pokemon: dict[str, RevealedPokemon] = {}
        self._public_ident_species: dict[str, str] = {}
        self._first_species_by_active_slot: dict[str, str] = {}
        self._appeared_species: list[str] = []
        self._transformed_idents: set[str] = set()
        self._ability_changed_idents: set[str] = set()
        self._copied_ability_idents: set[str] = set()
        self._gen = 9
        for species in preview_species:
            self.add_preview_species(str(species))

    def add_preview_species(self, species: str) -> None:
        key = to_id_str(species)
        if key and key not in self.pokemon:
            self.pokemon[key] = RevealedPokemon(species=species)

    def _known_preview_key(self, species: str) -> str | None:
        """Map a public battle forme back to one preview species, if unique."""
        key = to_id_str(species)
        if key in self.pokemon:
            return key
        dex = GenData.from_gen(self._gen).pokedex
        entry = dex.get(key, {})
        for field_name in ("battleOnly", "baseSpecies"):
            candidate = to_id_str(entry.get(field_name, ""))
            if candidate in self.pokemon:
                return candidate
        base = to_id_str(entry.get("baseSpecies", ""))
        if not base:
            return None
        matches = []
        for preview_key, preview in self.pokemon.items():
            preview_entry = dex.get(preview_key, {})
            preview_base = to_id_str(
                preview_entry.get("battleOnly")
                or preview_entry.get("baseSpecies")
                or preview.species
            )
            if preview_base == base:
                matches.append(preview_key)
        return matches[0] if len(matches) == 1 else None

    def update_from_battle(self, battle: Any) -> None:
        """Merge only team-preview species and explicit public protocol events.

        Never inspect ``battle.opponent_team``: the headless correctness
        simulator keeps the true hidden sets there, even though poke-env's
        public parser normally presents only revealed details. Reading it
        makes this tracker unsafe when the same code runs against the simulator.
        """
        preview = getattr(battle, "teampreview_opponent_team", ()) or ()
        for pokemon in preview:
            species = getattr(pokemon, "species", None)
            if species:
                self.add_preview_species(str(species))
                self._gen = int(getattr(battle, "gen", self._gen) or self._gen)
        role = str(getattr(battle, "player_role", "p1")).lower()
        opponent_side = "p2" if role.startswith("p1") else "p1"
        public_events = getattr(battle, "_replay_data", ()) or ()
        # poke-env exposes the complete public replay, not only the newest
        # events. Rebuild facts from that public history so transformed move
        # lists from an earlier update cannot remain attached to the base set.
        self._public_ident_species.clear()
        self._first_species_by_active_slot.clear()
        self._appeared_species.clear()
        self._transformed_idents.clear()
        self._ability_changed_idents.clear()
        self._copied_ability_idents.clear()
        for observed in self.pokemon.values():
            observed.moves.clear()
            observed.item = None
            observed.ability = None
            observed.active_ability = None
            observed.tera_type = None
            observed.status = None
            observed.hp_fraction = None
        self.update_from_public_events(public_events, opponent_side=opponent_side)

    def update_from_public_events(
        self, events: Iterable[Any], *, opponent_side: str = "p2"
    ) -> None:
        """Read species, moves, items, and abilities from public protocol lines."""
        opponent_side = "p2" if str(opponent_side).lower().startswith("p2") else "p1"
        for raw_event in events:
            if isinstance(raw_event, (list, tuple)):
                parts = [str(part).strip() for part in raw_event]
            else:
                parts = [part.strip() for part in str(raw_event).split("|")]
            if len(parts) < 3:
                continue
            event = parts[1]
            actor = parts[2]
            if ":" not in actor:
                continue
            side_and_slot, nickname = actor.split(":", 1)
            side_and_slot = side_and_slot.strip().lower()
            if not side_and_slot.startswith(opponent_side):
                continue
            ident_key = to_id_str(actor)
            if event in {"switch", "drag", "replace"} and len(parts) >= 4:
                self._transformed_idents.discard(ident_key)
                self._ability_changed_idents.discard(ident_key)
                self._copied_ability_idents.discard(ident_key)
                species = parts[3].split(",", 1)[0].strip()
                if species:
                    species_key = self._known_preview_key(species)
                    # With a full public preview available, a switch event may
                    # identify a Mega or battle forme. Canonicalize that form
                    # to its preview species; never expand the public roster
                    # with a battle-only forme or a non-preview species.
                    if species_key is None:
                        if self.pokemon:
                            continue
                        self.add_preview_species(species)
                        species_key = to_id_str(species)
                    self._public_ident_species[ident_key] = species_key
                    observed = self.pokemon.get(species_key)
                    if observed is not None and len(parts) >= 5:
                        _apply_public_condition(observed, parts[4])
                    if observed is not None:
                        observed.active_ability = None
                    if species_key not in self._appeared_species:
                        self._appeared_species.append(species_key)
                    # Preserve the publicly observed opening leads across later
                    # switches. A `replace` event (for example, Illusion
                    # breaking) corrects the species attached to that slot.
                    if side_and_slot[-1:] in {"a", "b"}:
                        slot_key = side_and_slot
                        if event == "replace" and slot_key in self._first_species_by_active_slot:
                            old_species = self._first_species_by_active_slot[slot_key]
                            if old_species in self._appeared_species and old_species not in self._public_ident_species.values():
                                self._appeared_species.remove(old_species)
                            self._first_species_by_active_slot[slot_key] = species_key
                        else:
                            self._first_species_by_active_slot.setdefault(slot_key, species_key)
                    continue
            if event in {"-mega", "-formechange", "detailschange"}:
                # The announced ability after a battle-time forme change is
                # the active forme's ability, not necessarily the set ability.
                self._ability_changed_idents.add(ident_key)
                continue
            if event == "-transform":
                self._transformed_idents.add(ident_key)
                self._ability_changed_idents.add(ident_key)
                continue
            species_key = self._public_ident_species.get(ident_key)
            if species_key is None:
                continue
            observed = self.pokemon.get(species_key)
            if observed is None:
                continue
            if event in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
                _apply_public_condition(observed, parts[3])
            elif event == "faint":
                observed.hp_fraction = 0.0
            elif event == "-status" and len(parts) >= 4:
                status = to_id_str(parts[3])
                if status in _PUBLIC_STATUSES:
                    observed.status = status
            elif event == "-curestatus":
                observed.status = None
            elif event == "-terastallize" and len(parts) >= 4:
                tera_type = str(parts[3]).strip()
                if tera_type and to_id_str(tera_type) not in {"unknown", "none"}:
                    observed.tera_type = tera_type
            elif event == "move" and len(parts) >= 4:
                move = to_id_str(parts[3])
                move_metadata = " ".join(parts[4:]).lower()
                copied_or_called = "[from]" in move_metadata and any(
                    source in move_metadata
                    for source in ("move:", "ability:", "item:")
                )
                # A battle can expose moves granted by Transform, Mimic,
                # Copycat, and similar effects without consistently tagging
                # every subsequent use as copied. Such moves are public to the
                # active battle state, but cannot condition the base set when
                # Showdown's Gen 9 learnset says the preview species cannot
                # learn them.
                learnset = GenData.from_gen(self._gen).learnset.get(
                    species_key, {}
                ).get("learnset", {})
                legal_set_move = move in learnset
                if (
                    move
                    and move != "struggle"
                    and legal_set_move
                    and ident_key not in self._transformed_idents
                    and not copied_or_called
                ):
                    observed.moves.add(move)
            elif event in {"-item", "-enditem"} and len(parts) >= 4:
                metadata = " ".join(parts[4:]).lower()
                if any(source in metadata for source in (
                    "move: trick", "move: switcheroo", "ability: symbiosis",
                )):
                    continue
                item = to_id_str(parts[3])
                if item and item not in {"unknownitem", "unknown", "none"}:
                    observed.item = item
            elif event == "-ability" and len(parts) >= 4:
                # Trace, Receiver, Power of Alchemy, and similar effects expose
                # a temporary copied ability, not the opponent's set ability.
                metadata = " ".join(parts[4:]).lower()
                if (
                    "[from]" in metadata
                    or any(str(part).strip().lower() in {"trace", "receiver", "power of alchemy"}
                           for part in parts[4:])
                ):
                    self._copied_ability_idents.add(ident_key)
                ability = to_id_str(parts[3])
                if ability and ability not in {"unknownability", "unknown", "none"}:
                    observed.active_ability = ability
                if (
                    "[from]" in metadata
                    or ident_key in self._ability_changed_idents
                    or ident_key in self._transformed_idents
                    or ident_key in self._copied_ability_idents
                ):
                    continue
                if ability and ability not in {"unknownability", "unknown", "none"}:
                    if (
                        ident_key not in self._ability_changed_idents
                        and ident_key not in self._transformed_idents
                        and ident_key not in self._copied_ability_idents
                    ):
                        entry = GenData.from_gen(self._gen).pokedex.get(species_key, {})
                        legal_abilities = {
                            to_id_str(value)
                            for value in entry.get("abilities", {}).values()
                        }
                        if ability in legal_abilities:
                            observed.ability = ability

    def public_roster(self) -> list[str]:
        """Return preview/revealed species only, in first-seen order."""
        return [row.species for row in self.pokemon.values()]

    def appeared_species(self) -> list[str]:
        """Return species publicly sent into battle, in first-appearance order."""
        return [self.pokemon[key].species for key in self._appeared_species if key in self.pokemon]

    def opening_leads(self) -> list[str]:
        """Return the two opening species from public slot events, when known."""
        ordered_slots = sorted(self._first_species_by_active_slot)
        return [
            self.pokemon[key].species
            for slot in ordered_slots
            if (key := self._first_species_by_active_slot[slot]) in self.pokemon
        ]

    def public_facts(self) -> dict[str, dict[str, Any]]:
        """Return a serializable player-visible projection."""
        return {
            key: {
                "species": value.species,
                "moves": sorted(value.moves),
                "item": value.item,
                "ability": value.ability,
                "active_ability": value.active_ability,
                "hp_fraction": value.hp_fraction,
                "status": value.status,
                "tera_type": value.tera_type,
            }
            for key, value in self.pokemon.items()
        }


def possible_selected_rosters(
    preview_species: Sequence[str],
    opening_leads: Sequence[str],
    appeared_species: Sequence[str],
) -> list[list[str]]:
    """Enumerate public-compatible four-Pokemon selections and back orders.

    The opening lead order is public. The two back Pokemon and their order are
    private until revealed, so this returns every selection consistent with
    the species seen in battle. Inputs contain species names only; no hidden
    simulator roster fields are consulted.
    """
    if len(preview_species) != 6 or len({to_id_str(row) for row in preview_species}) != 6:
        raise ValueError("preview_species must contain six unique species")
    if len(opening_leads) != 2 or len({to_id_str(row) for row in opening_leads}) != 2:
        raise ValueError("opening_leads must contain two distinct species")
    preview_by_id = {to_id_str(row): str(row) for row in preview_species}
    lead_ids = [to_id_str(row) for row in opening_leads]
    if any(row not in preview_by_id for row in lead_ids):
        raise ValueError("opening leads must come from the public preview")
    appeared_ids = {to_id_str(row) for row in appeared_species}
    if not appeared_ids.issubset(preview_by_id):
        raise ValueError("appeared species must come from the public preview")
    required_backs = appeared_ids.difference(lead_ids)
    if len(required_backs) > 2:
        raise ValueError("more than two publicly appeared non-leads cannot be selected")
    back_pool = [row for row in preview_by_id if row not in lead_ids]
    selections: list[list[str]] = []
    for pair in combinations(back_pool, 2):
        if not required_backs.issubset(pair):
            continue
        for ordered_pair in permutations(pair):
            selections.append([
                preview_by_id[lead_ids[0]], preview_by_id[lead_ids[1]],
                preview_by_id[ordered_pair[0]], preview_by_id[ordered_pair[1]],
            ])
    if not selections:
        raise ValueError("no four-Pokemon selection is consistent with public facts")
    return selections


def set_pool_from_player_views(records: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Extract unique full teams only from schema-v2 player-owned team fields."""
    # Import lazily: model imports load the belief package, while the training
    # package exports load the model. A module-level import creates a cycle.
    from src.training.perspective_data import is_perspective_safe_record

    seen: set[tuple[str, ...]] = set()
    teams: list[list[dict[str, Any]]] = []
    for record in records:
        if not is_perspective_safe_record(record):
            raise ValueError("hidden-set pool input must contain safe schema-v2 player views")
        side = record["perspective_team"]
        members = record["teams"][side]["pokemon"]
        if len(members) != 6:
            continue
        copied = [dict(member) for member in members]
        key = tuple(to_id_str(member.get("species", "")) for member in copied)
        if key not in seen:
            seen.add(key)
            teams.append(copied)
    return teams


def read_player_view_set_pool(path: str | Path) -> list[list[dict[str, Any]]]:
    """Read safe player views from JSONL and build a legal-set source pool."""
    with open_jsonl_binary(path) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return set_pool_from_player_views(records)


def sample_hidden_team_hypotheses(
    preview_species: Sequence[str],
    set_pool: Sequence[Sequence[dict[str, Any]]],
    revealed_facts: dict[str, dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    unique_items: bool = True,
) -> list[list[dict[str, Any]]]:
    """Sample uniform, public-fact-compatible team hypotheses.

    Set variants are drawn from the owned-team pool without using opponent
    action history or behavioral tendencies. The pool is an empirical legal
    set source, so these hypotheses are approximate and may not cover every
    valid set in the format.
    """
    if count < 1 or len(preview_species) != 6:
        raise ValueError("count must be positive and preview_species must have six entries")
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for team in set_pool:
        for pokemon in team:
            species = to_id_str(pokemon.get("species", ""))
            if species:
                by_species[species].append(dict(pokemon))

    candidates: list[list[dict[str, Any]]] = []
    for species_name in preview_species:
        species = to_id_str(species_name)
        facts = revealed_facts.get(species, {})
        known_moves = {to_id_str(move) for move in facts.get("moves", ())}
        moves_exact = bool(facts.get("moves_exact"))
        known_item = to_id_str(facts.get("item") or "")
        known_ability = to_id_str(facts.get("ability") or "")
        known_tera_type = to_id_str(facts.get("tera_type") or "")
        valid = []
        for member in by_species.get(species, ()):
            moves = {to_id_str(move) for move in member.get("moves", ())}
            if moves_exact and moves != known_moves:
                continue
            if known_moves and not known_moves.issubset(moves):
                continue
            if known_item and to_id_str(member.get("item", "")) != known_item:
                continue
            if known_ability and to_id_str(member.get("ability", "")) != known_ability:
                continue
            if known_tera_type and to_id_str(member.get("teraType", "")) != known_tera_type:
                continue
            valid.append(member)
        if not valid:
            raise ValueError(f"set pool has no hypothesis matching revealed facts for {species_name}")
        candidates.append(valid)

    samples = []
    for _ in range(count):
        order = list(range(6))
        rng.shuffle(order)
        selected: list[dict[str, Any] | None] = [None] * 6

        def assign(depth: int, used_items: set[str]) -> bool:
            if depth == len(order):
                return True
            slot = order[depth]
            options = list(candidates[slot])
            rng.shuffle(options)
            for member in options:
                item = to_id_str(member.get("item", ""))
                if unique_items and item and item in used_items:
                    continue
                selected[slot] = dict(member)
                if assign(depth + 1, used_items | ({item} if item else set())):
                    return True
            return False

        if not assign(0, set()):
            raise ValueError("set pool cannot form an item-legal hypothesis for the revealed roster")
        samples.append([member for member in selected if member is not None])
    return samples


def sample_showdown_hidden_teams(
    preview_species: Sequence[str],
    revealed_facts: dict[str, dict[str, Any]],
    *,
    format_id: str,
    count: int,
    seed: int,
    set_prior: str = "showdown_random_champions",
    showdown_path: str | Path | None = None,
    attempts_per_team: int = 500,
    sampler: ShowdownHiddenTeamSampler | None = None,
    timeout_seconds: float = 15.0,
) -> list[list[dict[str, Any]]]:
    """Sample validator-approved hypotheses using Showdown's local set builder."""
    root_value = showdown_path or os.environ.get("POKEMON_SHOWDOWN_PATH")
    if not root_value:
        raise ValueError("showdown_path or POKEMON_SHOWDOWN_PATH is required")
    root = Path(root_value)
    request = {
        "preview_species": list(preview_species),
        "revealed_facts": revealed_facts,
        "attempts_per_team": attempts_per_team,
        "seed": int(seed),
        "count": int(count),
        "set_prior": set_prior,
    }
    if sampler is not None:
        if sampler.format_id != format_id or sampler.showdown_path != root.resolve():
            raise ValueError("persistent sampler format or local Showdown path mismatch")
        return sampler.sample(request, timeout_seconds=timeout_seconds)
    try:
        result = subprocess.run(
            ["node", str(Path(__file__).resolve().parents[2] / "scripts" / "showdown_sample_hidden_teams.js"), format_id, str(seed), str(count)],
            input=json.dumps(request),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "POKEMON_SHOWDOWN_PATH": str(root.resolve())},
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "no sampler output").strip()
        raise RuntimeError(
            f"Showdown hidden-team sampling failed for prior={set_prior}, "
            f"seed={seed}: {detail}"
        ) from error
    payload = json.loads(result.stdout)
    return payload["teams"]
