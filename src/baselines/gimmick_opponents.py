"""Scripted legal gimmick opponents for controlled robustness evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.baselines.native_damage import choose_max_damage_action
from src.training.native_ppo import decode_native_action_index


def _legal_move_action_indices(
    mask: np.ndarray, battle: Any, position: int, move_id: str
) -> list[int]:
    active = battle.active_pokemon[position]
    if active is None:
        return []
    known_moves = list(active.moves.values())[:4]
    available_moves = list(battle.available_moves[position])
    available_ids = [move.id for move in available_moves]
    known_ids = [move.id for move in known_moves]
    move_list = (
        available_moves
        if len(available_ids) == 1 and available_ids[0] not in known_ids
        else known_moves
    )
    half = mask.size // 2
    start = position * half
    actions = []
    for index in np.flatnonzero(mask[start:start + half]):
        descriptor = decode_native_action_index(int(index))
        if (descriptor.kind == "move" and descriptor.mechanic == "base"
                and descriptor.move_index is not None
                and descriptor.move_index < len(move_list)
                and move_list[descriptor.move_index].id == move_id):
            actions.append(int(index))
    return actions


def choose_no_guard_fissure_action(action_mask: Any, battle: Any) -> np.ndarray:
    """Set up Skill Swap No Guard on Snorlax, then use Fissure.

    This bot brings Gardevoir and Machamp as leads, with Snorlax in the back.
    Machamp uses Wide Guard while Gardevoir takes No Guard, then Machamp
    switches to Snorlax as Gardevoir transfers the ability. Snorlax uses
    Fissure against a legal grounded active target. If the setup is disrupted,
    it falls back to the max-damage baseline.
    """
    mask = np.asarray(action_mask, dtype=np.int8)
    chosen = np.asarray(choose_max_damage_action(mask, battle), dtype=np.int64).copy()
    species = [
        str(getattr(mon, "species", "")).lower().replace("-", "").replace(" ", "")
        if mon is not None else ""
        for mon in battle.active_pokemon[:2]
    ]
    abilities = [
        str(getattr(mon, "ability", "") or "").lower().replace("-", "").replace(" ", "")
        if mon is not None else ""
        for mon in battle.active_pokemon[:2]
    ]
    team = list(battle.team.values())
    snorlax_index = next(
        (index for index, mon in enumerate(team)
         if str(getattr(mon, "species", "")).lower() == "snorlax"),
        None,
    )

    if "gardevoir" in species and "machamp" in species:
        gardevoir_position = species.index("gardevoir")
        machamp_position = species.index("machamp")
        partner_skill_swap = next(
            (action for action in _legal_move_action_indices(
                mask, battle, gardevoir_position, "skillswap"
            ) if decode_native_action_index(action).target_code == -1),
            None,
        )
        if partner_skill_swap is not None and abilities[machamp_position] == "noguard":
            chosen[gardevoir_position] = partner_skill_swap
            wide_guard = _legal_move_action_indices(mask, battle, machamp_position, "wideguard")
            if wide_guard:
                chosen[machamp_position] = wide_guard[0]
        elif partner_skill_swap is not None and abilities[gardevoir_position] == "noguard":
            chosen[gardevoir_position] = partner_skill_swap
            if snorlax_index is not None:
                switch_action = snorlax_index + 1
                half = mask.size // 2
                if mask[machamp_position * half + switch_action]:
                    chosen[machamp_position] = switch_action
    elif "gardevoir" in species and "snorlax" in species:
        gardevoir_position = species.index("gardevoir")
        snorlax_position = species.index("snorlax")
        fissure = _legal_move_action_indices(mask, battle, snorlax_position, "fissure")
        opposing_active = list(getattr(battle, "opponent_active_pokemon", ())[:2])
        grounded_targets = []
        for action in fissure:
            target_code = decode_native_action_index(action).target_code
            if target_code not in (1, 2):
                continue
            target_position = target_code - 1
            if target_position >= len(opposing_active):
                continue
            target = opposing_active[target_position]
            if target is None:
                continue
            target_types = {
                str(getattr(value, "name", value)).lower()
                for value in (getattr(target, "types", ()) or ())
            }
            target_ability = str(getattr(target, "ability", "") or "").lower()
            if "flying" not in target_types and target_ability != "levitate":
                grounded_targets.append(action)

        if abilities[snorlax_position] == "noguard" and grounded_targets:
            chosen[snorlax_position] = grounded_targets[0]
        elif abilities[gardevoir_position] == "noguard":
            partner_skill_swap = next(
                (action for action in _legal_move_action_indices(
                    mask, battle, gardevoir_position, "skillswap"
                ) if decode_native_action_index(action).target_code == -1),
                None,
            )
            if partner_skill_swap is not None:
                chosen[gardevoir_position] = partner_skill_swap
            body_slam = _legal_move_action_indices(mask, battle, snorlax_position, "bodyslam")
            if body_slam:
                chosen[snorlax_position] = body_slam[0]

    return chosen


def choose_perish_trap_action(action_mask: Any, battle: Any) -> np.ndarray:
    """Use Perish Song and Mean Look on a fixed Perish Trap roster.

    This is a controlled test opponent, not a general battle policy: it traps
    an opposing active with Mean Look, applies Perish Song, protects at counter
    two, and switches its own Pokemon before their counter reaches zero. The
    fixed roster includes two reserve Pokemon for cycling the Perish Song users.
    """
    mask = np.asarray(action_mask, dtype=np.int8)
    chosen = np.asarray(choose_max_damage_action(mask, battle), dtype=np.int64).copy()
    opponent_active = list(getattr(battle, "opponent_active_pokemon", ())[:2])
    team = list(battle.team.values())
    half = mask.size // 2

    def effects_of(pokemon: Any) -> set[str]:
        effects = getattr(pokemon, "effects", {}) or {}
        return {
            getattr(effect, "name", str(effect)).upper()
            .replace("_", "").replace("-", "")
            for effect in effects
        }

    def perish_counter(pokemon: Any) -> int | None:
        names = effects_of(pokemon)
        return next((counter for counter in range(4)
                     if f"PERISH{counter}" in names), None)

    reserved_team_indices: set[int] = set()

    def switch_to_reserve(position: int) -> bool:
        active = battle.active_pokemon[position]
        for team_index, reserve in enumerate(team):
            if (reserve is active or team_index in reserved_team_indices
                    or getattr(reserve, "fainted", False)):
                continue
            switch_action = team_index + 1
            if mask[position * half + switch_action]:
                chosen[position] = switch_action
                reserved_team_indices.add(team_index)
                return True
        return False

    # Escape the user's own countdown on the last safe turn. This preserves
    # the target's Mean Look volatile while cycling the Song user out.
    imminent_switches: set[int] = set()
    for position in range(2):
        active = battle.active_pokemon[position]
        counter = perish_counter(active) if active is not None else None
        if counter is not None and counter <= 1 and switch_to_reserve(position):
            imminent_switches.add(position)

    # Keep a trapped opponent from switching, then apply Perish Song to any
    # opposing active that does not already have its countdown.
    opposing_needs_song = any(
        target is not None and perish_counter(target) is None
        for target in opponent_active
    )
    for position in range(2):
        if position in imminent_switches or battle.active_pokemon[position] is None:
            continue
        mean_look = _legal_move_action_indices(mask, battle, position, "meanlook")
        for action in mean_look:
            target_code = decode_native_action_index(action).target_code
            if target_code not in (1, 2):
                continue
            target_position = target_code - 1
            if target_position >= len(opponent_active):
                continue
            target = opponent_active[target_position]
            if target is not None and "TRAPPED" not in effects_of(target):
                chosen[position] = action
                break
        else:
            active = battle.active_pokemon[position]
            counter = perish_counter(active)
            if opposing_needs_song:
                perish = _legal_move_action_indices(mask, battle, position, "perishsong")
                if perish:
                    chosen[position] = perish[0]
                    continue
            # Protect at counter two, then switch on the next turn. This also
            # spends a turn while the trapped opponent's counter advances.
            if counter is not None and counter <= 2:
                protect = _legal_move_action_indices(mask, battle, position, "protect")
                if protect:
                    chosen[position] = protect[0]
    return chosen
