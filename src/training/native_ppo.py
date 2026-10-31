"""Utilities for turning native Showdown trajectories into PPO batches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from src.training.perspective_data import is_perspective_safe_record
from src.features.public_mechanics import encode_public_mechanics
from src.features.public_mechanics import prepare_public_mechanics_static
from src.training.ppo import compute_gae, ppo_policy_loss, ppo_value_loss

MODEL_NATIVE_ACTIONS_PER_SLOT = 126
POKE_ENV_NATIVE_ACTIONS_PER_SLOT = 107
NATIVE_ACTION_BASE_FEATURE_VARIANT = "damage_ko"
NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT = "damage_ko_uncertainty"
NATIVE_ACTION_STATCALC_FEATURE_VARIANT = "damage_ko_uncertainty_statcalc"
NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT = "damage_ko_incoming_threat"
NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT = "damage_ko_counterplay"
NATIVE_ACTION_FRIENDLY_FIRE_FEATURE_VARIANT = "damage_ko_friendly_fire"
NATIVE_ACTION_FEATURE_VARIANTS: dict[str, int] = {
    NATIVE_ACTION_BASE_FEATURE_VARIANT: 4,
    NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT: 6,
    NATIVE_ACTION_STATCALC_FEATURE_VARIANT: 6,
    NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT: 6,
    NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT: 7,
    NATIVE_ACTION_FRIENDLY_FIRE_FEATURE_VARIANT: 5,
}
NATIVE_ACTION_FEATURE_DIM = NATIVE_ACTION_FEATURE_VARIANTS[NATIVE_ACTION_BASE_FEATURE_VARIANT]
NATIVE_ACTION_FEATURE_ENCODER_VERSION = 1


def normalize_counterplay_score_row(
    scores: Sequence[float] | np.ndarray,
    legal_action_mask: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Standardize one conditional counterplay-score row over legal actions."""
    score_array = np.asarray(scores, dtype=np.float32)
    mask_array = np.asarray(legal_action_mask, dtype=bool)
    if score_array.shape != (MODEL_NATIVE_ACTIONS_PER_SLOT,) or mask_array.shape != score_array.shape:
        raise ValueError("counterplay scores and legal mask must both have shape (126,)")
    supported = mask_array & np.isfinite(score_array) & (score_array > -1e5)
    normalized = np.zeros_like(score_array)
    if supported.any():
        scale = max(float(score_array[supported].std()), 1e-6)
        normalized[supported] = np.clip(
            (score_array[supported] - float(score_array[supported].mean())) / scale,
            -4.0, 4.0,
        )
    return normalized


@dataclass(frozen=True)
class NativeTransition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: np.ndarray
    reward: float
    old_log_prob: float
    value: float
    done: bool
    terminated: bool = False
    truncated: bool = False
    native_action_features: np.ndarray | None = None
    counterplay_action_scores: np.ndarray | None = None
    counterplay_teacher_first_action: int | None = None
    reward_components: dict[str, float] | None = None
    public_context_features: np.ndarray | None = None
    opponent_species: tuple[str, ...] | None = None
    step_index: int = -1
    teacher_actions: np.ndarray | None = None
    teacher_joint_actions: np.ndarray | None = None
    teacher_joint_action_probabilities: np.ndarray | None = None
    hidden_move_targets: tuple[tuple[str, ...], ...] | None = None
    hidden_move_masks: tuple[tuple[bool, ...], ...] | None = None
    hidden_item_targets: tuple[str, ...] | None = None
    hidden_item_masks: tuple[bool, ...] | None = None
    hidden_ability_targets: tuple[str, ...] | None = None
    hidden_ability_masks: tuple[bool, ...] | None = None
    win_target: float | None = None


@dataclass(frozen=True)
class NativeActionDescriptor:
    kind: str
    switch_index: int | None = None
    move_index: int | None = None
    target_code: int | None = None
    mechanic: str | None = None


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace(" ", "-")


def _team_members(team: dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(team, dict):
        return list(team.get("pokemon", []))
    return list(team)


def _clip_scalar(value: float, lower: float, upper: float) -> float:
    """Clamp one finite scalar without dispatching through NumPy."""
    value = float(value)
    if math.isnan(value):
        return value
    return min(max(value, lower), upper)


@lru_cache(maxsize=1)
def _gen_data() -> GenData:
    return GenData.from_gen(9)


def _pokedex_entry(name: str | None) -> dict[str, Any]:
    return _gen_data().pokedex.get(_normalize_name(name), {})


def _move_entry(name: str | None) -> dict[str, Any]:
    normalized = _normalize_name(name)
    if normalized == "protect":
        normalized = "protected"
    moves = _gen_data().moves
    entry = moves.get(normalized)
    if entry:
        return entry
    # Showdown's and poke-env's identifiers omit separators (earthpower,
    # solarbeam), while human-readable move names may contain spaces.
    return moves.get(normalized.replace("-", ""), {})


def _pokemon_stat(member: dict[str, Any], base_stats: dict[str, Any], stat: str) -> float:
    """Calculate a level-adjusted stat from the generated set's IVs, EVs, and nature."""
    level = max(1, int(member.get("level", 50) or 50))
    ivs = member.get("ivs", {}) or {}
    evs = member.get("evs", {}) or {}
    base = float(base_stats.get(stat, 85))
    iv = float(ivs.get(stat, 31))
    ev = float(evs.get(stat, 0))
    raw = math.floor(((2.0 * base + iv + math.floor(ev / 4.0)) * level) / 100.0)
    if stat == "hp":
        return float(raw + level + 10)
    nature_name = _normalize_name(member.get("nature"))
    nature = _gen_data().natures.get(nature_name, {})
    modifier = float(nature.get(stat, 1.0))
    return float(math.floor((raw + 5.0) * modifier))


def _boost_multiplier(stage: float) -> float:
    stage = _clip_scalar(stage, -6.0, 6.0)
    return (2.0 + stage) / 2.0 if stage >= 0.0 else 2.0 / (2.0 - stage)


def _screen_damage_multiplier(
    state: np.ndarray,
    category: str,
    *,
    defender_is_opponent: bool,
    infiltrates: bool = False,
) -> float:
    """Apply active Gen 9 doubles screen mitigation to estimated damage."""
    if infiltrates or state.size < 106:
        return 1.0
    start = 103 if defender_is_opponent else 100
    reflect, light_screen, aurora_veil = state[start : start + 3] > 0.5
    category = category.lower()
    if (
        (reflect and category == "physical")
        or (light_screen and category == "special")
        or aurora_veil
    ):
        # Showdown's Gen 9 doubles screen modifier is 2732/4096 (~2/3).
        return 2732.0 / 4096.0
    return 1.0


def _incoming_threats(
    player_members: Sequence[dict[str, Any]],
    opponent_members: Sequence[dict[str, Any]],
    state: np.ndarray,
    own_hp: np.ndarray,
    own_active_hp: np.ndarray,
    own_active_indices: Sequence[int],
    opponent_active_indices: Sequence[int],
) -> np.ndarray:
    """Estimate the strongest revealed move into each team member this turn."""
    threats = np.zeros((6, 2), dtype=np.float32)
    weather = state[52:56] if state.size >= 56 else np.zeros(4, dtype=np.float32)
    for defender_index, defender in enumerate(player_members[:6]):
        defender_name = defender.get("species") or defender.get("name")
        defender_entry = _pokedex_entry(defender_name)
        defender_types = tuple(defender_entry.get("types", ()))
        defender_stats = defender_entry.get("baseStats", {})
        active_slot = next(
            (slot for slot, index in enumerate(own_active_indices) if index == defender_index),
            None,
        )
        current_hp = (
            float(own_active_hp[active_slot]) if active_slot is not None and active_slot < len(own_active_hp)
            else float(own_hp[defender_index]) if defender_index < len(own_hp) else 0.0
        )
        worst_damage = 0.0
        for opponent_slot, attacker_index in enumerate(opponent_active_indices[:2]):
            if attacker_index >= len(opponent_members):
                continue
            attacker = opponent_members[attacker_index]
            attacker_name = attacker.get("species") or attacker.get("name")
            attacker_entry = _pokedex_entry(attacker_name)
            attacker_types = tuple(attacker_entry.get("types", ()))
            attacker_stats = attacker_entry.get("baseStats", {})
            for move_name in attacker.get("moves", ()):
                move = _move_entry(move_name)
                power = float(move.get("basePower") or 0.0)
                category = move.get("category")
                if power <= 0.0 or category not in {"Physical", "Special"}:
                    continue
                attack_name = "spa" if category == "Special" else "atk"
                defense_name = "spd" if category == "Special" else "def"
                attack = _pokemon_stat(attacker, attacker_stats, attack_name)
                defense = _pokemon_stat(defender, defender_stats, defense_name)
                if state.size >= 52:
                    attacker_stage = float(state[32 + (2 + opponent_slot) * 5
                                                  + (2 if category == "Special" else 0)]) * 6.0
                    attack *= _boost_multiplier(attacker_stage)
                    if active_slot is not None:
                        defender_stage = float(state[32 + active_slot * 5
                                                     + (3 if category == "Special" else 1)]) * 6.0
                        defense *= _boost_multiplier(defender_stage)
                stab = 1.5 if move.get("type") in attacker_types else 1.0
                effectiveness = _type_multiplier(move.get("type"), defender_types)
                spread = 0.75 if move.get("target") in {"allAdjacent", "allAdjacentFoes"} else 1.0
                accuracy = move.get("accuracy", 100)
                accuracy_factor = 1.0 if accuracy is True else float(
                    _clip_scalar(float(accuracy) / 100.0, 0.0, 1.0)
                )
                level = max(1, int(attacker.get("level", 50) or 50))
                damage = (((2.0 * level / 5.0 + 2.0) * power * attack
                           / max(defense, 1.0)) / 50.0) + 2.0
                damage *= stab * effectiveness * spread * accuracy_factor * 0.925
                move_type = str(move.get("type", "")).lower()
                if move_type == "water":
                    damage *= 1.5 if weather[0] else 0.5 if weather[1] else 1.0
                elif move_type == "fire":
                    damage *= 1.5 if weather[1] else 0.5 if weather[0] else 1.0
                damage *= _screen_damage_multiplier(
                    state, category, defender_is_opponent=False,
                    infiltrates=bool(move.get("flags", {}).get("infiltrates")),
                )
                damage_fraction = damage / max(_pokemon_stat(defender, defender_stats, "hp"), 1.0)
                worst_damage = max(worst_damage, damage_fraction)
        threats[defender_index, 0] = _clip_scalar(worst_damage, 0.0, 1.0)
        threats[defender_index, 1] = _clip_scalar(worst_damage / max(current_hp, 0.05), 0.0, 1.0)
    return threats


def _predicted_opponent_action_threats(
    player_members: Sequence[dict[str, Any]],
    opponent_members: Sequence[dict[str, Any]],
    state: np.ndarray,
    own_hp: np.ndarray,
    own_active_hp: np.ndarray,
    own_active_indices: Sequence[int],
    opponent_active_indices: Sequence[int],
    opponent_action: Sequence[int],
) -> np.ndarray:
    """Estimate damage from the opponent's actual max-damage action pair.

    The result is indexed by our roster member, then the active slot it would
    occupy, then ``[damage/max_hp, damage/current_hp]``. Target selection is
    therefore aligned with the deterministic heuristic's chosen moves.
    """
    threats = np.zeros((6, 2, 2), dtype=np.float32)
    weather = state[52:56] if state.size >= 56 else np.zeros(4, dtype=np.float32)
    for opponent_slot, action in enumerate(opponent_action[:2]):
        descriptor = decode_native_action_index(int(action))
        if descriptor.kind != "move" or descriptor.move_index is None:
            continue
        if opponent_slot >= len(opponent_active_indices):
            continue
        attacker_index = int(opponent_active_indices[opponent_slot])
        if attacker_index >= len(opponent_members):
            continue
        attacker = opponent_members[attacker_index]
        move_names = list(attacker.get("moves", ()))
        if descriptor.move_index >= len(move_names):
            continue
        move = _move_entry(move_names[descriptor.move_index])
        power = float(move.get("basePower") or 0.0)
        category = move.get("category")
        if power <= 0.0 or category not in {"Physical", "Special"}:
            continue
        target = move.get("target")
        is_spread = target in {"allAdjacent", "allAdjacentFoes", "all"}
        if is_spread:
            hit_slots = (0, 1)
            spread_factor = 0.75
        elif descriptor.target_code in (1, 2):
            hit_slots = (int(descriptor.target_code) - 1,)
            spread_factor = 1.0
        else:
            continue

        attacker_name = attacker.get("species") or attacker.get("name")
        attacker_entry = _pokedex_entry(attacker_name)
        attacker_types = tuple(attacker_entry.get("types", ()))
        attacker_stats = attacker_entry.get("baseStats", {})
        attack_name = "spa" if category == "Special" else "atk"
        defense_name = "spd" if category == "Special" else "def"
        attack = _pokemon_stat(attacker, attacker_stats, attack_name)
        if state.size >= 52:
            attack_stage = float(
                state[32 + (2 + opponent_slot) * 5
                      + (2 if category == "Special" else 0)]
            ) * 6.0
            attack *= _boost_multiplier(attack_stage)
        status_offset = 94 + attacker_index
        if (category == "Physical" and state.size > status_offset
                and np.isclose(float(state[status_offset]), 1.0 / 6.0)):
            attack *= 0.5
        accuracy = move.get("accuracy", 100)
        accuracy_factor = 1.0 if accuracy is True else float(
            _clip_scalar(float(accuracy) / 100.0, 0.0, 1.0)
        )
        level = max(1, int(attacker.get("level", 50) or 50))

        for target_slot in hit_slots:
            for defender_index, defender in enumerate(player_members[:6]):
                currently_active_slot = next(
                    (slot for slot, index in enumerate(own_active_indices)
                     if int(index) == defender_index),
                    None,
                )
                if currently_active_slot is not None and currently_active_slot != target_slot:
                    continue
                defender_name = defender.get("species") or defender.get("name")
                defender_entry = _pokedex_entry(defender_name)
                defender_types = tuple(defender_entry.get("types", ()))
                defender_stats = defender_entry.get("baseStats", {})
                defense = _pokemon_stat(defender, defender_stats, defense_name)
                if state.size >= 52:
                    defense_stage = float(
                        state[32 + target_slot * 5
                              + (3 if category == "Special" else 1)]
                    ) * 6.0
                    defense *= _boost_multiplier(defense_stage)
                stab = 1.5 if move.get("type") in attacker_types else 1.0
                effectiveness = _type_multiplier(move.get("type"), defender_types)
                damage = (((2.0 * level / 5.0 + 2.0) * power * attack
                           / max(defense, 1.0)) / 50.0) + 2.0
                damage *= stab * effectiveness * spread_factor * accuracy_factor * 0.925
                move_type = str(move.get("type", "")).lower()
                if move_type == "water":
                    damage *= 1.5 if weather[0] else 0.5 if weather[1] else 1.0
                elif move_type == "fire":
                    damage *= 1.5 if weather[1] else 0.5 if weather[0] else 1.0
                damage *= _screen_damage_multiplier(
                    state, category, defender_is_opponent=False,
                    infiltrates=bool(move.get("flags", {}).get("infiltrates")),
                )
                max_hp = max(_pokemon_stat(defender, defender_stats, "hp"), 1.0)
                damage_fraction = _clip_scalar(damage / max_hp, 0.0, 1.0)
                current_hp = (
                    float(own_active_hp[target_slot])
                    if currently_active_slot == target_slot
                    and target_slot < len(own_active_hp)
                    else float(own_hp[defender_index])
                    if defender_index < len(own_hp)
                    else 0.0
                )
                threats[defender_index, target_slot, 0] += damage_fraction
                threats[defender_index, target_slot, 1] += damage_fraction / max(current_hp, 0.05)
    return np.clip(threats, 0.0, 1.0)


def _feature_action_speed(
    member: dict[str, Any], state: np.ndarray, *, active_position: int,
    member_index: int, opponent_side: bool,
) -> float:
    """Estimate current speed for feature-level turn-order comparisons."""
    entry = _pokedex_entry(member.get("species") or member.get("name"))
    speed = _pokemon_stat(member, entry.get("baseStats", {}), "spe")
    stage_index = 32 + active_position * 5 + 4
    if state.size > stage_index:
        speed *= _boost_multiplier(float(state[stage_index]) * 6.0)
    ability = _normalize_name(member.get("ability")).replace("-", "")
    item = _normalize_name(member.get("item")).replace("-", "")
    weather = state[52:56] if state.size >= 56 else np.zeros(4, dtype=np.float32)
    fields = state[56:60] if state.size >= 60 else np.zeros(4, dtype=np.float32)
    if ability == "swiftswim" and weather[0]:
        speed *= 2.0
    elif ability == "chlorophyll" and weather[1]:
        speed *= 2.0
    elif ability == "sandrush" and weather[2]:
        speed *= 2.0
    elif ability == "slushrush" and weather[3]:
        speed *= 2.0
    elif ability == "surgesurfer" and fields[0]:
        speed *= 2.0
    tailwind_index = 63 if opponent_side else 62
    if state.size > tailwind_index and state[tailwind_index] > 0.5:
        speed *= 2.0
    status_index = (94 if opponent_side else 64) + member_index
    has_status = state.size > status_index and state[status_index] > 0.0
    paralyzed = state.size > status_index and np.isclose(float(state[status_index]), 0.5)
    if ability == "quickfeet" and has_status:
        speed *= 1.5
    elif paralyzed:
        speed *= 0.5
    if item == "choicescarf":
        speed *= 1.5
    elif item in {
        "ironball", "machobrace", "poweranklet", "powerband", "powerbelt",
        "powerbracer", "powerlens", "powerweight",
    }:
        speed *= 0.5
    if ability == "unburden" and item in {"", "none"}:
        speed *= 2.0
    if ability == "slowstart":
        speed *= 0.5
    return speed


def _feature_flinch_probability(move: dict[str, Any], attacker: dict[str, Any], state: np.ndarray) -> float:
    move_id = _normalize_name(move.get("name") or move.get("id")).replace("-", "")
    if move_id == "fakeout" and state.size and state[0] <= 0.01:
        return 1.0
    ability = _normalize_name(attacker.get("ability")).replace("-", "")
    chance_scale = 2.0 if ability == "serenegrace" else 1.0
    chance = max(
        (float(effect.get("chance", 0.0) or 0.0) / 100.0
         for effect in move.get("secondaries", ()) or ()
         if str(effect.get("volatileStatus", "")).lower() == "flinch"),
        default=0.0,
    )
    accuracy = move.get("accuracy", 100)
    accuracy = 1.0 if accuracy is True else _clip_scalar(float(accuracy) / 100.0, 0.0, 1.0)
    return _clip_scalar(chance * chance_scale * accuracy, 0.0, 1.0)


def _type_multiplier(move_type: str | None, defender_types: Sequence[str]) -> float:
    if not move_type or not defender_types:
        return 1.0
    try:
        move_enum = PokemonType[str(move_type).upper().replace("?", "THREE_QUESTION_MARKS")]
        type_1 = PokemonType[str(defender_types[0]).upper().replace("?", "THREE_QUESTION_MARKS")]
        type_2 = (
            PokemonType[str(defender_types[1]).upper().replace("?", "THREE_QUESTION_MARKS")]
            if len(defender_types) > 1
            else None
        )
    except KeyError:
        return 1.0
    return float(move_enum.damage_multiplier(type_1, type_2, type_chart=_gen_data().type_chart))


def _normalized_effectiveness(multiplier: float) -> float:
    if multiplier <= 0.0:
        return 0.0
    return _clip_scalar((math.log2(multiplier) + 2.0) / 4.0, 0.0, 1.0)


def _variant_multiplier(action: int) -> float:
    if action < 7:
        return 1.0
    return (1.0, 1.05, 1.25, 1.15, 1.1)[min((action - 7) // 20, 4)]


def resolve_native_action_feature_variant(
    feature_variant: str | None,
    feature_dim: int | None = None,
) -> tuple[str | None, int]:
    if feature_variant is None:
        if not feature_dim:
            return None, 0
        if feature_dim == NATIVE_ACTION_FEATURE_VARIANTS[NATIVE_ACTION_BASE_FEATURE_VARIANT]:
            return NATIVE_ACTION_BASE_FEATURE_VARIANT, feature_dim
        if feature_dim == NATIVE_ACTION_FEATURE_VARIANTS[NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT]:
            return NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT, feature_dim
        if feature_dim == NATIVE_ACTION_FEATURE_VARIANTS[NATIVE_ACTION_FRIENDLY_FIRE_FEATURE_VARIANT]:
            return NATIVE_ACTION_FRIENDLY_FIRE_FEATURE_VARIANT, feature_dim
        return None, feature_dim
    if feature_variant not in NATIVE_ACTION_FEATURE_VARIANTS:
        raise ValueError(f"unknown native action feature variant: {feature_variant}")
    resolved_dim = NATIVE_ACTION_FEATURE_VARIANTS[feature_variant]
    if feature_dim is not None and feature_dim != resolved_dim:
        raise ValueError(
            f"native action feature variant {feature_variant!r} expects dim {resolved_dim}, "
            f"got {feature_dim}"
        )
    return feature_variant, resolved_dim


def decode_native_action_index(action: int) -> NativeActionDescriptor:
    """Decode one poke-env native action index for a single doubles slot."""
    if action < 0 or action >= POKE_ENV_NATIVE_ACTIONS_PER_SLOT:
        raise ValueError(
            f"native action index must be in [0, {POKE_ENV_NATIVE_ACTIONS_PER_SLOT}), got {action}"
        )
    if action == 0:
        return NativeActionDescriptor(kind="pass")
    if 1 <= action <= 6:
        return NativeActionDescriptor(kind="switch", switch_index=action - 1)
    raw = action - 7
    return NativeActionDescriptor(
        kind="move",
        move_index=(raw % 20) // 5,
        target_code=(raw % 5) - 2,
        mechanic=("base", "mega", "zmove", "dynamax", "tera")[raw // 20],
    )


def _target_uncertainty(target_code: int | None, move_target: str | None) -> float:
    if target_code is None:
        return 0.0
    if move_target in {"allAdjacent", "allAdjacentFoes", "all"}:
        return 0.5
    if target_code in (1, 2):
        return 0.0
    if target_code == 0:
        return 0.75
    return 1.0


def _estimate_uncertainty(
    descriptor: NativeActionDescriptor,
    attacker_known: bool,
    move: dict[str, Any],
    defender_known: bool,
) -> float:
    if descriptor.kind != "move":
        return 0.0
    uncertainty = 0.0
    if not attacker_known:
        uncertainty += 0.35
    if not move:
        uncertainty += 0.45
    if descriptor.target_code in (1, 2) and not defender_known:
        uncertainty += 0.35
    uncertainty += _target_uncertainty(descriptor.target_code, move.get("target")) * 0.35
    if descriptor.mechanic in {"mega", "zmove", "dynamax", "tera"}:
        uncertainty += 0.15
    if not move.get("basePower") and move.get("category") != "Status":
        uncertainty += 0.2
    return _clip_scalar(uncertainty, 0.0, 1.0)


def estimate_native_action_features(
    player_team: dict[str, Any] | Sequence[dict[str, Any]],
    opponent_team: dict[str, Any] | Sequence[dict[str, Any]],
    observation: Sequence[float] | np.ndarray,
    legal_actions_mask: Sequence[int] | np.ndarray,
    feature_variant: str | None = NATIVE_ACTION_BASE_FEATURE_VARIANT,
    predicted_opponent_action: Sequence[int] | None = None,
) -> np.ndarray:
    """Estimate deterministic normalized native-action features.

    Base variant output is ``(2, 126, 4)`` with features
    ``[expected_damage, ko_pressure, effectiveness, switch_value]``.
    The friendly-fire variant adds a fifth indicator for a damaging move
    explicitly aimed at an ally; its damaging features are zeroed for that
    action so it is not incorrectly credited with opponent damage.
    Uncertainty variants append ``[estimate_uncertainty, target_uncertainty]``;
    the statcalc variant uses generated set stats and the level-scaled damage formula.
    """
    feature_variant, feature_dim = resolve_native_action_feature_variant(feature_variant)
    mask = np.asarray(legal_actions_mask, dtype=np.int8)
    if mask.ndim != 1 or mask.size % 2 or mask.size // 2 > 126:
        raise ValueError("native action mask must contain two individual masks")
    features = np.zeros((2, MODEL_NATIVE_ACTIONS_PER_SLOT, feature_dim), dtype=np.float32)
    player_members = _team_members(player_team)
    # Trajectory records may carry the opponent's complete hidden set. This
    # estimator runs on the acting player's perspective, so only public
    # species identity is usable until revealed-set tracking supplies facts.
    opponent_members = [
        {key: pokemon.get(key) for key in ("species", "name") if pokemon.get(key)}
        for pokemon in _team_members(opponent_team)
    ]
    state = np.asarray(observation, dtype=np.float32)
    actual_slot_actions = min(mask.size // 2, POKE_ENV_NATIVE_ACTIONS_PER_SLOT, MODEL_NATIVE_ACTIONS_PER_SLOT)

    own_hp = np.clip(state[1:7], 0.0, 1.0) if state.size >= 7 else np.ones(6, dtype=np.float32)
    opp_hp = (
        np.clip(state[7:13], 0.0, 1.0)
        if state.size >= 13
        else np.ones(6, dtype=np.float32)
    )
    own_active_hp = (
        np.clip(state[25:27], 0.0, 1.0)
        if state.size >= 27
        else np.clip(own_hp[:2], 0.0, 1.0)
    )
    opp_alive = (
        state[19:25] > 0.0
        if state.size >= 25
        else opp_hp > 0.0
    )
    target_indices = [idx for idx, alive in enumerate(opp_alive[:6]) if alive]
    own_active_indices = [0, 1]
    if state.size >= 82:
        own_active_map = state[70:82].reshape(2, 6)
        for slot in range(2):
            if own_active_map[slot].max(initial=0.0) > 0.5:
                own_active_indices[slot] = int(own_active_map[slot].argmax())
        # The observation stores two six-slot one-hot rows at 82:94.
        if state.size >= 94:
            opponent_active_map = state[82:94].reshape(2, 6)
            active_targets = [
                int(row.argmax()) for row in opponent_active_map if row.max(initial=0.0) > 0.5
            ]
            if active_targets:
                target_indices = active_targets[:2]
    if len(target_indices) < 2:
        for idx, hp in enumerate(opp_hp[:6]):
            if hp > 0.0 and idx not in target_indices:
                target_indices.append(idx)
            if len(target_indices) == 2:
                break
    while len(target_indices) < 2:
        target_indices.append(len(target_indices))

    incoming_threats = (
        _incoming_threats(
            player_members, opponent_members, state, own_hp, own_active_hp,
            own_active_indices, target_indices[:2],
        )
        if feature_variant in {
            NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT,
            NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT,
        }
        else np.zeros((6, 2), dtype=np.float32)
    )
    counterplay_threats = None
    if feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT:
        if predicted_opponent_action is not None:
            counterplay_threats = _predicted_opponent_action_threats(
                player_members,
                opponent_members,
                state,
                own_hp,
                own_active_hp,
                own_active_indices,
                target_indices[:2],
                predicted_opponent_action,
            )
        else:
            counterplay_threats = np.repeat(incoming_threats[:, None, :], 2, axis=1)

    predicted_attacks: list[tuple[int, int, dict[str, Any], tuple[int, ...]]] = []
    if feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT and predicted_opponent_action is not None:
        for opponent_slot, opponent_action in enumerate(predicted_opponent_action[:2]):
            opponent_descriptor = decode_native_action_index(int(opponent_action))
            if opponent_descriptor.kind != "move" or opponent_descriptor.move_index is None:
                continue
            if opponent_slot >= len(target_indices):
                continue
            opponent_index = int(target_indices[opponent_slot])
            if opponent_index >= len(opponent_members):
                continue
            opponent_member = opponent_members[opponent_index]
            opponent_moves = list(opponent_member.get("moves", ()))
            if opponent_descriptor.move_index >= len(opponent_moves):
                continue
            opponent_move = _move_entry(opponent_moves[opponent_descriptor.move_index])
            if float(opponent_move.get("basePower") or 0.0) <= 0.0:
                continue
            opponent_target = opponent_move.get("target")
            if opponent_target in {"allAdjacent", "allAdjacentFoes", "all"}:
                threatened_slots = (0, 1)
            elif opponent_descriptor.target_code in (1, 2):
                threatened_slots = (int(opponent_descriptor.target_code) - 1,)
            else:
                continue
            predicted_attacks.append(
                (opponent_slot, opponent_index, opponent_move, threatened_slots)
            )

    for slot in range(2):
        active_index = own_active_indices[slot]
        attacker = player_members[active_index] if active_index < len(player_members) else {}
        attacker_species = attacker.get("species") or attacker.get("name")
        attacker_entry = _pokedex_entry(attacker_species)
        attacker_types = tuple(attacker_entry.get("types", ()))
        attacker_stats = attacker_entry.get("baseStats", {})
        attacker_known = bool(attacker_entry)
        active_hp = (
            float(own_active_hp[slot])
            if slot < own_active_hp.shape[0]
            else float(own_hp[min(slot, max(own_hp.shape[0] - 1, 0))])
        )
        for action in range(actual_slot_actions):
            # Features are consumed only after the corresponding legal-action
            # mask is applied by policy inference and PPO. Avoid estimating
            # damage/utility for masked choices while keeping the output
            # shape stable for stored rollouts and old checkpoints.
            if not mask[slot * (mask.size // 2) + action]:
                continue
            descriptor = decode_native_action_index(action)
            if descriptor.kind == "pass":
                continue
            if descriptor.kind == "switch":
                switch_index = descriptor.switch_index or 0
                bench_hp = float(own_hp[switch_index]) if switch_index < own_hp.shape[0] else 0.0
                features[slot, action, 3] = float(
                    _clip_scalar(0.5 * bench_hp + 0.5 * max(bench_hp - active_hp, 0.0), 0.0, 1.0)
                )
                if feature_variant == NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT and switch_index < 6:
                    features[slot, action, 4:6] = incoming_threats[switch_index]
                elif feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT and switch_index < 6:
                    features[slot, action, 4:6] = counterplay_threats[
                        switch_index, slot
                    ]
                    active_risk = float(counterplay_threats[active_index, slot, 1])
                    switch_risk = float(counterplay_threats[switch_index, slot, 1])
                    features[slot, action, 6] = float(
                        _clip_scalar(active_risk - switch_risk, 0.0, 1.0)
                    )
                continue
            move_names = list(attacker.get("moves", []))
            if descriptor.move_index is None or descriptor.move_index >= len(move_names):
                continue
            move = _move_entry(move_names[descriptor.move_index])
            if (
                feature_variant == NATIVE_ACTION_FRIENDLY_FIRE_FEATURE_VARIANT
                and descriptor.target_code is not None
                and descriptor.target_code < 0
                and move.get("target") in {
                    "normal", "any", "adjacentAlly", "adjacentAllyOrSelf"
                }
                and float(move.get("basePower") or 0.0) > 0.0
                and move.get("category") != "Status"
            ):
                features[slot, action, 4] = 1.0
                continue
            if feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT:
                move_id = _normalize_name(move_names[descriptor.move_index]).replace("-", "")
                if move_id in {
                    "protect", "detect", "spikyshield", "banefulbunker",
                    "kingsshield", "silktrap", "burningbulwark", "obstruct",
                    "maxguard", "endure",
                }:
                    features[slot, action, 6] = _clip_scalar(
                        counterplay_threats[active_index, slot, 1], 0.0, 1.0
                    )
            base_power = float(move.get("basePower") or 0.0)
            if base_power <= 0.0 or move.get("category") == "Status":
                if feature_variant in {NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT,
                                       NATIVE_ACTION_STATCALC_FEATURE_VARIANT,
                                       NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT}:
                    features[slot, action, 4] = _estimate_uncertainty(
                        descriptor,
                        attacker_known,
                        move,
                        defender_known=False,
                    )
                    features[slot, action, 5] = _target_uncertainty(
                        descriptor.target_code, move.get("target")
                    )
                continue
            target_slot = (
                target_indices[min(max((descriptor.target_code or 0) - 1, 0), 1)]
                if descriptor.target_code in (1, 2)
                else None
            )
            defender = (
                opponent_members[target_slot]
                if target_slot is not None and target_slot < len(opponent_members)
                else (opponent_members[target_indices[0]] if opponent_members else {})
            )
            defender_species = defender.get("species") or defender.get("name")
            defender_entry = _pokedex_entry(defender_species)
            defender_types = tuple(defender_entry.get("types", ()))
            defender_stats = defender_entry.get("baseStats", {})
            defender_known = bool(defender_entry)
            category = move.get("category")
            attack_name = "spa" if category == "Special" else "atk"
            defense_name = "spd" if category == "Special" else "def"
            statcalc = feature_variant in {
                NATIVE_ACTION_STATCALC_FEATURE_VARIANT,
                NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT,
                NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT,
            }
            attack_stat = (
                _pokemon_stat(attacker, attacker_stats, attack_name)
                if statcalc else float(attacker_stats.get(attack_name, 85))
            )
            defense_stat = (
                _pokemon_stat(defender, defender_stats, defense_name)
                if statcalc else float(defender_stats.get(defense_name, 85))
            )
            attacker_stage_slot = slot
            defender_stage_slot = 2 + target_indices.index(target_slot) if target_slot in target_indices[:2] else 2
            if statcalc and state.size >= 52:
                attack_stat *= _boost_multiplier(float(state[32 + attacker_stage_slot * 5 + (2 if category == "Special" else 0)]) * 6.0)
                defense_stat *= _boost_multiplier(float(state[32 + defender_stage_slot * 5 + (3 if category == "Special" else 1)]) * 6.0)
            stab = 1.5 if move.get("type") in attacker_types else 1.0
            effectiveness = _type_multiplier(move.get("type"), defender_types)
            spread_penalty = 0.75 if move.get("target") in {"allAdjacent", "allAdjacentFoes"} else 1.0
            accuracy = move.get("accuracy", 100)
            accuracy_multiplier = 1.0 if accuracy is True else _clip_scalar(float(accuracy) / 100.0, 0.0, 1.0)
            if statcalc:
                level = max(1, int(attacker.get("level", 50) or 50))
                damage = (((2.0 * level / 5.0 + 2.0) * base_power * attack_stat
                           / max(defense_stat, 1.0)) / 50.0) + 2.0
                damage *= stab * effectiveness * spread_penalty * accuracy_multiplier * 0.925
                weather = state[52:56] if state.size >= 56 else np.zeros(4, dtype=np.float32)
                move_type = str(move.get("type", "")).lower()
                if move_type == "water":
                    damage *= 1.5 if weather[0] else 0.5 if weather[1] else 1.0
                elif move_type == "fire":
                    damage *= 1.5 if weather[1] else 0.5 if weather[0] else 1.0
                damage *= _screen_damage_multiplier(
                    state, category, defender_is_opponent=True,
                    infiltrates=bool(move.get("flags", {}).get("infiltrates")),
                )
                target_max_hp = max(_pokemon_stat(defender, defender_stats, "hp"), 1.0)
                damage /= target_max_hp
            else:
                damage = (base_power / 150.0) * min(attack_stat / max(defense_stat, 1.0), 2.0)
                damage *= 0.18 * stab * effectiveness * spread_penalty * accuracy_multiplier
            damage *= _variant_multiplier(action)
            damage = _clip_scalar(damage, 0.0, 1.0)
            target_hp = (
                float(opp_hp[target_slot])
                if target_slot is not None and target_slot < opp_hp.shape[0]
                else float(np.mean(opp_hp[target_indices[:2]]))
            )
            ko_pressure = _clip_scalar(damage / max(target_hp, 0.05), 0.0, 1.0)
            features[slot, action, 0] = damage
            features[slot, action, 1] = ko_pressure
            features[slot, action, 2] = _normalized_effectiveness(effectiveness)
            if feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT and predicted_attacks:
                incoming_risk = float(counterplay_threats[active_index, slot, 1])
                mitigation = 0.0
                own_target = move.get("target")
                own_is_spread = own_target in {"allAdjacent", "allAdjacentFoes", "all"}
                own_priority = int(move.get("priority", 0) or 0)
                own_speed = _feature_action_speed(
                    attacker, state, active_position=slot,
                    member_index=active_index, opponent_side=False,
                )
                for opponent_slot, opponent_index, opponent_move, threatened_slots in predicted_attacks:
                    if slot not in threatened_slots:
                        continue
                    hits_attacker = own_is_spread or (
                        descriptor.target_code == opponent_slot + 1
                    )
                    if not hits_attacker:
                        continue
                    opponent_priority = int(opponent_move.get("priority", 0) or 0)
                    opponent_member = opponent_members[opponent_index]
                    opponent_speed = _feature_action_speed(
                        opponent_member, state, active_position=2 + opponent_slot,
                        member_index=opponent_index, opponent_side=True,
                    )
                    trick_room = state.size > 60 and state[60] > 0.5
                    speed_order = own_speed < opponent_speed if trick_room else own_speed > opponent_speed
                    acts_first = own_priority > opponent_priority or (
                        own_priority == opponent_priority and speed_order
                    )
                    if not acts_first:
                        continue
                    ability = _normalize_name(opponent_member.get("ability")).replace("-", "")
                    item = _normalize_name(opponent_member.get("item")).replace("-", "")
                    flinch = 0.0
                    if ability != "innerfocus" and item != "covertcloak":
                        flinch = _feature_flinch_probability(move, attacker, state)
                    cancellation = 1.0 if ko_pressure >= 1.0 else flinch
                    mitigation = max(mitigation, incoming_risk * cancellation)
                features[slot, action, 6] = max(
                    float(features[slot, action, 6]), _clip_scalar(mitigation, 0.0, 1.0)
                )
            if feature_variant in {NATIVE_ACTION_UNCERTAINTY_FEATURE_VARIANT,
                                   NATIVE_ACTION_STATCALC_FEATURE_VARIANT,
                                   NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT}:
                features[slot, action, 4] = _estimate_uncertainty(
                    descriptor,
                    attacker_known,
                    move,
                    defender_known,
                )
                features[slot, action, 5] = _target_uncertainty(
                    descriptor.target_code, move.get("target")
                )
            elif feature_variant in {
                NATIVE_ACTION_INCOMING_THREAT_FEATURE_VARIANT,
                NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT,
            }:
                if feature_variant == NATIVE_ACTION_COUNTERPLAY_FEATURE_VARIANT:
                    features[slot, action, 4:6] = counterplay_threats[active_index, slot]
                else:
                    features[slot, action, 4:6] = incoming_threats[active_index]
    return features


def valid_native_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only complete, non-timeout, perspective-safe player views."""
    valid = []
    for record in records:
        if (
            not is_perspective_safe_record(record)
            or record.get("timed_out")
            or record.get("teams_differ") is not True
        ):
            continue
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            continue
        if any(
            not step.get("log_probs")
            or not set(step["log_probs"]).issubset(step.get("values", {}))
            for step in trajectory
        ):
            continue
        valid.append(record)
    return valid


def extract_player_transitions(record: dict[str, Any], player: str) -> list[NativeTransition]:
    """Extract one player's factorized native transitions from a record."""
    first_step = record["trajectory"][0]
    agent_to_team = record.get("agent_to_team") or {}
    player_order = list(first_step.get("observations", {}))
    if player in agent_to_team:
        own_team_name = agent_to_team[player]
    elif player in record["teams"]:
        own_team_name = player
    else:
        own_team_name = f"player{player_order.index(player) + 1}"
    opponent_team_name = "player2" if own_team_name == "player1" else "player1"
    own_team = _team_members(record["teams"][own_team_name])
    opponent_team = [
        {key: pokemon[key] for key in ("species", "name") if pokemon.get(key)}
        for pokemon in _team_members(record["teams"][opponent_team_name])
    ]
    public_mechanics_static_facts = prepare_public_mechanics_static(
        own_team, str(record.get("format", "")),
    )
    target_blob = record.get("hidden_set_targets") or {}
    hidden_team = _team_members(target_blob.get("pokemon", []))
    hidden_move_targets: list[tuple[str, ...]] = []
    hidden_move_ids: list[tuple[str, ...]] = []
    hidden_item_targets: list[str] = []
    hidden_ability_targets: list[str] = []
    if hidden_team:
        for slot in range(6):
            member = hidden_team[slot] if slot < len(hidden_team) else {}
            moves = tuple(str(move) for move in list(member.get("moves", ()))[:4])
            hidden_move_targets.append(moves)
            hidden_move_ids.append(tuple(to_id_str(move) for move in moves))
            hidden_item_targets.append(str(member.get("item") or ""))
            hidden_ability_targets.append(str(member.get("ability") or ""))
    terminal_outcome = None
    if isinstance(record.get("rewards"), dict) and player in record["rewards"]:
        reward_value = float(record["rewards"][player])
        if math.isfinite(reward_value) and reward_value != 0:
            terminal_outcome = 1.0 if reward_value > 0 else 0.0
    result = []
    for index, step in enumerate(record["trajectory"]):
        observation = np.asarray(step["observations"][player], dtype=np.float32)
        mask = np.asarray(step["action_masks"][player], dtype=np.int8)
        # In on-policy gated rollouts, Showdown executes the action after the
        # deterministic gate, while PPO must score the actor's sampled
        # proposal whose log-probability was recorded by the actor.
        policy_proposal = step.get("policy_proposed_actions", {}).get(player)
        action = np.asarray(
            policy_proposal if policy_proposal is not None else step["actions"][player],
            dtype=np.int64,
        )
        if mask.ndim != 1 or mask.size % 2 or action.shape != (2,):
            raise ValueError("native transition has invalid mask or action shape")
        terminated = bool(step.get("terminated", {}).get(player, False))
        truncated = bool(step.get("truncated", {}).get(player, False))
        native_action_features = None
        raw_features = step.get("native_action_features", {}).get(player)
        if raw_features is not None:
            native_action_features = np.asarray(raw_features, dtype=np.float32)
            if native_action_features.ndim != 3 or native_action_features.shape[:2] != (2, 126):
                raise ValueError("native action features must have shape (2, 126, feature_dim)")
        counterplay_action_scores = None
        raw_scores = step.get("counterplay_action_scores", {}).get(player)
        if raw_scores is not None:
            counterplay_action_scores = np.asarray(raw_scores, dtype=np.float32)
            if (
                counterplay_action_scores.ndim != 2
                or counterplay_action_scores.shape[0] != 2
                or counterplay_action_scores.shape[1] > MODEL_NATIVE_ACTIONS_PER_SLOT
            ):
                raise ValueError("counterplay action scores must have shape (2, <=126)")
            if not np.isfinite(counterplay_action_scores).all():
                raise ValueError("counterplay action scores must be finite")
            if counterplay_action_scores.shape[1] < MODEL_NATIVE_ACTIONS_PER_SLOT:
                counterplay_action_scores = np.pad(
                    counterplay_action_scores,
                    ((0, 0), (0, MODEL_NATIVE_ACTIONS_PER_SLOT - counterplay_action_scores.shape[1])),
                )
        teacher_first = step.get("counterplay_teacher_first_action", {}).get(player)
        raw_teacher_actions = step.get("teacher_actions", {}).get(player)
        teacher_actions = None
        if raw_teacher_actions is not None:
            teacher_actions = np.asarray(raw_teacher_actions, dtype=np.int64)
            if teacher_actions.shape != (2,):
                raise ValueError("teacher actions must have shape (2,)")
            for slot, teacher_action in enumerate(teacher_actions):
                if teacher_action < 0 or teacher_action >= mask.size // 2:
                    raise ValueError("teacher action is outside the native action space")
                if not mask.reshape(2, -1)[slot, teacher_action]:
                    raise ValueError("teacher action is illegal under the recorded action mask")
        raw_joint_distribution = step.get(
            "teacher_joint_action_distributions", {}
        ).get(player)
        teacher_joint_actions = None
        teacher_joint_action_probabilities = None
        if raw_joint_distribution is not None:
            try:
                teacher_joint_actions = np.asarray(
                    raw_joint_distribution["actions"], dtype=np.int64
                )
                teacher_joint_action_probabilities = np.asarray(
                    raw_joint_distribution["probabilities"], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid joint teacher distribution") from error
            if (
                teacher_joint_actions.ndim != 2
                or teacher_joint_actions.shape[1] != 2
                or teacher_joint_actions.shape[0] == 0
                or teacher_joint_action_probabilities.shape
                != (teacher_joint_actions.shape[0],)
            ):
                raise ValueError("joint teacher actions and probabilities have invalid shapes")
            if (
                (teacher_joint_actions < 0).any()
                or (teacher_joint_actions >= MODEL_NATIVE_ACTIONS_PER_SLOT).any()
                or not np.isfinite(teacher_joint_action_probabilities).all()
                or (teacher_joint_action_probabilities < 0).any()
            ):
                raise ValueError("joint teacher distribution contains invalid values")
            probability_total = float(teacher_joint_action_probabilities.sum())
            if probability_total <= 0:
                raise ValueError("joint teacher distribution has zero probability mass")
            if len({tuple(row) for row in teacher_joint_actions.tolist()}) != len(
                teacher_joint_actions
            ):
                raise ValueError("joint teacher distribution contains duplicate actions")
            teacher_joint_action_probabilities = (
                teacher_joint_action_probabilities / probability_total
            ).astype(np.float32)
            slot_masks = mask.reshape(2, -1)
            # Candidate indices are reconstructed from semantic Showdown
            # orders. Under transformed/reordered Pokemon, a subset can fail
            # to line up with the live native mask. Restrict the soft target
            # to candidates that remain legal in the recorded observation.
            legal_pairs = (
                slot_masks[0, teacher_joint_actions[:, 0]]
                & slot_masks[1, teacher_joint_actions[:, 1]]
            ).astype(bool)
            if not legal_pairs.all():
                teacher_joint_actions = teacher_joint_actions[legal_pairs]
                teacher_joint_action_probabilities = (
                    teacher_joint_action_probabilities[legal_pairs]
                )
                legal_mass = float(teacher_joint_action_probabilities.sum())
                if legal_mass > 0:
                    teacher_joint_action_probabilities = (
                        teacher_joint_action_probabilities / legal_mass
                    )
                else:
                    teacher_joint_actions = None
                    teacher_joint_action_probabilities = None
            if teacher_joint_actions is None:
                conditional_rows = np.asarray([], dtype=bool)
            else:
                behavior_first_action = int(action[0])
                conditional_rows = teacher_joint_actions[:, 0] == behavior_first_action
            if teacher_joint_actions is not None and not conditional_rows.any():
                # Candidate native indices can shift when the semantic Showdown
                # order is translated back through transformed or reordered
                # active Pokemon. Keep the executed search action as the hard
                # teacher label, but do not apply a soft conditional target
                # that cannot be conditioned on that executed first action.
                teacher_joint_actions = None
                teacher_joint_action_probabilities = None
            elif teacher_joint_actions is not None:
                conditional_second = teacher_joint_actions[conditional_rows, 1]
                if not slot_masks[1, conditional_second].all():
                    raise ValueError(
                        "joint teacher distribution contains an illegal conditional second action"
                    )
        raw_opponent_species = step.get("opponent_species", {}).get(player)
        opponent_species = (
            tuple(str(species) for species in raw_opponent_species[:6] if species)
            if isinstance(raw_opponent_species, (list, tuple)) else None
        )
        visible_opponent_team = (
            [{"species": species} for species in opponent_species]
            if opponent_species is not None else opponent_team
        )
        public_rows = (
            step.get("public_mechanics", {}).get(player, {}).get("opponent", [])
        )
        move_masks: list[tuple[bool, ...]] = []
        item_masks: tuple[bool, ...] = ()
        ability_masks: tuple[bool, ...] = ()
        if hidden_team:
            for slot in range(6):
                revealed = {
                    to_id_str(str(move))
                    for move in (
                        public_rows[slot].get("revealed_moves", ())
                        if slot < len(public_rows) and isinstance(public_rows[slot], dict)
                        else ()
                    )
                }
                move_masks.append(
                    tuple(move_id not in revealed for move_id in hidden_move_ids[slot])
                )
            # These facts are not yet represented in the live input schema.
            # Learn items and abilities from closed-sheet preview only.
            item_masks = tuple(bool(item) and index == 0 for item in hidden_item_targets)
            ability_masks = tuple(
                bool(ability) and index == 0 for ability in hidden_ability_targets
            )
        behavior_log_prob = step.get(
            "search_action_behavior_log_probs", {}
        ).get(player)
        if behavior_log_prob is None:
            behavior_log_prob = step["log_probs"][player]
        else:
            behavior_log_prob = float(behavior_log_prob)
            if not math.isfinite(behavior_log_prob) or behavior_log_prob > 1e-6:
                raise ValueError("search behavior log-probability must be finite and nonpositive")
            if player not in step.get("teacher_joint_action_distributions", {}):
                raise ValueError(
                    "search behavior log-probability requires its joint teacher distribution"
                )
        # Older JSONL records did not persist the two Gymnasium flags.
        if "terminated" not in step and "truncated" not in step:
            terminated = index == len(record["trajectory"]) - 1
        result.append(
            NativeTransition(
                observation=observation,
                action_mask=mask,
                action=action,
                reward=float(step["rewards"].get(player, 0.0)),
                old_log_prob=float(behavior_log_prob),
                value=float(step["values"][player]),
                done=terminated or truncated,
                terminated=terminated,
                truncated=truncated,
                native_action_features=native_action_features,
                counterplay_action_scores=counterplay_action_scores,
                counterplay_teacher_first_action=(
                    int(teacher_first) if teacher_first is not None else None
                ),
                reward_components=step.get("reward_components", {}).get(player),
                public_context_features=encode_public_mechanics(
                    step.get("public_mechanics", {}).get(player),
                    observation,
                    own_team,
                    visible_opponent_team,
                    str(record.get("format", "")),
                    static_facts=public_mechanics_static_facts,
                ),
                opponent_species=opponent_species,
                step_index=index,
                teacher_actions=teacher_actions,
                teacher_joint_actions=teacher_joint_actions,
                teacher_joint_action_probabilities=teacher_joint_action_probabilities,
                hidden_move_targets=tuple(hidden_move_targets) if hidden_team else None,
                hidden_move_masks=tuple(move_masks) if hidden_team else None,
                hidden_item_targets=tuple(hidden_item_targets) if hidden_team else None,
                hidden_item_masks=item_masks if hidden_team else None,
                hidden_ability_targets=tuple(hidden_ability_targets) if hidden_team else None,
                hidden_ability_masks=ability_masks if hidden_team else None,
                win_target=terminal_outcome,
            )
        )
    return result


def factorized_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    value_weight: float = 0.5,
) -> torch.Tensor:
    """Apply PPO to joint doubles actions represented by summed slot log-probs."""
    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError("policy tensors must have matching shapes")
    return ppo_policy_loss(log_probs, old_log_probs, advantages) + value_weight * ppo_value_loss(
        values, returns
    )


def compute_native_gae(
    transitions: list[NativeTransition], reward_mode: str = "mixed",
    gae_lambda: float = 0.95, gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for one extracted player's trajectory."""
    if not transitions:
        raise ValueError("cannot compute GAE for an empty trajectory")
    if reward_mode not in {"mixed", "terminal", "hp_progress"}:
        raise ValueError("reward_mode must be mixed, terminal, or hp_progress")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be finite and between 0 and 1")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between 0 and 1")
    reward_values = []
    for item in transitions:
        components = item.reward_components or {}
        if reward_mode == "mixed":
            reward_values.append(item.reward)
        elif reward_mode == "terminal":
            reward_values.append(float(components.get("terminal_outcome", item.reward)))
        else:
            reward_values.append(float(components.get("hp_progress", item.reward)))
    rewards = torch.tensor([reward_values], dtype=torch.float32)
    values = torch.tensor([[item.value for item in transitions]], dtype=torch.float32)
    dones = torch.tensor([[float(item.done) for item in transitions]], dtype=torch.float32)
    return compute_gae(rewards, values, dones, gamma=gamma, lam=gae_lambda)
