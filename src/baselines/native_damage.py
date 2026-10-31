"""Deterministic, public-state damage heuristics for native doubles.

Counterplay estimates add common public item, ability, and terrain modifiers
to the baseline damage model. The max-damage benchmark keeps its separate
legacy score so its policy remains fixed across experiments.
"""
from __future__ import annotations

import heapq
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poke_env.battle import Move
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder
from src.training.native_ppo import NativeActionDescriptor, decode_native_action_index


_PROTECT_MOVE_IDS = {
    "protect", "detect", "spikyshield", "banefulbunker", "kingsshield",
    "silktrap", "burningbulwark", "obstruct", "maxguard", "endure",
}
_QUICK_GUARD_MOVE_IDS = {"quickguard"}
_REDIRECTION_MOVE_IDS = {"followme", "ragepowder"}
_OHKO_MOVE_IDS = {"fissure", "guillotine", "horndrill", "sheercold"}
_RANDOM_TEAM_PRIOR_PATH = Path(__file__).resolve().parents[1] / "features" / "randomteam_set_priors.json"
_RANDOM_TEAM_FULL_PRIOR_PATH = Path(__file__).resolve().parents[1] / "features" / "randomteam_full_set_priors.json"
try:
    with _RANDOM_TEAM_PRIOR_PATH.open(encoding="utf-8") as _stream:
        _RANDOM_TEAM_SET_PRIORS = json.load(_stream)
except FileNotFoundError:
    _RANDOM_TEAM_SET_PRIORS = {"species_sets": {}, "move_data": {}}
try:
    with _RANDOM_TEAM_FULL_PRIOR_PATH.open(encoding="utf-8") as _stream:
        _RANDOM_TEAM_FULL_SET_PRIORS = json.load(_stream)
except FileNotFoundError:
    _RANDOM_TEAM_FULL_SET_PRIORS = {"full_species_sets": {}}

_TYPE_BOOSTING_ITEMS = {
    "normal": {"silkscarf"}, "fire": {"charcoal"},
    "water": {"mysticwater"}, "electric": {"magnet"},
    "grass": {"miracleseed"}, "ice": {"nevermeltice"},
    "fighting": {"blackbelt"}, "poison": {"poisonbarb"},
    "ground": {"softsand"}, "flying": {"sharpbeak"},
    "psychic": {"twistedspoon"}, "bug": {"silverpowder"},
    "rock": {"hardstone"}, "ghost": {"spelltag"},
    "dragon": {"dragonfang"}, "dark": {"blackglasses"},
    "steel": {"metalcoat"}, "fairy": {"fairyfeather"},
}


def _id(value) -> str:
    return str(getattr(value, "name", value) or "").lower().replace("_", "").replace("-", "").replace(" ", "")


def _move_has_flag(move, flag: str) -> bool:
    flags = getattr(move, "flags", ()) or ()
    if hasattr(flags, "keys"):
        return bool(flags.get(flag, False))
    return flag in flags


def _public_damage_modifiers(attacker, defender, move, battle, category: str) -> float:
    """Estimate common M-C damage modifiers visible in the current battle."""
    ability = _id(getattr(attacker, "ability", ""))
    defender_ability = _id(getattr(defender, "ability", ""))
    item = _id(getattr(attacker, "item", ""))
    defender_item = _id(getattr(defender, "item", ""))
    move_type = _id(getattr(move, "type", ""))
    move_id = _id(getattr(move, "id", ""))
    power = float(getattr(move, "base_power", 0) or 0)
    modifier = 1.0
    effectiveness = float(defender.damage_multiplier(move))

    if (
        ability == "adaptability"
        and move_type in {_id(t) for t in (getattr(attacker, "types", ()) or ())}
    ):
        # The base estimator already applies the standard 1.5x STAB.
        modifier *= 4.0 / 3.0
    if effectiveness < 1.0 and ability == "tintedlens":
        modifier *= 2.0
    if effectiveness > 1.0 and ability == "neuroforce":
        modifier *= 1.25
    if effectiveness > 1.0 and defender_ability in {"filter", "solidrock", "prismarmor"}:
        modifier *= 0.75

    status = _id(getattr(attacker, "status", ""))
    if category == "PHYSICAL":
        if ability in {"hugepower", "purepower"}:
            modifier *= 2.0
        elif ability == "gorillatactics":
            modifier *= 1.5
        elif ability == "guts" and status not in {"", "none"}:
            modifier *= 1.5
        if ability == "hustle":
            modifier *= 1.5
        if status in {"brn", "burn"} and ability != "guts":
            modifier *= 0.5
    elif category == "SPECIAL":
        if ability == "solarpower" and "sunnyday" in _condition_names(getattr(battle, "weather", {})):
            modifier *= 1.5

    # Ability boosts tied to move properties commonly present in the M-C pool.
    if ability == "technician" and 0 < power <= 60:
        modifier *= 1.5
    if ability == "ironfist" and _move_has_flag(move, "punch"):
        modifier *= 1.2
    if ability == "sharpness" and _move_has_flag(move, "slicing"):
        modifier *= 1.5
    if ability == "toughclaws" and _move_has_flag(move, "contact"):
        modifier *= 1.3
    if ability == "reckless" and float(getattr(move, "recoil", 0) or 0) > 0:
        modifier *= 1.2
    if ability == "sheerforce" and (getattr(move, "secondary", ()) or ()):
        modifier *= 1.3
    if ability in {"blaze", "overgrow", "torrent", "swarm"}:
        hp = float(getattr(attacker, "current_hp", 0) or 0)
        max_hp = max(float(getattr(attacker, "max_hp", 1) or 1), 1.0)
        boosted_type = {"blaze": "fire", "overgrow": "grass", "torrent": "water", "swarm": "bug"}[ability]
        if hp / max_hp <= 1 / 3 and move_type == boosted_type:
            modifier *= 1.5

    if category == "PHYSICAL" and item == "choiceband":
        modifier *= 1.5
    elif category == "SPECIAL" and item == "choicespecs":
        modifier *= 1.5
    if item == "lifeorb":
        modifier *= 1.3
    elif item == "expertbelt" and effectiveness > 1.0:
        modifier *= 1.2
    elif category == "PHYSICAL" and item == "muscleband":
        modifier *= 1.1
    elif category == "SPECIAL" and item == "wiseglasses":
        modifier *= 1.1
    if item in _TYPE_BOOSTING_ITEMS.get(move_type, set()):
        modifier *= 1.2

    # Public defensive modifiers that substantially change common damage rolls.
    if defender_ability == "thickfat" and move_type in {"fire", "ice"}:
        modifier *= 0.5
    if defender_ability == "furcoat" and category == "PHYSICAL":
        modifier *= 0.5
    if defender_ability == "icescales" and category == "SPECIAL":
        modifier *= 0.5
    if defender_ability == "multiscale":
        hp = float(getattr(defender, "current_hp", 0) or 0)
        max_hp = max(float(getattr(defender, "max_hp", 1) or 1), 1.0)
        if hp >= max_hp:
            modifier *= 0.5
    if defender_ability == "fluffy" and category == "PHYSICAL" and _move_has_flag(move, "contact"):
        modifier *= 0.5
    if defender_ability == "fluffy" and move_type == "fire":
        modifier *= 2.0
    if category == "SPECIAL" and defender_item == "assaultvest":
        modifier *= 2.0 / 3.0

    fields = _condition_names(getattr(battle, "fields", {}))
    try:
        grounded = bool(battle.is_grounded(attacker))
    except (AttributeError, TypeError):
        grounded = True
    if grounded:
        if ("electricterrain" in fields and move_type == "electric") or (
            "grassyterrain" in fields and move_type == "grass"
        ) or ("psychicterrain" in fields and move_type == "psychic"):
            modifier *= 1.3
    if "mistyterrain" in fields and move_type == "dragon":
        try:
            target_grounded = bool(battle.is_grounded(defender))
        except (AttributeError, TypeError):
            target_grounded = True
        if target_grounded:
            modifier *= 0.5
    return modifier


def _target_name(move) -> str:
    return str(getattr(getattr(move, "target", None), "name", "")).lower().replace("_", "")


@lru_cache(maxsize=512)
def _public_prior_move(move_id: str):
    """Build a move object from the public format prior, never a hidden set."""
    move = Move(move_id, 9)
    metadata = _RANDOM_TEAM_SET_PRIORS.get("move_data", {}).get(move_id, {})
    if metadata:
        move._base_power_override = int(metadata.get("base_power", move.base_power) or 0)
    return move


def _max_damage_move_for_public_set(attacker, move_ids, battle, *, objective="damage"):
    """Predict this species' highest-damage move under one possible public set."""
    if objective not in {"damage", "hp_fraction"}:
        raise ValueError("objective must be 'damage' or 'hp_fraction'")
    defenders = [
        mon for mon in (getattr(battle, "active_pokemon", ()) or ())[:2]
        if mon is not None and not getattr(mon, "fainted", False)
    ]
    public_attacker = SimpleNamespace(
        level=getattr(attacker, "level", 50),
        base_stats=getattr(attacker, "base_stats", {}),
        stats=None,
        boosts=getattr(attacker, "boosts", {}),
        types=getattr(attacker, "types", ()),
        status=getattr(attacker, "status", None),
        current_hp=getattr(attacker, "current_hp", None),
        max_hp=getattr(attacker, "max_hp", None),
        ability="",
        item="",
        fainted=False,
    )
    best = None
    best_score = -1.0
    for move_id in move_ids:
        try:
            move = _public_prior_move(str(move_id))
        except (KeyError, ValueError):
            continue
        if move.category.name == "STATUS" or move.base_power <= 0:
            continue
        spread = _target_name(move) in {"alladjacent", "alladjacentfoes", "all"}
        if spread:
            damage = sum(
                _public_expected_damage(
                    public_attacker, defender, move, spread=len(defenders) > 1,
                    battle=battle, incoming=True,
                )
                for defender in defenders
            )
            score = damage
            if objective == "hp_fraction":
                score = sum(
                    _public_expected_damage(
                        public_attacker, defender, move, spread=len(defenders) > 1,
                        battle=battle, incoming=True,
                    )
                    / max(float(getattr(defender, "current_hp", 1) or 1), 1.0)
                    for defender in defenders
                )
        else:
            damages = [
                _public_expected_damage(
                    public_attacker, defender, move, battle=battle, incoming=True
                )
                for defender in defenders
            ]
            score = max(damages, default=0.0)
            if objective == "hp_fraction":
                score = max((
                    damage / max(float(getattr(defender, "current_hp", 1) or 1), 1.0)
                    for damage, defender in zip(damages, defenders)
                ), default=0.0)
        if score > best_score:
            best, best_score = move, score
    return (best, public_attacker) if best_score > 0 else (None, public_attacker)


def _random_set_attacker(attacker, set_data):
    """Build a public-stat attacker hypothesis from a sampled local format set."""
    level = int(set_data.get("level", getattr(attacker, "level", 50)) or 50)
    base_stats = getattr(attacker, "base_stats", {}) or {}
    evs = set_data.get("evs", {}) or {}
    ivs = set_data.get("ivs", {}) or {}
    stats = {}
    for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
        base = int(base_stats.get(stat, 85) or 85)
        iv = int(ivs.get(stat, 31) or 0)
        ev = int(evs.get(stat, 0) or 0)
        value = int(((2 * base + iv + ev // 4) * level) // 100)
        stats[stat] = value + level + 10 if stat == "hp" else value + 5
    known_max_hp = max(float(getattr(attacker, "max_hp", 1) or 1.0), 1.0)
    known_hp = max(float(getattr(attacker, "current_hp", 0) or 0.0), 0.0)
    hp_fraction = min(known_hp / known_max_hp, 1.0)
    return SimpleNamespace(
        level=level,
        base_stats=base_stats,
        stats=stats,
        boosts=getattr(attacker, "boosts", {}) or {},
        types=getattr(attacker, "types", ()),
        status=getattr(attacker, "status", None),
        current_hp=hp_fraction * stats["hp"],
        max_hp=stats["hp"],
        ability=set_data.get("ability", ""),
        item=set_data.get("item", ""),
        fainted=False,
    )


def _max_damage_move_for_sampled_set(attacker, set_data, battle, *, objective="damage"):
    """Predict Max Damage's move under a sampled ability/item/stat/move set."""
    if objective not in {"damage", "hp_fraction"}:
        raise ValueError("objective must be 'damage' or 'hp_fraction'")
    defenders = [
        mon for mon in (getattr(battle, "active_pokemon", ()) or ())[:2]
        if mon is not None and not getattr(mon, "fainted", False)
    ]
    hypothesis = _random_set_attacker(attacker, set_data)
    best = None
    best_score = -1.0
    for move_id in set_data.get("moves", ()):
        try:
            move = _public_prior_move(str(move_id))
        except (KeyError, ValueError):
            continue
        if move.category.name == "STATUS" or move.base_power <= 0:
            continue
        target = _target_name(move)
        spread = target in {"alladjacent", "alladjacentfoes", "all"}
        damages = [
            _public_expected_damage(
                hypothesis, defender, move,
                spread=spread and len(defenders) > 1,
                battle=battle, incoming=True,
            )
            for defender in defenders
        ]
        if spread:
            score = sum(damages)
            if objective == "hp_fraction":
                score = sum(
                    damage / max(float(getattr(defender, "current_hp", 1) or 1), 1.0)
                    for damage, defender in zip(damages, defenders)
                )
        else:
            score = max(damages, default=0.0)
            if objective == "hp_fraction":
                score = max((
                    damage / max(float(getattr(defender, "current_hp", 1) or 1), 1.0)
                    for damage, defender in zip(damages, defenders)
                ), default=0.0)
        if score > best_score:
            best, best_score = move, score
    return (best, hypothesis) if best_score > 0 else (None, hypothesis)


def _damage_fraction(
    attacker,
    defender,
    move,
    *,
    spread: bool = False,
    battle=None,
    target_is_opponent: bool = True,
    cache: dict[tuple[int, int, int, bool, int, bool], tuple[float, float]] | None = None,
) -> tuple[float, float]:
    key = None
    if cache is not None:
        key = (
            id(attacker), id(defender), id(move), bool(spread), id(battle),
            bool(target_is_opponent),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
    raw = expected_damage(
        attacker, defender, move, spread=spread, battle=battle,
        target_is_opponent=target_is_opponent,
    )
    max_hp = max(float(getattr(defender, "max_hp", 1) or 1), 1.0)
    current_hp = max(float(getattr(defender, "current_hp", max_hp) or 0), 1.0)
    result = (raw / max_hp, raw / current_hp)
    if cache is not None and key is not None:
        cache[key] = result
    return result


def _flinch_probability(move, attacker, defender) -> float:
    """Return the chance this move prevents its target's still-pending action."""
    ability = str(getattr(defender, "ability", "")).lower().replace("-", "")
    item = str(getattr(defender, "item", "")).lower().replace("-", "")
    if ability == "innerfocus":
        return 0.0
    move_id = str(getattr(move, "id", "")).lower().replace("-", "")
    if move_id == "fakeout":
        return float(bool(getattr(attacker, "first_turn", False)))
    if item == "covertcloak":
        return 0.0
    secondaries = getattr(move, "secondary", ()) or ()
    chance = max(
        (
            float(effect.get("chance", 0.0) or 0.0) / 100.0
            for effect in secondaries
            if str(effect.get("volatileStatus", "")).lower() == "flinch"
        ),
        default=0.0,
    )
    accuracy = getattr(move, "accuracy", 1.0)
    accuracy = 1.0 if accuracy is True else float(accuracy or 0.0)
    return float(np.clip(chance * accuracy, 0.0, 1.0))


def choose_counterplay_action(
    action_mask: np.ndarray,
    battle,
    opponent_battle,
    predicted_opponent_action: np.ndarray | list[int] | None,
    *,
    incoming_weight: float = 1.0,
    support_actions: bool = False,
    ko_bonus: float = 0.7,
    speed_control_scale: float = 1.0,
    respect_protect_streaks: bool = True,
    cap_overkill: bool = True,
    return_action_scores: bool = False,
    condition_first_action: int | None = None,
    order_validator=None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Choose the legal joint action with the best attack-versus-survival value.

    Unlike max-damage, this one-ply heuristic scores the opponent's already
    predicted attacks. It rewards damage and KOs, penalizes incoming damage
    relative to current HP, and credits Protect/switch actions for damage they
    actually avoid. It is intentionally deterministic so it can also serve as
    a compact behavior-cloning teacher.
    """
    masks = np.asarray(action_mask, dtype=np.int8).reshape(2, -1)
    candidates: list[
        list[tuple[int, object, str, object, NativeActionDescriptor]]
    ] = []
    for slot in range(2):
        slot_candidates = []
        for action in np.flatnonzero(masks[slot]):
            action = int(action)
            descriptor = decode_native_action_index(action)
            try:
                # The mask was generated from current valid orders. `fake=True`
                # preserves the action-to-order conversion but skips repeating
                # poke-env's O(number of legal orders) string scan for every
                # candidate and every joint pair.
                order = DoublesEnv._action_to_order_individual(
                    np.int64(action), battle, True, slot
                )
            except (IndexError, ValueError):
                continue
            move = order.order if descriptor.kind == "move" else None
            move_id = str(getattr(move, "id", "")).lower().replace("-", "")
            slot_candidates.append((action, move, move_id, order, descriptor))
        candidates.append(slot_candidates)

    predicted = [int(x) for x in predicted_opponent_action[:2]] if predicted_opponent_action is not None else []
    opponent_team_members = list(opponent_battle.team.values())
    opponent_after = list(opponent_battle.active_pokemon)
    opponent_protection_probability = [0.0, 0.0]
    predicted_orders: list[object | None] = [None, None]
    for slot, action in enumerate(predicted):
        if decode_native_action_index(action).kind != "move":
            continue
        try:
            predicted_orders[slot] = DoublesEnv._action_to_order_individual(
                np.int64(action), opponent_battle, True, slot
            )
        except (IndexError, ValueError):
            continue
    for slot, action in enumerate(predicted):
        descriptor = decode_native_action_index(action)
        if descriptor.kind == "switch" and descriptor.switch_index is not None:
            if descriptor.switch_index < len(opponent_team_members):
                opponent_after[slot] = opponent_team_members[descriptor.switch_index]
        elif descriptor.kind == "move":
            order = predicted_orders[slot]
            if order is None:
                continue
            move_id = str(getattr(order.order, "id", "")).lower().replace("-", "")
            if move_id in _PROTECT_MOVE_IDS and opponent_after[slot] is not None:
                protect_counter = int(
                    getattr(opponent_after[slot], "protect_counter", 0) or 0
                )
                opponent_protection_probability[slot] = (
                    1.0 / (3.0 ** protect_counter) if respect_protect_streaks else 1.0
                )
    opponent_attacks: list[tuple[int, object, object, bool, tuple[int, ...]]] = []
    for slot, action in enumerate(predicted):
        descriptor = decode_native_action_index(action)
        if descriptor.kind != "move":
            continue
        order = predicted_orders[slot]
        if order is None:
            continue
        move = order.order
        if (
            float(getattr(move, "base_power", 0) or 0) <= 0
            and _id(getattr(move, "id", "")) not in _OHKO_MOVE_IDS
        ):
            continue
        target = _target_name(move)
        if target in {"alladjacent", "alladjacentfoes", "all"}:
            target_slots = (0, 1)
            spread = True
        elif descriptor.target_code in (1, 2):
            target_slots = (int(descriptor.target_code) - 1,)
            spread = False
        else:
            continue
        attacker = opponent_battle.active_pokemon[slot]
        if attacker is not None:
            opponent_attacks.append((slot, attacker, move, spread, target_slots))

    best_score = -float("inf")
    best_pair = None
    pair_scores: dict[int, dict[int, float]] = {}
    damage_cache: dict[tuple[int, int, int, bool, int, bool], tuple[float, float]] = {}
    team_members = list(battle.team.values())
    fields = _condition_names(getattr(battle, "fields", {}))
    own_conditions = _condition_names(getattr(battle, "side_conditions", {}))
    trick_room = "trickroom" in fields
    opponent_speed_pokemon = [
        p for p in opponent_after
        if p is not None and not p.fainted
    ]
    action_speed_cache: dict[tuple[int, int], float] = {}
    fast_pair_validation = order_validator is not None and not return_action_scores

    def cached_action_speed(pokemon, speed_battle) -> float:
        key = (id(pokemon), id(speed_battle))
        if key not in action_speed_cache:
            action_speed_cache[key] = _action_speed(pokemon, speed_battle)
        return action_speed_cache[key]

    for first in candidates[0]:
        for second in candidates[1]:
            pair = (first[0], second[0])
            if fast_pair_validation:
                # The action masks and individual order conversion above
                # already prove each component legal. During scoring, check
                # only the incompatibilities between slots. Strictly
                # converting every joint pair repeats the same Showdown work
                # O(candidates[0] * candidates[1]) times; validate the
                # selected pair once below, with a ranked fallback if needed.
                if not DoubleBattleOrder.join_orders([first[3]], [second[3]]):
                    continue
            elif order_validator is not None:
                try:
                    order_validator(np.asarray(pair, dtype=np.int64), battle, strict=True)
                except ValueError:
                    continue
            elif not DoubleBattleOrder.join_orders([first[3]], [second[3]]):
                # Each candidate is individually legal by construction from
                # the supplied action mask. The join check enforces exactly
                # the remaining doubles constraints (duplicate switch,
                # duplicate mechanic, and pass/pass) without revalidating the
                # individual orders against the full legal-order lists.
                continue

            selected = (first, second)
            active_after = []
            protection_probability = []
            wide_guard = False
            quick_guard = False
            for slot, (action, move, move_id, _, descriptor) in enumerate(selected):
                if descriptor.kind == "switch" and descriptor.switch_index is not None:
                    defender = (
                        team_members[descriptor.switch_index]
                        if descriptor.switch_index < len(team_members)
                        else battle.active_pokemon[slot]
                    )
                else:
                    defender = battle.active_pokemon[slot]
                active_after.append(defender)
                if move_id in _PROTECT_MOVE_IDS and defender is not None:
                    protect_counter = int(getattr(defender, "protect_counter", 0) or 0)
                    probability = (
                        1.0 / (3.0 ** protect_counter)
                        if respect_protect_streaks else 1.0
                    )
                else:
                    probability = 0.0
                protection_probability.append(probability)
                wide_guard = wide_guard or move_id == "wideguard"
                quick_guard = quick_guard or move_id in _QUICK_GUARD_MOVE_IDS

            outgoing = 0.0
            slot_damage = [0.0, 0.0]
            damage_by_opponent = [0.0, 0.0]
            for slot, (action, move, _, _, descriptor) in enumerate(selected):
                if descriptor.kind != "move" or move is None:
                    continue
                target = _target_name(move)
                attacker = battle.active_pokemon[slot]
                if attacker is None:
                    continue
                if target in {"alladjacent", "alladjacentfoes", "all"}:
                    foes = [
                        (foe_slot, foe)
                        for foe_slot, foe in enumerate(battle.opponent_active_pokemon)
                        if foe is not None and not foe.fainted
                    ]
                    spread = len(foes) + int(
                        target in {"alladjacent", "all"}
                        and battle.active_pokemon[1 - slot] is not None
                        and not battle.active_pokemon[1 - slot].fainted
                    ) > 1
                    for foe_slot, _ in foes:
                        foe = opponent_after[foe_slot]
                        if foe is None or foe.fainted:
                            continue
                        hp_fraction, _ = _damage_fraction(
                            attacker, foe, move, spread=spread, battle=battle,
                            cache=damage_cache,
                        )
                        hp_fraction *= 1.0 - opponent_protection_probability[foe_slot]
                        slot_damage[slot] += hp_fraction
                        if cap_overkill:
                            damage_by_opponent[foe_slot] += hp_fraction
                        else:
                            outgoing += hp_fraction
                            if hp_fraction >= float(getattr(foe, "current_hp", 1) or 0) / max(float(getattr(foe, "max_hp", 1) or 1), 1.0):
                                outgoing += ko_bonus
                    if target in {"alladjacent", "all"}:
                        ally = battle.active_pokemon[1 - slot]
                        if ally is not None and not ally.fainted:
                            ally_fraction, _ = _damage_fraction(
                            attacker, ally, move, spread=spread, battle=battle,
                            target_is_opponent=False,
                            cache=damage_cache,
                            )
                            outgoing -= 0.8 * ally_fraction
                elif descriptor.target_code in (1, 2):
                    target_slot = int(descriptor.target_code) - 1
                    defender = opponent_after[target_slot]
                    if defender is not None and not defender.fainted:
                        hp_fraction, current_fraction = _damage_fraction(
                            attacker, defender, move, battle=battle,
                            cache=damage_cache,
                        )
                        hp_fraction *= 1.0 - opponent_protection_probability[target_slot]
                        current_fraction *= 1.0 - opponent_protection_probability[target_slot]
                        slot_damage[slot] += hp_fraction
                        if cap_overkill:
                            damage_by_opponent[target_slot] += hp_fraction
                        else:
                            outgoing += hp_fraction + (ko_bonus if current_fraction >= 1.0 else 0.0)
                elif descriptor.target_code in (-1, -2):
                    ally_slot = -int(descriptor.target_code) - 1
                    ally = battle.active_pokemon[ally_slot]
                    if ally is not None and not ally.fainted:
                        ally_fraction, _ = _damage_fraction(
                        attacker, ally, move, battle=battle,
                        target_is_opponent=False,
                        cache=damage_cache,
                        )
                        outgoing -= 0.8 * ally_fraction

            # Score joint damage once per opposing active: two allies should
            # not each receive a full KO bonus for the same target, and excess
            # damage beyond that target's remaining HP has no value this turn.
            if cap_overkill:
                for foe_slot, expected_damage_fraction in enumerate(damage_by_opponent):
                    foe = opponent_after[foe_slot]
                    if foe is None or getattr(foe, "fainted", False):
                        continue
                    max_hp = max(float(getattr(foe, "max_hp", 1) or 1), 1.0)
                    remaining_fraction = max(
                        float(getattr(foe, "current_hp", 0) or 0) / max_hp, 0.0
                    )
                    outgoing += min(expected_damage_fraction, remaining_fraction)
                    if expected_damage_fraction >= remaining_fraction > 0.0:
                        outgoing += ko_bonus

            # Helping Hand is a high-value doubles action when its partner
            # attacks: the 1.5x damage boost applies before the partner's
            # selected move resolves. Keep the bonus tied to actual damage
            # from the paired action instead of assigning a generic status
            # move value.
            support_bonus = 0.0
            for slot, (_, _, move_id, _, _) in enumerate(selected):
                if move_id == "helpinghand":
                    support_bonus += 0.5 * slot_damage[1 - slot]

            # Speed control has value beyond this turn. Reward Tailwind only
            # when it can change the order against a currently faster foe;
            # reward Trick Room when our active pair is slower overall.
            speed_control_bonus = 0.0
            for _, _, move_id, _, _ in selected:
                if move_id == "tailwind" and "tailwind" not in own_conditions:
                    crossings = 0
                    for ally in active_after:
                        if ally is None or getattr(ally, "fainted", False):
                            continue
                        own_speed = cached_action_speed(ally, battle)
                        if any(
                            own_speed < cached_action_speed(foe, opponent_battle)
                            <= own_speed * 2.0
                            for foe in opponent_speed_pokemon
                        ):
                            crossings += 1
                    speed_control_bonus += 0.18 * speed_control_scale * crossings
                elif move_id == "trickroom" and "trickroom" not in fields:
                    slower_matchups = 0
                    faster_matchups = 0
                    for ally in active_after:
                        if ally is None or getattr(ally, "fainted", False):
                            continue
                        own_speed = cached_action_speed(ally, battle)
                        for foe in opponent_speed_pokemon:
                            foe_speed = cached_action_speed(foe, opponent_battle)
                            slower_matchups += own_speed < foe_speed
                            faster_matchups += own_speed > foe_speed
                    speed_control_bonus += 0.12 * speed_control_scale * max(
                        slower_matchups - faster_matchups, 0
                    )
            if not support_actions:
                support_bonus = 0.0
                speed_control_bonus = 0.0

            incoming_raw = [0.0, 0.0]
            for opponent_slot, attacker, move, spread, target_slots in opponent_attacks:
                # Do not charge an attack that our selected moves are expected
                # to prevent by KOing its user first. The old score treated
                # both sides' attacks as simultaneous, which made it overly
                # defensive in obvious priority/speed races.
                attacker_hp = max(
                    float(getattr(attacker, "current_hp", 0) or 0), 1.0
                )
                damage_before_attack = 0.0
                flinch_probability = 0.0
                for own_slot, (own_action, own_move, _, _, own_descriptor) in enumerate(selected):
                    if own_move is None or own_descriptor.kind != "move":
                        continue
                    own_attacker = battle.active_pokemon[own_slot]
                    if own_attacker is None:
                        continue
                    own_target = _target_name(own_move)
                    if own_target in {"alladjacent", "alladjacentfoes", "all"}:
                        hits_attacker = True
                        own_spread = (
                            len([p for p in battle.opponent_active_pokemon if p is not None and not p.fainted])
                            + int(
                                own_target in {"alladjacent", "all"}
                                and battle.active_pokemon[1 - own_slot] is not None
                                and not battle.active_pokemon[1 - own_slot].fainted
                            )
                            > 1
                        )
                    else:
                        hits_attacker = (
                            own_descriptor.target_code == opponent_slot + 1
                        )
                        own_spread = False
                    if not hits_attacker:
                        continue
                    own_priority = int(getattr(own_move, "priority", 0) or 0)
                    incoming_priority = int(getattr(move, "priority", 0) or 0)
                    own_speed = cached_action_speed(own_attacker, battle)
                    incoming_speed = cached_action_speed(attacker, opponent_battle)
                    speed_order = own_speed < incoming_speed if trick_room else own_speed > incoming_speed
                    acts_first = own_priority > incoming_priority or (
                        own_priority == incoming_priority and speed_order
                    )
                    if acts_first:
                        _, damage_fraction_current_hp = _damage_fraction(
                            own_attacker, attacker, own_move, spread=own_spread,
                            battle=battle, cache=damage_cache,
                        )
                        if damage_fraction_current_hp > 0.0:
                            flinch_probability = max(
                                flinch_probability,
                                _flinch_probability(own_move, own_attacker, attacker),
                            )
                        damage_before_attack += damage_fraction_current_hp
                if damage_before_attack >= 1.0:
                    continue
                if flinch_probability >= 1.0:
                    continue

                target = _target_name(move)
                if spread and wide_guard:
                    continue
                if quick_guard and int(getattr(move, "priority", 0) or 0) > 0:
                    continue
                for target_slot in target_slots:
                    if protection_probability[target_slot] >= 1.0:
                        continue
                    defender = active_after[target_slot]
                    if defender is None or getattr(defender, "fainted", False):
                        continue
                    raw, _ = _damage_fraction(
                        attacker, defender, move, spread=spread,
                        battle=opponent_battle, cache=damage_cache,
                    )
                    max_hp = max(float(getattr(defender, "max_hp", 1) or 1), 1.0)
                    incoming_raw[target_slot] += (
                        raw * max_hp * (1.0 - flinch_probability)
                        * (1.0 - protection_probability[target_slot])
                    )

            incoming_cost = 0.0
            for slot, damage in enumerate(incoming_raw):
                defender = active_after[slot]
                if defender is None:
                    continue
                current_hp = max(float(getattr(defender, "current_hp", 1) or 0), 1.0)
                damage_ratio = damage / current_hp
                incoming_cost += min(damage_ratio, 1.5)
                if damage_ratio >= 1.0:
                    incoming_cost += 0.35

            # Perish Song is a terminal threat that ordinary damage scoring
            # cannot see. A legal switch removes the endangered Pokemon from
            # the field while its countdown continues; the Pokemon can return
            # later after the volatile expires. Keep this bonus narrow (only
            # at counter 1 or 0) so the heuristic does not switch too early
            # or overfit ordinary play to rare Perish teams. Trapping is
            # naturally respected because Showdown omits illegal switches
            # from the action mask.
            perish_escape_bonus = 0.0
            for slot, (action, _, _, _, descriptor) in enumerate(selected):
                active = battle.active_pokemon[slot]
                if active is None:
                    continue
                effects = getattr(active, "effects", {}) or {}
                effect_names = {
                    str(getattr(effect, "name", effect)).upper()
                    .replace("_", "").replace("-", "")
                    for effect in effects
                }
                perish_counter = next(
                    (counter for counter in range(4)
                     if f"PERISH{counter}" in effect_names),
                    None,
                )
                if (perish_counter is not None and perish_counter <= 1
                        and descriptor.kind == "switch"):
                    perish_escape_bonus += 2.0

            score = (
                outgoing + support_bonus + speed_control_bonus + perish_escape_bonus
                - incoming_weight * incoming_cost
            )
            pair_scores.setdefault(pair[0], {})[pair[1]] = score
            if score > best_score:
                best_score, best_pair = score, pair
    if order_validator is not None and not return_action_scores and best_pair is not None:
        ranked_pairs = sorted(
            (
                (score, (first_action, second_action))
                for first_action, second_scores in pair_scores.items()
                for second_action, score in second_scores.items()
            ),
            key=lambda scored_pair: scored_pair[0],
            reverse=True,
        )
        best_pair = None
        for score, pair in ranked_pairs:
            try:
                order_validator(np.asarray(pair, dtype=np.int64), battle, strict=True)
            except ValueError:
                continue
            best_score, best_pair = score, pair
            break
    if best_pair is None:
        raise ValueError("No compatible legal counterplay action pair")
    chosen = np.asarray(best_pair, dtype=np.int64)
    if not return_action_scores:
        return chosen
    action_scores = np.full((2, masks.shape[1]), -1e6, dtype=np.float32)
    for first_action, second_scores in pair_scores.items():
        action_scores[0, first_action] = max(second_scores.values())
    score_condition = int(best_pair[0])
    if condition_first_action in pair_scores:
        score_condition = int(condition_first_action)
    conditioned_scores = pair_scores[score_condition]
    for second_action, score in conditioned_scores.items():
        action_scores[1, second_action] = score
    conditioned_second = max(conditioned_scores, key=conditioned_scores.get)
    joint_scores = np.full(
        (masks.shape[1], masks.shape[1]), -1e6, dtype=np.float32
    )
    for first_action, second_scores in pair_scores.items():
        for second_action, score in second_scores.items():
            joint_scores[first_action, second_action] = score
    return chosen, action_scores, int(conditioned_second), joint_scores


def _stat(pokemon, name: str) -> float:
    value = (getattr(pokemon, "stats", None) or {}).get(name)
    if not value:
        base = (getattr(pokemon, "base_stats", None) or {}).get(name, 85)
        value = (2 * base + 31) * (getattr(pokemon, "level", 50) or 50) / 100 + 5
    stage = (getattr(pokemon, "boosts", None) or {}).get(name, 0)
    return float(value) * (2 + max(stage, 0)) / (2 - min(stage, 0))


def _condition_names(conditions) -> set[str]:
    if hasattr(conditions, "keys"):
        conditions = conditions.keys()
    return {
        str(getattr(condition, "name", condition)).lower().replace("_", "").replace("-", "")
        for condition in conditions or ()
    }


def _effect_names(pokemon) -> set[str]:
    effects = getattr(pokemon, "effects", {}) or {}
    if hasattr(effects, "keys"):
        effects = effects.keys()
    return {_id(effect) for effect in effects or ()}


def _ohko_target_is_immune(attacker, defender, move, move_id: str, battle=None) -> bool:
    """Conservatively account for public immunity to OHKO moves.

    This is used only by the damage heuristic/teacher, not by Showdown's
    resolver. Showdown remains authoritative for less common interactions.
    """
    attacker_ability = _id(getattr(attacker, "ability", ""))
    defender_ability = _id(getattr(defender, "ability", ""))
    defender_item = _id(getattr(defender, "item", ""))
    effects = _effect_names(defender)
    fields = _condition_names(getattr(battle, "fields", {})) if battle is not None else set()
    active = ()
    if battle is not None:
        active = tuple(getattr(battle, "active_pokemon", ()) or ()) + tuple(
            getattr(battle, "opponent_active_pokemon", ()) or ()
        )
    neutralizing_gas = "neutralizinggas" in fields or any(
        mon is not None and not getattr(mon, "fainted", False)
        and _id(getattr(mon, "ability", "")) == "neutralizinggas"
        for mon in active
    )

    # Sturdy blocks OHKO moves; No Guard affects accuracy only and does not
    # bypass this immunity. Mold Breaker family suppression is handled for
    # the common public cases unless Ability Shield protects the target.
    suppresses_defender_ability = attacker_ability in {
        "moldbreaker", "teravolt", "turboblaze"
    } and defender_item != "abilityshield"
    ability_suppressed = (
        neutralizing_gas
        or suppresses_defender_ability
    )
    if defender_ability == "sturdy" and not ability_suppressed:
        return True

    defender_types = {
        _id(value) for value in (getattr(defender, "types", ()) or ())
    }
    if move_id == "sheercold" and "ice" in defender_types:
        return True

    gravity = "gravity" in fields
    grounded_by_state = (
        gravity or "ingrain" in effects or "smackdown" in effects
        or defender_item == "ironball"
    )
    if move_id == "fissure":
        if not grounded_by_state and "magnetrise" in effects:
            return True
        if not grounded_by_state and "telekinesis" in effects:
            return True
        if not grounded_by_state and defender_ability == "levitate" and not ability_suppressed:
            return True
        if not grounded_by_state and defender_item == "airballoon":
            return True
        if (not grounded_by_state and "flying" in defender_types
                and defender_item != "ringtarget"):
            return True

    try:
        effectiveness = float(defender.damage_multiplier(move))
    except (AttributeError, TypeError, ValueError):
        effectiveness = 1.0
    # Grounding field effects let Fissure hit Flying types despite their
    # chart immunity; Ring Target removes type immunities but not Levitate or
    # Air Balloon (those were handled above).
    ground_immunity_overridden = move_id == "fissure" and grounded_by_state
    if (effectiveness <= 0.0 and defender_item != "ringtarget"
            and not ground_immunity_overridden):
        return True
    if defender_ability == "wonderguard" and effectiveness <= 1.0 and not ability_suppressed:
        return True
    return False


def _action_speed(pokemon, battle) -> float:
    """Estimate effective move-order speed under common M-C conditions."""
    speed = _stat(pokemon, "spe")
    ability = str(getattr(pokemon, "ability", "")).lower().replace("-", "")
    item = str(getattr(pokemon, "item", "")).lower().replace("-", "")
    status = str(getattr(getattr(pokemon, "status", None), "name", "")).upper()
    weather = _condition_names(getattr(battle, "weather", {}))
    fields = _condition_names(getattr(battle, "fields", {}))
    side_conditions = _condition_names(getattr(battle, "side_conditions", {}))

    if "tailwind" in side_conditions:
        speed *= 2.0
    if ability == "swiftswim" and weather.intersection({"raindance", "primordialsea"}):
        speed *= 2.0
    elif ability == "chlorophyll" and weather.intersection({"sunnyday", "desolateland"}):
        speed *= 2.0
    elif ability == "sandrush" and "sandstorm" in weather:
        speed *= 2.0
    elif ability == "slushrush" and weather.intersection({"hail", "snowscape"}):
        speed *= 2.0
    elif ability == "surgesurfer" and "electricterrain" in fields:
        speed *= 2.0

    if ability == "quickfeet" and status not in {"", "NONE"}:
        speed *= 1.5
    elif status in {"PAR", "PARALYSIS"}:
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


def expected_damage(
    attacker,
    defender,
    move,
    *,
    spread: bool = False,
    battle=None,
    target_is_opponent: bool = True,
) -> float:
    """Estimate damage including STAB, typing, weather, screens and stat stages.

    Passing battle state enables weather and screen adjustments. The max-damage
    benchmark deliberately calls this without battle state, preserving its
    fixed opponent policy while counterplay labels use the richer estimate.
    """
    if attacker is None or defender is None or getattr(defender, "fainted", False):
        return 0.0
    move_id = str(getattr(move, "id", "")).lower().replace("-", "")
    if move_id in _OHKO_MOVE_IDS:
        if float(getattr(attacker, "level", 50) or 50) < float(
            getattr(defender, "level", 50) or 50
        ):
            return 0.0
        if _ohko_target_is_immune(attacker, defender, move, move_id, battle):
            return 0.0
        attacker_ability = _id(getattr(attacker, "ability", ""))
        defender_ability = _id(getattr(defender, "ability", ""))
        fields = _condition_names(getattr(battle, "fields", {})) if battle is not None else set()
        neutralizing_gas = "neutralizinggas" in fields or any(
            mon is not None and not getattr(mon, "fainted", False)
            and _id(getattr(mon, "ability", "")) == "neutralizinggas"
            for mon in (
                tuple(getattr(battle, "active_pokemon", ()) or ())
                + tuple(getattr(battle, "opponent_active_pokemon", ()) or ())
            )
        ) if battle is not None else False
        has_no_guard = (
            not neutralizing_gas and "noguard" in {attacker_ability, defender_ability}
        )
        accuracy = getattr(move, "accuracy", 1.0)
        accuracy = 1.0 if accuracy is True else float(accuracy or 0.0)
        if move_id == "sheercold":
            attacker_types = {
                _id(value) for value in (getattr(attacker, "types", ()) or ())
            }
            if "ice" not in attacker_types:
                accuracy = 0.2
        # Showdown adds one percentage point for each level the user is above
        # the target. No Guard overrides this OHKO accuracy via Accuracy; it
        # does not override type, ability, or item immunity.
        level_delta = max(
            int(getattr(attacker, "level", 50) or 50)
            - int(getattr(defender, "level", 50) or 50),
            0,
        )
        chance = 1.0 if has_no_guard else float(
            np.clip(accuracy + level_delta / 100.0, 0.0, 1.0)
        )
        current_hp = max(
            float(getattr(defender, "current_hp", 0) or 0), 0.0
        )
        return current_hp * chance
    category = getattr(getattr(move, "category", None), "name", "STATUS")
    power = float(getattr(move, "base_power", 0) or 0)
    if category == "STATUS" or power <= 0:
        return 0.0
    attack, defense = ("spa", "spd") if category == "SPECIAL" else ("atk", "def")
    level = getattr(attacker, "level", 50) or 50
    damage = ((2 * level / 5 + 2) * power * _stat(attacker, attack) / max(_stat(defender, defense), 1) / 50 + 2)
    damage *= 1.5 if move.type in getattr(attacker, "types", ()) else 1.0
    damage *= defender.damage_multiplier(move)
    damage *= float(getattr(move, "accuracy", 1.0) or 0.0)
    damage *= 0.925 * (0.75 if spread else 1.0)
    if (
        battle is None
        and category == "PHYSICAL"
        and getattr(getattr(attacker, "status", None), "name", "") == "BRN"
    ):
        damage *= 0.5
    if battle is not None:
        damage *= _public_damage_modifiers(attacker, defender, move, battle, category)
        weather = _condition_names(getattr(battle, "weather", {}))
        move_type = str(getattr(getattr(move, "type", None), "name", getattr(move, "type", ""))).lower()
        if weather.intersection({"raindance", "primordialsea"}):
            damage *= 1.5 if move_type == "water" else 0.5 if move_type == "fire" else 1.0
        elif weather.intersection({"sunnyday", "desolateland"}):
            damage *= 1.5 if move_type == "fire" else 0.5 if move_type == "water" else 1.0
        if target_is_opponent:
            side = getattr(battle, "opponent_side_conditions", {})
            screens = _condition_names(side)
            screen_applies = (
                ("reflect" in screens and category == "PHYSICAL")
                or ("lightscreen" in screens and category == "SPECIAL")
                or "auroraveil" in screens
            )
            infiltrates = bool(getattr(move, "infiltrates", False))
            if screen_applies and not infiltrates:
                damage *= 2.0 / 3.0
    return float(damage)


def _public_expected_damage(
    attacker, defender, move, *, spread=False, battle=None, incoming=False
) -> float:
    """Estimate damage in the acting player's public field state."""
    damage = expected_damage(
        attacker, defender, move, spread=spread, battle=battle,
        target_is_opponent=not incoming,
    )
    if incoming and battle is not None and not bool(getattr(move, "infiltrates", False)):
        screens = _condition_names(getattr(battle, "side_conditions", {}))
        category = getattr(getattr(move, "category", None), "name", "")
        if (
            ("reflect" in screens and category == "PHYSICAL")
            or ("lightscreen" in screens and category == "SPECIAL")
            or "auroraveil" in screens
        ):
            damage *= 2.0 / 3.0
    return float(damage)


def _public_speed_estimate(pokemon, battle, *, own_side: bool) -> float:
    """Estimate speed using only the acting player's visible state or a set prior."""
    speed = _stat(pokemon, "spe")
    own_conditions = _condition_names(
        getattr(battle, "side_conditions" if own_side else "opponent_side_conditions", {})
    )
    weather = _condition_names(getattr(battle, "weather", {}))
    fields = _condition_names(getattr(battle, "fields", {}))
    ability = _id(getattr(pokemon, "ability", ""))
    item = _id(getattr(pokemon, "item", ""))
    status = _id(getattr(pokemon, "status", ""))
    if "tailwind" in own_conditions:
        speed *= 2.0
    if ability == "swiftswim" and weather.intersection({"raindance", "primordialsea"}):
        speed *= 2.0
    elif ability == "chlorophyll" and weather.intersection({"sunnyday", "desolateland"}):
        speed *= 2.0
    elif ability == "sandrush" and "sandstorm" in weather:
        speed *= 2.0
    elif ability == "slushrush" and weather.intersection({"hail", "snowscape"}):
        speed *= 2.0
    elif ability == "surgesurfer" and "electricterrain" in fields:
        speed *= 2.0
    if status in {"par", "paralysis"}:
        speed *= 0.5
    if item == "choicescarf":
        speed *= 1.5
    if ability == "unburden" and item in {"", "none"}:
        speed *= 2.0
    return speed


def choose_max_damage_action(action_mask: np.ndarray, battle, *, order_validator=None) -> np.ndarray:
    """Score legal move/target combinations and choose a compatible action pair."""
    mask = np.asarray(action_mask).reshape(2, -1)
    candidates = []
    for slot in range(2):
        ranked = []
        attacker = battle.active_pokemon[slot]
        for action in np.flatnonzero(mask[slot]):
            descriptor = decode_native_action_index(int(action))
            score = -1.0 if descriptor.kind == "switch" else 0.0
            if descriptor.kind == "move" and attacker is not None:
                # The supplied native mask is the legality source. A PP-less
                # locked move can be represented by a targetless action even
                # when poke-env's parsed order list contains only fake target
                # variants, so skip its redundant membership scan here.
                order = DoublesEnv._action_to_order_individual(np.int64(action), battle, True, slot)
                move = order.order
                target = getattr(getattr(move, "target", None), "name", "")
                if target in {"ALL_ADJACENT", "ALL_ADJACENT_FOES", "ALL"}:
                    foes = [p for p in battle.opponent_active_pokemon if p is not None and not p.fainted]
                    ally = battle.active_pokemon[1 - slot]
                    hits_ally = target in {"ALL_ADJACENT", "ALL"} and ally is not None and not ally.fainted
                    spread = len(foes) + int(hits_ally) > 1
                    score = sum(expected_damage(attacker, p, move, spread=spread) for p in foes)
                    if hits_ally:
                        score -= expected_damage(attacker, ally, move, spread=spread)
                elif descriptor.target_code in (1, 2):
                    score = expected_damage(attacker, battle.opponent_active_pokemon[descriptor.target_code - 1], move)
                elif descriptor.target_code in (-1, -2):
                    score = -expected_damage(attacker, battle.active_pokemon[-descriptor.target_code - 1], move)
            ranked.append((score, int(action)))
        candidates.append(ranked)
    # The old path materialized and sorted every Cartesian action pair, though
    # it usually returned after validating the first one. Pair scores are
    # additive, so sort slot 2 once and merge one score-ordered row per slot-1
    # action. The heap key is exactly the old (-score, action1, action2) key;
    # this returns the first valid pair in that order without building the full
    # Cartesian product.
    first_candidates = sorted(candidates[0], key=lambda candidate: candidate[1])
    second_candidates = sorted(
        candidates[1], key=lambda candidate: (-candidate[0], candidate[1])
    )
    pairs: list[tuple[float, int, int, int, int]] = []

    def push_pair(first_index: int, second_index: int) -> None:
        first_score, first_action = first_candidates[first_index]
        second_score, second_action = second_candidates[second_index]
        heapq.heappush(pairs, (
            -(first_score + second_score), first_action, second_action,
            first_index, second_index,
        ))

    if second_candidates:
        for first_index in range(len(first_candidates)):
            push_pair(first_index, 0)
    validation_failures = []
    pairs_checked = 0
    while pairs:
        _, first, second, first_index, second_index = heapq.heappop(pairs)
        pairs_checked += 1
        result = np.array([first, second], dtype=np.int64)
        try:
            validate = order_validator or DoublesEnv.action_to_order
            validate(result, battle, strict=True)
        except ValueError as error:
            if len(validation_failures) < 3:
                validation_failures.append(str(error))
        else:
            return result
        if second_index + 1 < len(second_candidates):
            push_pair(first_index, second_index + 1)
    raise ValueError(
        "No compatible native action pair: "
        f"{pairs_checked} combinations checked; "
        f"first validation failures={validation_failures}"
    )


def score_max_damage_action(action: int, battle, slot: int) -> float:
    """Return the unchanged max-damage heuristic score for one legal action."""
    descriptor = decode_native_action_index(int(action))
    if descriptor.kind == "switch":
        return -1.0
    if descriptor.kind != "move":
        return 0.0
    attacker = battle.active_pokemon[slot]
    if attacker is None:
        return 0.0
    order = DoublesEnv._action_to_order_individual(
        np.int64(action), battle, True, slot
    )
    move = order.order
    target = getattr(getattr(move, "target", None), "name", "")
    if target in {"ALL_ADJACENT", "ALL_ADJACENT_FOES", "ALL"}:
        foes = [
            pokemon for pokemon in battle.opponent_active_pokemon
            if pokemon is not None and not pokemon.fainted
        ]
        ally = battle.active_pokemon[1 - slot]
        hits_ally = target in {"ALL_ADJACENT", "ALL"} and ally is not None and not ally.fainted
        spread = len(foes) + int(hits_ally) > 1
        score = sum(expected_damage(attacker, foe, move, spread=spread) for foe in foes)
        if hits_ally:
            score -= expected_damage(attacker, ally, move, spread=spread)
        return score
    if descriptor.target_code in (1, 2):
        return expected_damage(
            attacker,
            battle.opponent_active_pokemon[descriptor.target_code - 1],
            move,
        )
    if descriptor.target_code in (-1, -2):
        return -expected_damage(
            attacker,
            battle.active_pokemon[-descriptor.target_code - 1],
            move,
        )
    return 0.0


def choose_public_counterplay_action(
    action_mask: np.ndarray,
    battle,
    public_history,
    *,
    incoming_weight: float = 1.0,
    ko_bonus: float = 0.7,
    setup_bonus: float = 0.0,
    switch_penalty: float = 0.05,
    support_actions: bool = False,
    joint_outgoing: bool = False,
    sequential_damage: bool = False,
    redirect_actions: bool = False,
    threat_move_objective: str = "damage",
    threat_aggregation: str = "sum",
    return_action_scores: bool = False,
    condition_first_action: int | None = None,
    order_validator=None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Choose a tactical joint action using only this player's public battle view.

    Incoming threats are built only from moves recorded in
    ``PublicBattleHistory``. The function deliberately does not accept the
    opponent's battle object or an opponent action prediction, because either
    can expose hidden moves and unrevealed team details in a two-player local
    simulator. Until an opponent move is revealed, selection falls back to
    outgoing damage and legal defensive actions.
    """
    if threat_aggregation not in {"sum", "max_per_attacker"}:
        raise ValueError("threat_aggregation must be 'sum' or 'max_per_attacker'")
    if not np.isfinite(switch_penalty) or switch_penalty < 0:
        raise ValueError("switch_penalty must be finite and nonnegative")
    if threat_move_objective not in {"damage", "hp_fraction"}:
        raise ValueError("threat_move_objective must be 'damage' or 'hp_fraction'")
    masks = np.asarray(action_mask, dtype=np.int8).reshape(2, -1)
    own_team = list(getattr(battle, "team", {}).values())
    opponent_active = list(getattr(battle, "opponent_active_pokemon", ()) or ())[:2]
    own_side = "p2" if str(getattr(battle, "player_role", "p1")).lower().startswith("p2") else "p1"
    opponent_side = "p1" if own_side == "p2" else "p2"

    threats: list[tuple[object, object, bool, int, float, list[tuple[object, float]]]] = []
    for opponent_slot, attacker in enumerate(opponent_active):
        if attacker is None or getattr(attacker, "fainted", False):
            continue
        protocol_slot = f"{opponent_side}{'a' if opponent_slot == 0 else 'b'}"
        public_ident = getattr(public_history, "active_by_slot", {}).get(protocol_slot)
        known = {
            _id(move) for move in getattr(public_history, "revealed_moves", {}).get(public_ident, ())
        } if public_ident is not None else set()
        public_species = (
            getattr(public_history, "species_by_pokemon", {}).get(public_ident, "")
            if public_ident is not None else ""
        )
        public_species = _id(public_species or getattr(attacker, "species", ""))
        sampled_sets = _RANDOM_TEAM_FULL_SET_PRIORS.get("full_species_sets", {}).get(
            public_species, ()
        )
        compatible_sampled = [
            row for row in sampled_sets
            if known.issubset(set(row.get("moves", ())))
        ]
        if compatible_sampled:
            total = sum(int(row.get("count", 0)) for row in compatible_sampled)
            move_hypotheses = [
                (row, int(row.get("count", 0)) / max(total, 1), True)
                for row in compatible_sampled
            ]
        else:
            possible_sets = _RANDOM_TEAM_SET_PRIORS.get("species_sets", {}).get(public_species, ())
            if len(known) >= 4 or not possible_sets:
                move_sets = [(sorted(known), 1.0)] if known else []
            else:
                compatible = [
                    row for row in possible_sets
                    if known.issubset(set(row.get("moves", ())))
                ]
                if not compatible and known:
                    move_sets = [(sorted(known), 1.0)]
                else:
                    total = sum(int(row.get("count", 0)) for row in compatible)
                    move_sets = [
                        (row.get("moves", ()), int(row.get("count", 0)) / max(total, 1))
                        for row in compatible
                    ]
            move_hypotheses = [(moves, probability, False) for moves, probability in move_sets]
        selected_move_weights: dict[str, tuple[object, list[tuple[object, float]]]] = {}
        for set_hypothesis, set_probability, full_set in move_hypotheses:
            move, damage_attacker = (
                _max_damage_move_for_sampled_set(
                    attacker, set_hypothesis, battle,
                    objective=threat_move_objective,
                )
                if full_set
                else _max_damage_move_for_public_set(
                    attacker, set_hypothesis, battle,
                    objective=threat_move_objective,
                )
            )
            if move is None:
                continue
            move_id = _id(getattr(move, "id", ""))
            previous = selected_move_weights.get(move_id)
            models = list(previous[1]) if previous is not None else []
            models.append((damage_attacker, set_probability))
            selected_move_weights[move_id] = (
                move,
                models,
            )
        for move, models in selected_move_weights.values():
            probability = sum(weight for _, weight in models)
            target = _target_name(move)
            threats.append((
                attacker,
                move,
                target in {"alladjacent", "alladjacentfoes", "all"},
                int(getattr(move, "priority", 0) or 0),
                probability,
                models,
            ))

    candidates: list[list[tuple[int, object, NativeActionDescriptor, object]]] = []
    for slot in range(2):
        slot_candidates = []
        for raw_action in np.flatnonzero(masks[slot]):
            action = int(raw_action)
            descriptor = decode_native_action_index(action)
            try:
                order = DoublesEnv._action_to_order_individual(
                    np.int64(action), battle, True, slot
                )
            except (IndexError, ValueError):
                continue
            slot_candidates.append((action, order, descriptor, getattr(order, "order", None)))
        candidates.append(slot_candidates)

    # The joint-action scorer evaluates the same attacker/defender/move damage
    # repeatedly across compatible action pairs. Battle state is unchanged for
    # this decision, so memoize those deterministic values for this call only.
    damage_cache: dict[tuple[int, int, int, bool, bool], float] = {}

    def cached_public_damage(
        attacker, defender, move, *, spread: bool = False, incoming: bool = False
    ) -> float:
        key = (id(attacker), id(defender), id(move), bool(spread), bool(incoming))
        cached = damage_cache.get(key)
        if cached is not None:
            return cached
        damage = _public_expected_damage(
            attacker, defender, move, spread=spread, battle=battle,
            incoming=incoming,
        )
        damage_cache[key] = damage
        return damage

    best_pair: tuple[int, int] | None = None
    best_score = -float("inf")
    ranked: list[tuple[float, tuple[int, int]]] = []
    pair_scores: dict[int, dict[int, float]] = {}
    for first in candidates[0]:
        for second in candidates[1]:
            pair = (first[0], second[0])
            if not DoubleBattleOrder.join_orders([first[1]], [second[1]]):
                continue
            active_after = list(getattr(battle, "active_pokemon", ()) or ())[:2]
            for slot, candidate in enumerate((first, second)):
                descriptor = candidate[2]
                if descriptor.kind == "switch" and descriptor.switch_index is not None:
                    if descriptor.switch_index < len(own_team):
                        active_after[slot] = own_team[descriptor.switch_index]

            outgoing = 0.0
            slot_damage = [0.0, 0.0]
            damage_by_foe = [0.0, 0.0]
            damage_by_slot_foe = [[0.0, 0.0], [0.0, 0.0]]
            slot_ko_bonus = [0.0, 0.0]
            helping_hand_slots = set()
            setup_score = 0.0
            own_guard_ids = set()
            own_wide_guard = False
            own_quick_guard = False
            for slot, candidate in enumerate((first, second)):
                action, _, descriptor, move = candidate
                if descriptor.kind == "switch":
                    outgoing -= float(switch_penalty)
                    continue
                if descriptor.kind != "move" or move is None:
                    continue
                move_id = _id(getattr(move, "id", ""))
                if move_id == "helpinghand":
                    helping_hand_slots.add(slot)
                boosts = getattr(move, "self_boost", None)
                if boosts is None and _target_name(move) == "self":
                    # Some Showdown entries encode self-targeted stat boosts
                    # under the generic `boosts` field rather than
                    # `selfBoost`.
                    boosts = getattr(move, "boosts", None)
                if isinstance(boosts, dict):
                    setup_score += float(setup_bonus) * sum(
                        max(0, int(value)) for value in boosts.values()
                    )
                if move_id in _PROTECT_MOVE_IDS:
                    own_guard_ids.add(slot)
                if move_id == "wideguard":
                    own_wide_guard = True
                if move_id in _QUICK_GUARD_MOVE_IDS:
                    own_quick_guard = True
                attacker = getattr(battle, "active_pokemon", ())[slot]
                if attacker is None:
                    continue
                target_code = descriptor.target_code
                target_name = _target_name(move)
                if target_name in {"alladjacent", "alladjacentfoes", "all"}:
                    foes = [p for p in opponent_active if p is not None and not getattr(p, "fainted", False)]
                    ally = getattr(battle, "active_pokemon", ())[1 - slot]
                    hits_ally = target_name in {"alladjacent", "all"} and ally is not None and not getattr(ally, "fainted", False)
                    spread = len(foes) + int(hits_ally) > 1
                    raw = sum(
                        cached_public_damage(
                            attacker, foe, move, spread=spread
                        )
                        for foe in foes
                    )
                    if hits_ally:
                        raw -= cached_public_damage(
                            attacker, ally, move, spread=spread,
                            incoming=True,
                        )
                    denom = sum(max(float(getattr(p, "current_hp", 1) or 1), 1.0) for p in foes)
                    local_damage = max(raw, 0.0) / max(denom, 1.0)
                    slot_damage[slot] += local_damage
                    if joint_outgoing:
                        for foe_index, foe in enumerate(opponent_active):
                            if foe is not None and not getattr(foe, "fainted", False):
                                dealt = cached_public_damage(
                                    attacker, foe, move, spread=spread
                                )
                                damage_by_foe[foe_index] += dealt
                                damage_by_slot_foe[slot][foe_index] += dealt
                    else:
                        outgoing += local_damage
                    if not joint_outgoing and any(
                        cached_public_damage(
                            attacker, p, move, spread=spread
                        ) >= float(getattr(p, "current_hp", 1) or 1)
                        for p in foes
                    ):
                        slot_ko_bonus[slot] += float(ko_bonus)
                elif target_code in (1, 2):
                    target = opponent_active[target_code - 1] if target_code - 1 < len(opponent_active) else None
                    if target is not None:
                        hp = max(float(getattr(target, "current_hp", 1) or 1), 1.0)
                        raw = cached_public_damage(
                            attacker, target, move
                        )
                        local_damage = raw / hp
                        slot_damage[slot] += local_damage
                        if joint_outgoing:
                            damage_by_foe[target_code - 1] += raw
                            damage_by_slot_foe[slot][target_code - 1] += raw
                        else:
                            if raw >= hp:
                                slot_ko_bonus[slot] += float(ko_bonus)
                elif target_code in (-1, -2):
                    target = getattr(battle, "active_pokemon", ())[abs(target_code) - 1]
                    if target is not None:
                        hp = max(float(getattr(target, "current_hp", 1) or 1), 1.0)
                        raw = cached_public_damage(
                            attacker, target, move, incoming=True
                        )
                        local_damage = -raw / hp
                        slot_damage[slot] += local_damage

            incoming_cost = 0.0
            damage_before_our_move = [0.0, 0.0]
            strongest_threat_by_attacker: dict[int, float] = {}
            for attacker, move, spread, priority, probability, models in threats:
                if spread and own_wide_guard:
                    continue
                if not spread and own_quick_guard and priority > 0:
                    continue
                candidate_cost = 0.0
                foe_slot = next(
                    (i for i, pokemon in enumerate(opponent_active) if pokemon is attacker),
                    None,
                )
                trick_room = "trickroom" in _condition_names(getattr(battle, "fields", {}))
                for model_attacker, model_probability in models:
                    hp = max(float(getattr(attacker, "current_hp", 1) or 1), 1.0)
                    damage_before_attack = 0.0
                    fakeout_stop_probability = 0.0
                    for own_slot, selected in enumerate((first, second)):
                        own_action, _, own_descriptor, own_move = selected
                        own_attacker = getattr(battle, "active_pokemon", ())[own_slot]
                        if (
                            own_attacker is None or own_move is None
                            or own_descriptor.kind != "move"
                        ):
                            continue
                        own_target = _target_name(own_move)
                        if own_target in {"alladjacent", "alladjacentfoes", "all"}:
                            targets_this_foe = True
                            own_spread = sum(
                                p is not None and not getattr(p, "fainted", False)
                                for p in opponent_active
                            ) > 1
                        else:
                            targets_this_foe = (
                                foe_slot is not None
                                and own_descriptor.target_code == foe_slot + 1
                            )
                            own_spread = False
                        if not targets_this_foe:
                            continue
                        own_priority = int(getattr(own_move, "priority", 0) or 0)
                        if own_priority < priority:
                            continue
                        own_speed = _public_speed_estimate(own_attacker, battle, own_side=True)
                        foe_speed = _public_speed_estimate(
                            model_attacker, battle, own_side=False
                        )
                        if own_priority == priority:
                            if own_speed == foe_speed:
                                acts_first = 0.5
                            else:
                                acts_first = float(
                                    own_speed < foe_speed if trick_room else own_speed > foe_speed
                                )
                            if acts_first <= 0:
                                continue
                        else:
                            acts_first = 1.0
                        dealt = cached_public_damage(
                            own_attacker, attacker, own_move,
                            spread=own_spread,
                        )
                        damage_before_attack += acts_first * dealt
                        if _id(getattr(own_move, "id", "")) == "fakeout":
                            fakeout_stop_probability = max(
                                fakeout_stop_probability, 0.5 * acts_first
                            )
                    stopped = min(damage_before_attack / hp, 1.0)
                    stopped = 1.0 - (1.0 - stopped) * (1.0 - fakeout_stop_probability)
                    redirect_slot: int | None = None
                    redirect_probability = 0.0
                    if (
                        redirect_actions and not spread
                        and _target_name(move) in {"normal", "adjacentfoe", "any"}
                    ):
                        for own_slot, selected in enumerate((first, second)):
                            redirect_move = selected[3]
                            redirector = active_after[own_slot]
                            if (
                                redirector is None or getattr(redirector, "fainted", False)
                                or redirect_move is None
                                or _id(getattr(redirect_move, "id", "")) not in _REDIRECTION_MOVE_IDS
                            ):
                                continue
                            redirect_id = _id(getattr(redirect_move, "id", ""))
                            if redirect_id == "ragepowder":
                                types = {
                                    _id(value) for value in getattr(redirector, "types", ())
                                }
                                ability = _id(getattr(redirector, "ability", ""))
                                item = _id(getattr(redirector, "item", ""))
                                if "grass" in types or ability == "overcoat" or item == "safetygoggles":
                                    continue
                            redirect_priority = int(getattr(redirect_move, "priority", 0) or 0)
                            if redirect_priority < priority:
                                continue
                            if redirect_priority == priority:
                                redirect_speed = _public_speed_estimate(
                                    redirector, battle, own_side=True
                                )
                                attacker_speed = _public_speed_estimate(
                                    model_attacker, battle, own_side=False
                                )
                                if redirect_speed == attacker_speed:
                                    redirect_probability = 0.5
                                else:
                                    acts_first = (
                                        redirect_speed < attacker_speed
                                        if trick_room else redirect_speed > attacker_speed
                                    )
                                    redirect_probability = float(acts_first)
                            else:
                                redirect_probability = 1.0
                            if redirect_probability > 0:
                                redirect_slot = own_slot
                                break
                    if sequential_damage:
                        trick_room = "trickroom" in _condition_names(
                            getattr(battle, "fields", {})
                        )
                        for own_slot, selected in enumerate((first, second)):
                            _, _, own_descriptor, own_move = selected
                            own_attacker = getattr(battle, "active_pokemon", ())[own_slot]
                            if (
                                own_attacker is None or own_move is None
                                or own_descriptor.kind != "move"
                            ):
                                continue
                            own_priority = int(getattr(own_move, "priority", 0) or 0)
                            if priority > own_priority:
                                acts_first = 1.0
                            elif priority < own_priority:
                                acts_first = 0.0
                            else:
                                own_speed = _public_speed_estimate(
                                    own_attacker, battle, own_side=True
                                )
                                foe_speed = _public_speed_estimate(
                                    model_attacker, battle, own_side=False
                                )
                                if own_speed == foe_speed:
                                    acts_first = 0.5
                                else:
                                    acts_first = float(
                                        foe_speed < own_speed if trick_room
                                        else foe_speed > own_speed
                                    )
                            if acts_first <= 0:
                                continue
                            target_index = own_slot
                            if not spread:
                                target_damages = [
                                    cached_public_damage(
                                        model_attacker, defender, move,
                                        incoming=True,
                                    ) if defender is not None and not getattr(defender, "fainted", False)
                                    else -1.0
                                    for defender in active_after
                                ]
                                target_index = int(np.argmax(target_damages))
                            if target_index != own_slot:
                                continue
                            defender = active_after[own_slot]
                            if defender is None or getattr(defender, "fainted", False):
                                continue
                            raw = cached_public_damage(
                                model_attacker, defender, move,
                                spread=spread and sum(
                                    p is not None and not getattr(p, "fainted", False)
                                    for p in active_after
                                ) > 1,
                                incoming=True,
                            )
                            if own_slot in own_guard_ids:
                                guard_move = getattr((first, second)[own_slot][3], "id", "")
                                streak = int(getattr(
                                    getattr(battle, "active_pokemon", ())[own_slot],
                                    "protect_counter", 0,
                                ) or 0)
                                guard_probability = (
                                    1.0 / (3.0 ** streak)
                                    if _id(guard_move) in _PROTECT_MOVE_IDS else 1.0
                                )
                            else:
                                guard_probability = 0.0
                            damage_before_our_move[own_slot] += (
                                model_probability * (1.0 - stopped) * acts_first
                                * raw * (1.0 - guard_probability)
                            )
                    model_cost = 0.0
                    natural_target_cost = 0.0
                    redirected_target_cost = 0.0
                    for own_slot, defender in enumerate(active_after):
                        if defender is None or getattr(defender, "fainted", False):
                            continue
                        if own_slot in own_guard_ids:
                            guard_move = getattr((first, second)[own_slot][3], "id", "")
                            streak = int(getattr(getattr(battle, "active_pokemon", ())[own_slot], "protect_counter", 0) or 0)
                            guard_probability = 1.0 / (3.0 ** streak) if _id(guard_move) in _PROTECT_MOVE_IDS else 1.0
                        else:
                            guard_probability = 0.0
                        incoming_damage = cached_public_damage(
                            model_attacker, defender, move,
                            spread=spread and sum(
                                p is not None and not getattr(p, "fainted", False)
                                for p in active_after
                            ) > 1,
                            incoming=True,
                        ) * (1.0 - guard_probability)
                        defender_hp = max(float(getattr(defender, "current_hp", 1) or 1), 1.0)
                        cost = min(incoming_damage / defender_hp, 1.5)
                        if incoming_damage >= defender_hp:
                            cost += 0.35
                        natural_target_cost = max(natural_target_cost, cost)
                        if own_slot == redirect_slot:
                            redirected_target_cost = cost
                    if redirect_slot is not None:
                        model_cost = (
                            redirect_probability * redirected_target_cost
                            + (1.0 - redirect_probability) * natural_target_cost
                        )
                    else:
                        model_cost = natural_target_cost
                    candidate_cost += model_probability * (1.0 - stopped) * model_cost
                if threat_aggregation == "max_per_attacker":
                    key = id(attacker)
                    strongest_threat_by_attacker[key] = max(
                        strongest_threat_by_attacker.get(key, 0.0), candidate_cost
                    )
                else:
                    incoming_cost += candidate_cost
            if threat_aggregation == "max_per_attacker":
                incoming_cost = sum(strongest_threat_by_attacker.values())

            survival = [1.0, 1.0]
            if sequential_damage:
                for slot, pokemon in enumerate(getattr(battle, "active_pokemon", ())[:2]):
                    if pokemon is not None and not getattr(pokemon, "fainted", False):
                        hp = max(float(getattr(pokemon, "current_hp", 1) or 1), 1.0)
                        survival[slot] = float(np.clip(
                            1.0 - damage_before_our_move[slot] / hp, 0.0, 1.0
                        ))
            outgoing = 0.0
            if joint_outgoing:
                for foe_index, foe in enumerate(opponent_active):
                    if foe is None or getattr(foe, "fainted", False):
                        continue
                    hp = max(float(getattr(foe, "current_hp", 1) or 1), 1.0)
                    dealt = sum(
                        damage_by_slot_foe[slot][foe_index] * survival[slot]
                        for slot in range(2)
                    )
                    outgoing += min(dealt / hp, 1.0)
                    if dealt >= hp:
                        outgoing += float(ko_bonus)
                outgoing += sum(min(value, 0.0) * survival[slot] for slot, value in enumerate(slot_damage))
            else:
                outgoing = sum(
                    (slot_damage[slot] + slot_ko_bonus[slot]) * survival[slot]
                    for slot in range(2)
                )
            if support_actions:
                for slot in helping_hand_slots:
                    outgoing += (
                        0.5 * max(slot_damage[1 - slot], 0.0)
                        * survival[slot] * survival[1 - slot]
                    )

            score = outgoing + setup_score - float(incoming_weight) * incoming_cost
            ranked.append((score, pair))
            pair_scores.setdefault(pair[0], {})[pair[1]] = score
            if score > best_score:
                best_score, best_pair = score, pair

    if best_pair is None:
        raise ValueError("No compatible legal public-counterplay action pair")
    if order_validator is not None:
        for _, pair in sorted(ranked, key=lambda row: row[0], reverse=True):
            try:
                order_validator(np.asarray(pair, dtype=np.int64), battle, strict=True)
            except ValueError:
                continue
            best_pair = pair
            break
        else:
            raise ValueError("No public-counterplay action pair passed strict validation")
    chosen = np.asarray(best_pair, dtype=np.int64)
    if not return_action_scores:
        return chosen
    action_scores = np.full((2, masks.shape[1]), -1e6, dtype=np.float32)
    for first_action, second_scores in pair_scores.items():
        action_scores[0, first_action] = max(second_scores.values())
    joint_scores = np.full((masks.shape[1], masks.shape[1]), -1e6, dtype=np.float32)
    for first_action, second_scores in pair_scores.items():
        for second_action, score in second_scores.items():
            joint_scores[first_action, second_action] = score
    score_condition = int(best_pair[0])
    if condition_first_action in pair_scores:
        score_condition = int(condition_first_action)
    conditioned_scores = pair_scores[score_condition]
    for second_action, score in conditioned_scores.items():
        action_scores[1, second_action] = score
    conditioned_second = max(conditioned_scores, key=conditioned_scores.get)
    return chosen, action_scores, int(conditioned_second), joint_scores
