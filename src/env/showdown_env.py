"""Native poke-env environment for real Pokémon Showdown battles."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)

from src.features.volatile_effects import (
    OBSERVATION_VOLATILE_OFFSET,
    POKEMON_VOLATILE_EFFECTS,
)


class ShowdownDoublesEnv(DoublesEnv):
    """Concrete two-player VGC environment backed by a Showdown websocket.

    The action format is poke-env's native ``MultiDiscrete([N, N])`` format:
    one individual action for each active Pokémon. Observations are compact
    numeric vectors plus the native action mask, allowing callers to plug in
    their own policy encoder without inventing a second protocol.
    """

    def __init__(
        self,
        *,
        battle_format: str = "gen8doublesou",
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        server_configuration: Optional[ServerConfiguration] = None,
        team: Optional[str] = None,
        accept_open_team_sheet: bool = False,
        start_listening: bool = True,
        fake: bool = False,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            battle_format=battle_format,
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            server_configuration=server_configuration or LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
            start_listening=start_listening,
            fake=fake,
            strict=strict,
            **kwargs,
        )
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(512,),
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        self._previous_hp_advantage: Dict[str, float] = {}
        self._last_reward_components: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _battle_reward_key(battle: AbstractBattle) -> str:
        """Identify one player's view of a battle for reward bookkeeping.

        The two mirrored ``AbstractBattle`` objects in a DoublesEnv share the
        same Showdown battle tag.  Include the player identity so progress
        baselines and diagnostics cannot bleed across the two perspectives.
        """
        battle_tag = str(getattr(battle, "battle_tag", ""))
        player = getattr(battle, "player_username", None)
        if player:
            return f"{battle_tag}:{player}"
        return battle_tag or str(id(battle))

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Encode observable battle state into a normalized fixed-size vector."""
        observation = np.zeros(512, dtype=np.float32)
        observation[0] = min(max(battle.turn / 100.0, 0.0), 1.0)

        team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())
        for index, pokemon in enumerate((team + opponent_team)[:12]):
            hp = pokemon.current_hp
            max_hp = pokemon.max_hp
            observation[1 + index] = (
                float(hp) / float(max_hp) if hp is not None and max_hp else 0.0
            )
            observation[13 + index] = 0.0 if pokemon.fainted else 1.0

        for index, pokemon in enumerate(battle.active_pokemon[:2]):
            if pokemon is not None:
                observation[25 + index] = (
                    float(pokemon.current_hp) / float(pokemon.max_hp)
                    if pokemon.current_hp is not None and pokemon.max_hp
                    else 0.0
                )

        # The original 32 used features stopped at HP/alive flags. Reserve the
        # next 32 positions for tactical state needed to judge incoming damage
        # and defensive choices in doubles.
        boost_names = ("atk", "def", "spa", "spd", "spe")
        active = list(battle.active_pokemon[:2]) + list(
            getattr(battle, "opponent_active_pokemon", ())[:2]
        )
        for slot, pokemon in enumerate(active):
            boosts = getattr(pokemon, "boosts", {}) or {} if pokemon is not None else {}
            for boost_index, name in enumerate(boost_names):
                observation[32 + slot * len(boost_names) + boost_index] = (
                    float(np.clip(boosts.get(name, 0), -6, 6)) / 6.0
                )

        weather_names = {getattr(weather, "name", str(weather)).upper()
                         for weather in getattr(battle, "weather", {})}
        observation[52:56] = [
            float(bool(weather_names & {"RAINDANCE", "PRIMORDIALSEA"})),
            float(bool(weather_names & {"SUNNYDAY", "DESOLATELAND"})),
            float("SANDSTORM" in weather_names),
            float(bool(weather_names & {"HAIL", "SNOW", "SNOWSCAPE"})),
        ]
        field_names = {getattr(field, "name", str(field)).upper()
                       for field in getattr(battle, "fields", {})}
        observation[56:60] = [
            float(name in field_names)
            for name in ("ELECTRIC_TERRAIN", "GRASSY_TERRAIN", "MISTY_TERRAIN", "PSYCHIC_TERRAIN")
        ]
        observation[60] = float("TRICK_ROOM" in field_names)
        observation[61] = float("GRAVITY" in field_names)
        own_conditions = {getattr(item, "name", str(item)).upper()
                          for item in getattr(battle, "side_conditions", {})}
        opposing_conditions = {getattr(item, "name", str(item)).upper()
                               for item in getattr(battle, "opponent_side_conditions", {})}
        observation[62] = float("TAILWIND" in own_conditions)
        observation[63] = float("TAILWIND" in opposing_conditions)
        own_normalized = {name.replace("_", "").replace("-", "") for name in own_conditions}
        opposing_normalized = {
            name.replace("_", "").replace("-", "") for name in opposing_conditions
        }
        screen_names = ("reflect", "lightscreen", "auroraveil")
        observation[100:103] = [float(name in own_normalized) for name in screen_names]
        observation[103:106] = [
            float(name in opposing_normalized) for name in screen_names
        ]

        status_indices = {"BRN": 1, "FRZ": 2, "PAR": 3, "PSN": 4, "SLP": 5, "TOX": 6}
        for index, pokemon in enumerate(team[:6]):
            status_name = getattr(getattr(pokemon, "status", None), "name", "")
            observation[64 + index] = status_indices.get(status_name, 0) / 6.0
        for index, pokemon in enumerate(opponent_team[:6]):
            status_name = getattr(getattr(pokemon, "status", None), "name", "")
            observation[94 + index] = status_indices.get(status_name, 0) / 6.0

        # Preserve rare but decisive volatile state for the battle policy.
        # These slots were reserved in the original 512-wide observation.
        for index, pokemon in enumerate((team + opponent_team)[:12]):
            effects = getattr(pokemon, "effects", {}) or {}
            effect_names = {
                getattr(effect, "name", str(effect)).upper()
                .replace("_", "").replace("-", "")
                for effect in effects
            }
            perish = next(
                (counter for counter in range(4)
                 if f"PERISH{counter}" in effect_names),
                None,
            )
            offset = 106 + index * 3
            observation[offset] = float(perish is not None)
            observation[offset + 1] = (float(perish) / 3.0) if perish is not None else 0.0
            observation[offset + 2] = float("TRAPPED" in effect_names)

            # The original 512-value observation reserved 370 positions after
            # the known volatile block. Preserve a selected, fixed-schema set
            # of currently observable volatile effects per roster member.
            effect_offset = OBSERVATION_VOLATILE_OFFSET + index * len(
                POKEMON_VOLATILE_EFFECTS
            )
            for effect_index, effect_name in enumerate(POKEMON_VOLATILE_EFFECTS):
                observation[effect_offset + effect_index] = float(
                    effect_name in effect_names
                )

        def mark_active(roster, active_pokemon, offset):
            for slot, active_mon in enumerate(active_pokemon[:2]):
                if active_mon is None:
                    continue
                roster_index = next(
                    (index for index, member in enumerate(roster) if member is active_mon),
                    None,
                )
                if roster_index is None:
                    active_ident = getattr(active_mon, "ident", None)
                    roster_index = next(
                        (index for index, member in enumerate(roster)
                         if getattr(member, "ident", None) == active_ident),
                        None,
                    )
                if roster_index is not None and roster_index < 6:
                    observation[offset + slot * 6 + roster_index] = 1.0

        mark_active(team, battle.active_pokemon, 70)
        mark_active(opponent_team, getattr(battle, "opponent_active_pokemon", ()), 82)

        return observation

    @staticmethod
    def get_action_mask(battle: DoubleBattle) -> list[int]:
        """Mask action indices by round-tripping poke-env's actual legal orders.

        poke-env's generic doubles mask is reconstructed from available moves
        and targets. That can include target variants which are not present in
        Showdown's current request (and miss some transient switch requests).
        Retain indices that convert to current orders, with one protocol
        exception: a PP-less locked move is submitted targetless even if
        poke-env exposes fake target-bearing variants for it.
        """
        action_count = DoublesEnv.get_action_space_size(battle.gen)
        masks: list[int] = []
        for position in range(2):
            legal_orders = {str(order) for order in battle.valid_orders[position]}
            position_mask = [0] * action_count
            is_trapped = bool(getattr(battle, "trapped", [False, False])[position])
            maybe_trapped = bool(
                getattr(battle, "maybe_trapped", [False, False])[position]
            )
            is_forced_switch = bool(
                getattr(battle, "force_switch", [False, False])[position]
            )
            for action_index in range(action_count):
                # The request parser can retain switch orders in
                # ``valid_orders`` after the active Pokemon becomes trapped.
                # Showdown still rejects those voluntary switches. A forced
                # replacement is different: trapping does not block a
                # phaze/forced switch, so preserve those choices.
                if 1 <= action_index <= 6 and (is_trapped or maybe_trapped) and not is_forced_switch:
                    continue
                try:
                    # `fake=True` skips poke-env's own list-based validity
                    # scan; the cached set membership below performs the same
                    # legality check without rebuilding that list per action.
                    order = DoublesEnv._action_to_order_individual(
                        np.int64(action_index), battle, True, position
                    )
                except (IndexError, ValueError):
                    continue
                if str(order) in legal_orders:
                    position_mask[action_index] = 1
            # A locked move request omits PP and Showdown expects a targetless
            # continuation. poke-env may still expose fake explicit-target
            # orders for that move, so the ordinary round-trip above misses
            # the valid action. Add the targetless native index that both
            # headless and local-server adapters submit for this request.
            last_request = getattr(battle, "_last_request", {}) or {}
            active_requests = last_request.get("active", [])
            active_request = (
                active_requests[position]
                if position < len(active_requests) else {}
            )
            request_moves = active_request.get("moves", [])
            if (
                len(request_moves) == 1
                and request_moves[0].get("pp") is None
                and battle.active_pokemon[position] is not None
            ):
                move_id = str(request_moves[0].get("id", ""))
                available = list(battle.available_moves[position])
                available_ids = [move.id for move in available]
                known_moves = list(battle.active_pokemon[position].moves.values())[:4]
                known_ids = [move.id for move in known_moves]
                move_order = (
                    available
                    if len(available_ids) == 1 and available_ids[0] not in known_ids
                    else known_moves
                )
                move_index = next(
                    (index for index, move in enumerate(move_order) if move.id == move_id),
                    None,
                )
                if move_index is not None:
                    targetless_index = 7 + 5 * move_index + 2
                    if targetless_index < action_count:
                        position_mask[targetless_index] = 1
                    if active_request.get("canMegaEvo"):
                        mega_index = targetless_index + 20
                        if mega_index < action_count:
                            position_mask[mega_index] = 1
                    if active_request.get("canTerastallize"):
                        tera_index = targetless_index + 80
                        if tera_index < action_count:
                            position_mask[tera_index] = 1
            # Match poke-env's pass sentinel for a side that is waiting while
            # its partner resolves an action. It is not a battle order.
            if not any(position_mask) and getattr(battle, "_wait", False):
                position_mask[0] = 1
            masks.extend(position_mask)
        return masks

    @staticmethod
    def action_to_order(
        action: np.ndarray,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> Any:
        """Translate preview indices against the server's current valid orders."""
        action_values = np.asarray(action, dtype=np.int64).reshape(-1)
        for position, value in enumerate(action_values[:2]):
            is_trapped = bool(getattr(battle, "trapped", [False, False])[position])
            maybe_trapped = bool(
                getattr(battle, "maybe_trapped", [False, False])[position]
            )
            is_forced_switch = bool(
                getattr(battle, "force_switch", [False, False])[position]
            )
            if (
                1 <= int(value) <= 6
                and (is_trapped or maybe_trapped)
                and not is_forced_switch
            ):
                raise ValueError(
                    f"voluntary switch action {int(value)} is illegal while "
                    f"slot {position} is trapped"
                )
        # During the short request/update gap after an atomic /team command,
        # poke-env can expose no active Pokemon while valid_orders already
        # contains the post-preview switch orders.  Only use that fallback
        # when both order lists are populated; otherwise the native converter
        # must handle the regular battle request (or its default behavior).
        if any(pokemon is None for pokemon in battle.active_pokemon) and all(
            battle.valid_orders
        ):
            orders = []
            for position in range(2):
                valid = [
                    order
                    for order in battle.valid_orders[position]
                    if hasattr(getattr(order, "order", None), "base_species")
                    and not getattr(getattr(order, "order", None), "selected_in_teampreview", False)
                ]
                if valid:
                    index = min(max(int(action[position]) - 1, 0), len(valid) - 1)
                    orders.append(valid[index])
            if len(orders) == 2:
                joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
                if joined:
                    return joined[0]
        targetless_orders = {}
        last_request = getattr(battle, "_last_request", {}) or {}
        active_requests = last_request.get("active", [])
        for position, value in enumerate(action_values[:2]):
            active_request = (
                active_requests[position]
                if position < len(active_requests) else {}
            )
            moves = active_request.get("moves", [])
            if len(moves) != 1 or moves[0].get("pp") is not None:
                continue
            try:
                order = DoublesEnv._action_to_order_individual(
                    np.int64(value), battle, True, position
                )
            except (IndexError, ValueError):
                continue
            if (
                getattr(getattr(order, "order", None), "id", None) == moves[0].get("id")
                and getattr(order, "move_target", None) == 0
                and (not order.mega or active_request.get("canMegaEvo"))
                and (not order.terastallize or active_request.get("canTerastallize"))
            ):
                targetless_orders[position] = order
        if targetless_orders:
            orders = []
            for position, value in enumerate(action_values[:2]):
                if position in targetless_orders:
                    orders.append(targetless_orders[position])
                else:
                    orders.append(
                        DoublesEnv._action_to_order_individual(
                            np.int64(value), battle, fake, position
                        )
                    )
            joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
            if joined:
                return joined[0]
            if strict:
                raise ValueError("targetless locked move is incompatible with the other action")
        return DoublesEnv.action_to_order(action, battle, fake=fake, strict=strict)

    @staticmethod
    def _hp_advantage(battle: AbstractBattle) -> float:
        own_hp = sum(
            (pokemon.current_hp or 0) / (pokemon.max_hp or 1)
            for pokemon in battle.team.values()
        )
        opponent_hp = sum(
            (pokemon.current_hp or 0) / (pokemon.max_hp or 1)
            for pokemon in battle.opponent_team.values()
        )
        return float((own_hp - opponent_hp) / 12.0)

    def calc_reward_components(self, battle: AbstractBattle) -> Dict[str, float]:
        """Return separately auditable terminal and HP-progress rewards.

        Progress is the change in the team's HP advantage since this battle's
        previous reward calculation, rather than the absolute HP difference.
        This makes the shaping signal invariant to the initial team health.
        """
        battle_key = self._battle_reward_key(battle)
        advantage = self._hp_advantage(battle)
        previous = self._previous_hp_advantage.setdefault(battle_key, advantage)
        hp_progress = float(np.clip(advantage - previous, -0.1, 0.1))
        terminal_outcome = 1.0 if battle.won else -1.0 if battle.lost else 0.0
        components = {
            "terminal_outcome": terminal_outcome,
            "hp_progress": hp_progress,
            "total": terminal_outcome + hp_progress,
        }
        self._previous_hp_advantage[battle_key] = advantage
        self._last_reward_components[battle_key] = components
        return components

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Return terminal outcome plus delta-based HP-progress shaping."""
        return self.calc_reward_components(battle)["total"]

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset native state and discard reward history from prior battles."""
        self._previous_hp_advantage.clear()
        self._last_reward_components.clear()
        return super().reset(*args, **kwargs)

    def step(self, actions: Any) -> Any:
        """Preserve Gymnasium termination semantics and expose reward details."""
        result = super().step(actions)
        observations, rewards, terminated, truncated, infos = result
        battle_by_agent = dict(zip(self.possible_agents, (self.battle1, self.battle2)))
        for agent in self.possible_agents:
            info = infos.setdefault(agent, {})
            battle = battle_by_agent.get(agent)
            key = self._battle_reward_key(battle) if battle is not None else ""
            components = self._last_reward_components.get(key)
            if components is not None:
                info["reward_components"] = dict(components)
            info["terminated"] = bool(terminated.get(agent, False))
            info["truncated"] = bool(truncated.get(agent, False))
            info["episode_done"] = info["terminated"] or info["truncated"]
        return observations, rewards, terminated, truncated, infos
