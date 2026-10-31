"""Projection of omniscient match records into closed-sheet player views."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable

import numpy as np
from poke_env.data.normalize import to_id_str


_AGENT_MAP_FIELDS = (
    "observations",
    "action_masks",
    "raw_action_masks",
    "actions",
    "policy_proposed_actions",
    "public_mechanics",
    "opponent_species",
    "predicted_opponent_actions",
    "counterplay_action_scores",
    "counterplay_teacher_first_action",
    "rewards",
    "reward_components",
    "terminated",
    "truncated",
    "log_probs",
    "search_action_behavior_log_probs",
    "values",
    "native_action_features",
    "teacher_actions",
    "teacher_joint_action_distributions",
    "teacher_action_scores",
    "teacher_action_conditioning",
)
_AGENT_MAP_FIELD_SET = frozenset(_AGENT_MAP_FIELDS)
_ALLOWED_STEP_FIELDS = _AGENT_MAP_FIELD_SET | {"turn", "turn_index"}

_EPISODE_FIELDS = (
    "format",
    "opponent",
    "opponent_tag",
    "player_strategy",
    "policy_checkpoint",
    "opponent_checkpoint",
    "policy_temperature",
    "policy_deterministic",
    "trainable_agent",
    "counterplay_prior_weight",
    "team_selection_temperature",
    "native_action_feature_dim",
    "native_action_feature_variant",
    "native_action_feature_encoder_version",
    "teams_differ",
    "steps",
    "action_adjustments",
    "max_damage_score_gate_overrides",
    "max_damage_gate_on_policy",
    "timed_out",
    "error",
)

_PLAYER_VIEW_FIELDS = frozenset(
    _EPISODE_FIELDS
    + (
        "record_schema_version",
        "perspective_agent",
        "perspective_team",
        "episode",
        "teams",
        "agent_to_team",
        "trajectory",
        "rewards",
        "terminated",
        "truncated",
        "preview_actions",
        "preview_log_probs",
        "first_turn_action_data",
        # Training labels only. They are never passed to observation builders.
        "hidden_set_targets",
    )
)

_FIRST_TURN_ACTION_FIELDS = frozenset(
    {
        "action", "observation", "legal_action_mask",
        "recorded_action_mask", "source",
    }
)
_FIRST_TURN_FEATURE_FIELDS = frozenset(
    {"native_action_features", "public_context_features"}
)


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, list):
        return all(_finite_numeric_tree(item) for item in value)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_native_action_features(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, list) or len(value) != 2:
        return False
    if any(not isinstance(slot, list) or len(slot) != 126 for slot in value):
        return False
    feature_width = None
    for slot in value:
        for action_features in slot:
            if not isinstance(action_features, list) or not action_features:
                return False
            if feature_width is None:
                feature_width = len(action_features)
            if len(action_features) != feature_width:
                return False
    # The shape checks above establish a rectangular 2 x 126 x feature-width
    # tree. Let NumPy check finiteness in compiled code. `np.asarray` promotes
    # bools mixed with numbers to integer, so explicitly reject bool leaves;
    # pure-bool, string, nested, and object arrays fail the dtype check.
    numeric = np.asarray(value)
    if numeric.dtype.kind not in "iuf":
        return False
    if any(
        type(component) is bool
        for slot in value
        for action_features in slot
        for component in action_features
    ):
        return False
    if not np.isfinite(numeric).all():
        return False
    return True


def _valid_public_context_features(value: Any) -> bool:
    return (
        value is None
        or (
            isinstance(value, list)
            and len(value) == 73
            and _finite_numeric_tree(value)
        )
    )


def _members(team: Any) -> list[dict[str, Any]]:
    if isinstance(team, dict):
        team = team.get("pokemon", team.get("team", []))
    if not isinstance(team, list):
        return []
    return [pokemon for pokemon in team if isinstance(pokemon, dict)]


def _public_preview(team: Any) -> dict[str, list[dict[str, str]]]:
    """Keep only the species disclosed by ordinary six-Pokemon preview."""
    species = []
    for pokemon in _members(team):
        name = pokemon.get("species") or pokemon.get("name")
        if name:
            species.append({"species": str(name)})
    return {"pokemon": species}


def _agent_map(value: Any, agent: str) -> dict[str, Any]:
    return {agent: deepcopy(value[agent])} if isinstance(value, dict) and agent in value else {}


def project_episode_to_player_views(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a live match record into one closed-sheet record per player.

    The acting player's own set stays available to training. The opponent's
    observation roster retains preview species only. Per-step values and
    actions are restricted to that player's perspective. A separate
    `hidden_set_targets` field supplies auxiliary prediction labels to the
    trainer; observation builders must never read that field.
    """
    teams = record.get("teams")
    agent_to_team = record.get("agent_to_team")
    trajectory = record.get("trajectory")
    if not isinstance(teams, dict) or set(teams) != {"player1", "player2"}:
        raise ValueError("match record must include player1 and player2 teams")
    if not isinstance(agent_to_team, dict) or not isinstance(trajectory, list):
        raise ValueError("match record must include agent mappings and a trajectory")

    views = []
    for agent, side in agent_to_team.items():
        if side not in teams:
            raise ValueError(f"agent {agent!r} maps to unknown team {side!r}")
        opponent_side = "player2" if side == "player1" else "player1"
        safe_teams = {
            side: deepcopy(teams[side]),
            opponent_side: _public_preview(teams[opponent_side]),
        }
        trajectory_by_agent = record.get("trajectory_by_agent", {})
        player_trajectory = (
            trajectory_by_agent.get(agent, trajectory)
            if isinstance(trajectory_by_agent, dict)
            else trajectory
        )
        perspective_trajectory = []
        for step in player_trajectory:
            safe_step = {
                field: _agent_map(step.get(field), agent)
                for field in _AGENT_MAP_FIELDS
                if field in step
            }
            # Keep non-agent scalar metadata, if present, without copying any
            # arbitrary nested structures from an omniscient step.
            for field in ("turn", "turn_index"):
                if field in step:
                    safe_step[field] = deepcopy(step[field])
            perspective_trajectory.append(safe_step)

        perspective = {
            field: deepcopy(record[field])
            for field in _EPISODE_FIELDS
            if field in record
        }
        perspective.update(
            {
                "record_schema_version": 2,
                "perspective_agent": agent,
                "perspective_team": side,
                "episode": record.get("episode"),
                "teams": safe_teams,
                "agent_to_team": {agent: side},
                "trajectory": perspective_trajectory,
                "rewards": _agent_map(record.get("rewards"), agent),
                "terminated": _agent_map(record.get("terminated"), agent),
                "truncated": _agent_map(record.get("truncated"), agent),
                # Keep targets in a separate, explicitly supervised field. The
                # opponent roster in `teams` remains species-only, and runtime
                # policy inputs are built exclusively from that roster plus
                # per-turn public facts.
                "hidden_set_targets": {
                    "pokemon": [
                        {
                            "moves": [str(move) for move in list(member.get("moves", ()))[:4]],
                            "item": str(member.get("item") or ""),
                            "ability": str(member.get("ability") or ""),
                        }
                        for member in _members(teams[opponent_side])[:6]
                    ]
                },
                "teams_differ": record.get("teams_differ") is True,
                "preview_actions": (
                    {side: record["preview_actions"][side]}
                    if isinstance(record.get("preview_actions"), dict)
                    and side in record["preview_actions"]
                    else {}
                ),
                "preview_log_probs": (
                    {side: record["preview_log_probs"][side]}
                    if isinstance(record.get("preview_log_probs"), dict)
                    and side in record["preview_log_probs"]
                    else {}
                ),
            }
        )
        first_turn_action_data = record.get("first_turn_action_data", {})
        if isinstance(first_turn_action_data, dict) and agent in first_turn_action_data:
            perspective["first_turn_action_data"] = deepcopy(
                first_turn_action_data[agent]
            )
        views.append(perspective)
    return views


def is_perspective_safe_record(record: dict[str, Any]) -> bool:
    """Return whether a row satisfies the closed-sheet v2 data contract."""
    if record.get("record_schema_version") != 2:
        return False
    agent = record.get("perspective_agent")
    side = record.get("perspective_team")
    if not isinstance(agent, str) or side not in {"player1", "player2"}:
        return False
    if set(record) - _PLAYER_VIEW_FIELDS:
        return False
    if record.get("agent_to_team") != {agent: side}:
        return False
    teams = record.get("teams")
    if not isinstance(teams, dict) or set(teams) != {"player1", "player2"}:
        return False
    opponent_side = "player2" if side == "player1" else "player1"
    opponent = teams[opponent_side]
    if not isinstance(opponent, dict) or set(opponent) != {"pokemon"}:
        return False
    opponent_members = opponent.get("pokemon")
    if not isinstance(opponent_members, list) or len(opponent_members) != 6:
        return False
    if any(
        not isinstance(pokemon, dict)
        or set(pokemon) != {"species"}
        or not isinstance(pokemon.get("species"), str)
        for pokemon in opponent_members
    ):
        return False
    if not isinstance(record.get("trajectory"), list):
        return False
    targets = record.get("hidden_set_targets")
    if targets is not None:
        if not isinstance(targets, dict) or set(targets) != {"pokemon"}:
            return False
        target_members = targets.get("pokemon")
        if not isinstance(target_members, list) or len(target_members) != 6:
            return False
        if any(
            not isinstance(member, dict)
            or set(member) != {"moves", "item", "ability"}
            or not isinstance(member["moves"], list)
            or len(member["moves"]) > 4
            or any(not isinstance(move, str) for move in member["moves"])
            or not isinstance(member["item"], str)
            or not isinstance(member["ability"], str)
            for member in target_members
        ):
            return False
    for step in record["trajectory"]:
        if not isinstance(step, dict) or not step.keys() <= _ALLOWED_STEP_FIELDS:
            return False
        for field, value in step.items():
            if field in _AGENT_MAP_FIELD_SET and value is not None and (
                not isinstance(value, dict)
                or len(value) > 1
                or (value and agent not in value)
            ):
                return False
        opponent_species = step.get("opponent_species")
        if opponent_species is not None and agent in opponent_species:
            species = opponent_species[agent]
            if (
                not isinstance(species, list)
                or len(species) != 6
                or any(not isinstance(name, str) or not name for name in species)
            ):
                return False
            expected_species = [
                to_id_str(pokemon["species"]) for pokemon in opponent_members
            ]
            if [to_id_str(name) for name in species] != expected_species:
                return False
        public_mechanics = step.get("public_mechanics")
        if public_mechanics is not None and agent in public_mechanics:
            context = public_mechanics[agent]
            if not isinstance(context, dict) or set(context) != {"own", "opponent"}:
                return False
            for side_name in ("own", "opponent"):
                members = context[side_name]
                if not isinstance(members, list) or len(members) != 6:
                    return False
                for member in members:
                    if not isinstance(member, dict) or set(member) not in (
                        {"revealed_moves", "last_move", "protect_streak"},
                        {
                            "revealed_moves", "last_move", "protect_streak",
                            "turns_since_switch",
                        },
                        {
                            "revealed_moves", "recent_moves", "last_move",
                            "protect_streak", "turns_since_switch",
                        },
                        {
                            "revealed_moves", "recent_moves", "last_move",
                            "protect_streak", "turns_since_switch",
                            "move_actions_since_switch",
                        },
                    ):
                        return False
                    moves = member["revealed_moves"]
                    if not isinstance(moves, list) or any(
                        not isinstance(move, str) for move in moves
                    ):
                        return False
                    recent_moves = member.get("recent_moves", [])
                    if (
                        not isinstance(recent_moves, list)
                        or len(recent_moves) > 2
                        or any(not isinstance(move, str) for move in recent_moves)
                        or any(move not in moves for move in recent_moves)
                    ):
                        return False
                    action_count = member.get("move_actions_since_switch")
                    if action_count is not None and (
                        not isinstance(action_count, int)
                        or isinstance(action_count, bool)
                        or action_count < 0
                    ):
                        return False
                    last_move = member["last_move"]
                    if last_move is not None and not isinstance(last_move, str):
                        return False
                    streak = member["protect_streak"]
                    if isinstance(streak, bool) or not isinstance(streak, int) or not 0 <= streak <= 6:
                        return False
                    switch_age = member.get("turns_since_switch")
                    if switch_age is not None and (
                        isinstance(switch_age, bool)
                        or not isinstance(switch_age, int)
                        or switch_age < 0
                    ):
                        return False
    for field in ("rewards", "terminated", "truncated"):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, dict) or set(value) - {agent}
        ):
            return False
    for field in ("preview_actions", "preview_log_probs"):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, dict) or set(value) - {side}
        ):
            return False
    first_turn_action = record.get("first_turn_action_data")
    if first_turn_action is not None:
        if not isinstance(first_turn_action, dict):
            return False
        first_turn_fields = set(first_turn_action)
        if not (
            first_turn_fields == _FIRST_TURN_ACTION_FIELDS
            or first_turn_fields == _FIRST_TURN_ACTION_FIELDS | _FIRST_TURN_FEATURE_FIELDS
        ):
            return False
        action = first_turn_action["action"]
        observation = first_turn_action["observation"]
        legal_mask = first_turn_action["legal_action_mask"]
        recorded_mask = first_turn_action["recorded_action_mask"]
        if (
            not isinstance(action, list)
            or len(action) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in action)
            or not isinstance(observation, list)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in observation)
            or not isinstance(legal_mask, list)
            or not isinstance(recorded_mask, list)
            or len(legal_mask) != len(recorded_mask)
            or any(value not in (0, 1) for value in legal_mask + recorded_mask)
            or first_turn_action["source"] not in {"policy", "max_damage", "counterfactual"}
            or not _valid_native_action_features(
                first_turn_action.get("native_action_features")
            )
            or not _valid_public_context_features(
                first_turn_action.get("public_context_features")
            )
        ):
            return False
    return True


def iter_player_views(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield perspective-safe rows for a sequence of match-level records."""
    for record in records:
        yield from project_episode_to_player_views(record)
