"""Encode public, turn-dependent mechanics facts for the action policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from poke_env.data.normalize import to_id_str

from src.env.public_battle_history import STALLING_MOVES


PUBLIC_CONTEXT_DIM = 73
_PRIOR_PATH = Path(__file__).with_name("format_move_priors.json")
with _PRIOR_PATH.open(encoding="utf-8") as _stream:
    _MOVE_PRIORS = json.load(_stream)


def prepare_public_mechanics_static(
    own_team: Sequence[dict[str, Any]], format_id: str,
) -> tuple[frozenset[str], frozenset[str], tuple[bool, ...]]:
    """Prepare team facts reused across a trajectory's public observations."""
    own_fakeout = frozenset(
        to_id_str(str(member.get("species") or member.get("name", "")))
        for member in own_team[:6]
        if any(
            to_id_str(str(move)) == "fakeout"
            for move in member.get("moves", ())
        )
    )
    learnable = frozenset(
        _MOVE_PRIORS.get("learnable_species_by_move", {}).get("fakeout", ())
    ) if format_id == _MOVE_PRIORS.get("format_id") else frozenset()
    own_fakeout_slots = tuple(
        to_id_str(str(member.get("species") or member.get("name", "")))
        in own_fakeout
        for member in own_team[:6]
    )
    return own_fakeout, learnable, own_fakeout_slots


def encode_public_mechanics(
    public_mechanics: dict[str, Any] | None,
    observation: Sequence[float] | np.ndarray,
    own_team: Sequence[dict[str, Any]],
    opponent_team: Sequence[dict[str, Any]],
    format_id: str,
    static_facts: tuple[frozenset[str], frozenset[str], tuple[bool, ...]] | None = None,
) -> np.ndarray:
    """Return public context in roster order; unknown facts stay zero/unknown.

    Layout: own/opponent Protect streaks (12), last-stall flags (12),
    switch ages (12), age-known flags (12), own active Fake Out windows (6),
    opponent learnability/possible-now/confirmed-now flags (18), and a
    context-present flag (1). No hidden opponent set field is read here.
    """
    features = np.zeros(PUBLIC_CONTEXT_DIM, dtype=np.float32)
    if not isinstance(public_mechanics, dict) or set(public_mechanics) != {"own", "opponent"}:
        return features
    own_rows = list(public_mechanics.get("own", ()))[:6]
    opponent_rows = list(public_mechanics.get("opponent", ()))[:6]
    state = np.asarray(observation, dtype=np.float32).reshape(-1)
    own_active = (
        np.maximum(state[70:76], state[76:82]) > 0.5
        if state.size >= 82 else np.zeros(6, dtype=bool)
    )
    opponent_active = (
        np.maximum(state[82:88], state[88:94]) > 0.5
        if state.size >= 94 else np.zeros(6, dtype=bool)
    )
    for side_offset, side_rows in ((0, own_rows), (6, opponent_rows)):
        for slot in range(6):
            row = side_rows[slot] if slot < len(side_rows) else {}
            if not isinstance(row, dict):
                continue
            index = side_offset + slot
            protect_streak = float(row.get("protect_streak", 0))
            features[index] = min(max(protect_streak, 0.0), 6.0) / 6.0
            last_move = to_id_str(str(row.get("last_move") or ""))
            features[12 + index] = float(last_move in STALLING_MOVES)
            age = row.get("turns_since_switch")
            if isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 0:
                features[24 + index] = min(float(age), 6.0) / 6.0
                features[36 + index] = 1.0

    _, learnable, own_fakeout_slots = (
        static_facts
        if static_facts is not None
        else prepare_public_mechanics_static(own_team, format_id)
    )
    for index, row in enumerate(own_rows):
        age = row.get("turns_since_switch") if isinstance(row, dict) else None
        actions = row.get("move_actions_since_switch") if isinstance(row, dict) else None
        in_window = (
            own_active[index]
            and isinstance(age, (int, float))
            and not isinstance(age, bool)
            and 0 <= age <= 1
        )
        no_prior_action = (
            actions == 0 if isinstance(actions, int) and not isinstance(actions, bool)
            else True
        )
        features[48 + index] = float(
            in_window
            and no_prior_action
            and index < len(own_fakeout_slots)
            and own_fakeout_slots[index]
        )

    for index, row in enumerate(opponent_rows):
        member = opponent_team[index] if index < len(opponent_team) else {}
        species = to_id_str(str(member.get("species") or member.get("name", "")))
        can_learn = species in learnable
        revealed = row.get("revealed_moves", ()) if isinstance(row, dict) else ()
        confirmed = any(to_id_str(str(move)) == "fakeout" for move in revealed)
        age = row.get("turns_since_switch") if isinstance(row, dict) else None
        actions = row.get("move_actions_since_switch") if isinstance(row, dict) else None
        in_window = (
            opponent_active[index]
            and isinstance(age, (int, float))
            and not isinstance(age, bool)
            and 0 <= age <= 1
        )
        no_prior_action = (
            actions == 0 if isinstance(actions, int) and not isinstance(actions, bool)
            else True
        )
        still_possible = confirmed or (can_learn and len(set(revealed)) < 4)
        features[54 + index] = float(can_learn)
        features[60 + index] = float(in_window and no_prior_action and still_possible)
        features[66 + index] = float(in_window and no_prior_action and confirmed)
    features[72] = 1.0
    return features
