"""Closed-sheet terminal-outcome dataset and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from poke_env.data.normalize import to_id_str
import orjson

from src.training.data import build_model_inputs
from src.training.perspective_data import is_perspective_safe_record
from src.jsonl_io import open_jsonl_binary
from src.features.vocab import token_id
from src.env.public_battle_history import STALLING_MOVES
from src.features.volatile_effects import FIELD_FEATURE_SCHEMA, OBSERVATION_VOLATILE_WIDTH


STATE_VALUE_BASE_FIELD_DIM = 100
STATE_VALUE_ACTION_COUNT = 126
STATE_VALUE_ACTION_MASK_DIM = STATE_VALUE_ACTION_COUNT * 2
STATE_VALUE_HISTORY_TURNS = 2
STATE_VALUE_HISTORY_ROW_DIM = 24 + STATE_VALUE_ACTION_MASK_DIM
STATE_VALUE_FAKEOUT_FEATURE_DIM = 18
STATE_VALUE_SCREEN_FEATURE_DIM = 6
STATE_VALUE_CONDITION_AGE_COUNT = 18
STATE_VALUE_CONDITION_AGE_FEATURE_DIM = STATE_VALUE_CONDITION_AGE_COUNT * 2
# Raw, player-visible flags for four weather classes, four terrains, Trick
# Room, Gravity, own/opponent Tailwind, and six screens.
STATE_VALUE_CONDITION_OBSERVATION_INDICES = (
    *range(52, 64), *range(100, 106),
)
# Twelve Protect streaks, twelve last-stall flags, twelve normalized switch
# ages, twelve age-known flags, three six-member Fake Out feature groups, and
# one public-context-present bit.
STATE_VALUE_MECHANICS_DIM = 48 + STATE_VALUE_FAKEOUT_FEATURE_DIM + 1
STATE_VALUE_PUBLIC_CONTEXT_INDEX = (
    STATE_VALUE_BASE_FIELD_DIM
    + STATE_VALUE_ACTION_MASK_DIM
    + STATE_VALUE_HISTORY_TURNS * STATE_VALUE_HISTORY_ROW_DIM
    + STATE_VALUE_MECHANICS_DIM
    - 1
)
STATE_VALUE_VOLATILE_FIELD_DIM = (
    STATE_VALUE_BASE_FIELD_DIM
    + STATE_VALUE_ACTION_MASK_DIM
    + STATE_VALUE_HISTORY_TURNS * STATE_VALUE_HISTORY_ROW_DIM
    + STATE_VALUE_MECHANICS_DIM
    + OBSERVATION_VOLATILE_WIDTH
)
STATE_VALUE_FIELD_DIM = (
    STATE_VALUE_VOLATILE_FIELD_DIM
    + STATE_VALUE_SCREEN_FEATURE_DIM
    + STATE_VALUE_CONDITION_AGE_FEATURE_DIM
)
STATE_VALUE_SCREEN_FIELD_DIM = (
    STATE_VALUE_VOLATILE_FIELD_DIM + STATE_VALUE_SCREEN_FEATURE_DIM
)
STATE_VALUE_FIELD_SCHEMA = (
    f"{FIELD_FEATURE_SCHEMA}+state_value_public_condition_observation_age_v1"
)

_FORMAT_MOVE_PRIORS_PATH = (
    Path(__file__).resolve().parents[1] / "features" / "format_move_priors.json"
)
with _FORMAT_MOVE_PRIORS_PATH.open(encoding="utf-8") as _prior_file:
    _FORMAT_MOVE_PRIORS = json.load(_prior_file)
_FORMAT_MOVE_PRIORS_ID = str(_FORMAT_MOVE_PRIORS.get("format_id", ""))
_LEARNABLE_FAKEOUT_BY_FORMAT = {
    _FORMAT_MOVE_PRIORS_ID: frozenset(
        to_id_str(species)
        for species in _FORMAT_MOVE_PRIORS.get("learnable_species_by_move", {}).get("fakeout", ())
    ),
}


def _numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def _prepare_state_value_history_contexts(
    prepared_observations: list[torch.Tensor | None],
    prepared_actions: list[torch.Tensor],
) -> np.ndarray:
    """Build each state's two-turn own-history block once per trajectory."""
    if len(prepared_observations) != len(prepared_actions):
        raise ValueError("prepared observation/action histories must have equal length")
    state_count = len(prepared_observations)
    observation_rows = np.zeros((state_count, 24), dtype=np.float32)
    action_rows = np.zeros((state_count, STATE_VALUE_ACTION_MASK_DIM), dtype=np.float32)
    for index, observation in enumerate(prepared_observations):
        if observation is not None:
            state = observation.numpy().reshape(-1)
            if state.size >= 94:
                observation_rows[index] = np.clip(state[70:94], 0.0, 1.0)
        for slot, action_index in enumerate(prepared_actions[index].tolist()[:2]):
            action_index = int(action_index)
            if 0 <= action_index < STATE_VALUE_ACTION_COUNT:
                action_rows[index, slot * STATE_VALUE_ACTION_COUNT + action_index] = 1.0

    history = np.zeros(
        (state_count, STATE_VALUE_HISTORY_TURNS * STATE_VALUE_HISTORY_ROW_DIM),
        dtype=np.float32,
    )
    row_width = STATE_VALUE_HISTORY_ROW_DIM
    for back in range(1, STATE_VALUE_HISTORY_TURNS + 1):
        if state_count <= back:
            break
        offset = (back - 1) * row_width
        history[back:, offset:offset + 24] = observation_rows[:-back]
        history[back:, offset + 24:offset + row_width] = action_rows[:-back]
    return history


def _state_value_context(
    trajectory: list[dict[str, Any]], index: int,
    agent: str, current_action_mask: Any,
    model_inputs: dict[str, torch.Tensor], token_vocab_version: int,
    opponent_team: Any, format_id: str,
    include_recent_opponent_moves: bool = True,
    prepared_observations: list[torch.Tensor | None] | None = None,
    prepared_actions: list[torch.Tensor] | None = None,
    prepared_condition_ages: torch.Tensor | None = None,
    prepared_history_contexts: np.ndarray | None = None,
) -> torch.Tensor:
    """Return current legal actions and two own-action/active-lineup records.

    Only actions selected by this player are included. Past active one-hots
    attach each slot action to the roster member that actually chose it, which
    lets the value model distinguish repeated Protect choices across switches.
    """
    values = np.zeros(
        STATE_VALUE_FIELD_DIM - STATE_VALUE_BASE_FIELD_DIM, dtype=np.float32
    )
    mask = _numpy_float32(current_action_mask).reshape(-1)
    mask_take = min(mask.size, STATE_VALUE_ACTION_MASK_DIM)
    values[:mask_take] = np.clip(mask[:mask_take], 0.0, 1.0)
    mechanics_offset = STATE_VALUE_ACTION_MASK_DIM + (
        STATE_VALUE_HISTORY_TURNS * STATE_VALUE_HISTORY_ROW_DIM
    )
    if prepared_history_contexts is not None:
        expected_shape = (
            len(trajectory), STATE_VALUE_HISTORY_TURNS * STATE_VALUE_HISTORY_ROW_DIM
        )
        if prepared_history_contexts.shape != expected_shape:
            raise ValueError("prepared history contexts do not match trajectory shape")
        values[STATE_VALUE_ACTION_MASK_DIM:mechanics_offset] = prepared_history_contexts[index]
    else:
        cursor = STATE_VALUE_ACTION_MASK_DIM
        for back in range(1, STATE_VALUE_HISTORY_TURNS + 1):
            previous_index = index - back
            if previous_index < 0:
                cursor += STATE_VALUE_HISTORY_ROW_DIM
                continue
            previous = trajectory[previous_index]
            observation = previous.get("observations", {}).get(agent)
            if prepared_observations is not None:
                observation = prepared_observations[previous_index]
            if observation is not None:
                state = _numpy_float32(observation).reshape(-1)
                if state.size >= 94:
                    values[cursor:cursor + 24] = np.clip(state[70:94], 0.0, 1.0)
            action = previous.get("actions", {}).get(agent, ())
            if prepared_actions is not None:
                action = prepared_actions[previous_index]
            if isinstance(action, torch.Tensor):
                action = action.tolist()
            else:
                action = np.asarray(action).reshape(-1).tolist()
            action_offset = cursor + 24
            for slot in range(min(2, len(action))):
                action_index = int(action[slot])
                if 0 <= action_index < STATE_VALUE_ACTION_COUNT:
                    values[action_offset + slot * STATE_VALUE_ACTION_COUNT + action_index] = 1.0
            cursor += STATE_VALUE_HISTORY_ROW_DIM
    public_map = trajectory[index].get("public_mechanics", {})
    mechanics = public_map.get(agent) if isinstance(public_map, dict) else None
    opponent_members = (
        opponent_team.get("pokemon", opponent_team.get("team", []))
        if isinstance(opponent_team, dict) else opponent_team
    )
    learnable_fakeout = _LEARNABLE_FAKEOUT_BY_FORMAT.get(format_id, ())
    fakeout_ages: list[int | None] = [None] * 6
    revealed_fakeout = [False] * 6
    revealed_move_counts = [0] * 6
    move_actions_since_switch: list[int | None] = [None] * 6
    if isinstance(mechanics, dict) and set(mechanics) == {"own", "opponent"}:
        values[mechanics_offset + STATE_VALUE_MECHANICS_DIM - 1] = 1.0
        members = list(mechanics["own"][:6]) + list(mechanics["opponent"][:6])
        for member_index, member in enumerate(members):
            if not isinstance(member, dict):
                continue
            streak = member.get("protect_streak", 0)
            if isinstance(streak, (int, float)):
                values[mechanics_offset + member_index] = min(max(float(streak), 0.0), 6.0) / 6.0
            last_move = str(member.get("last_move") or "")
            values[mechanics_offset + 12 + member_index] = float(
                to_id_str(last_move) in STALLING_MOVES
            )
            age = member.get("turns_since_switch")
            if isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 0:
                values[mechanics_offset + 24 + member_index] = min(float(age), 6.0) / 6.0
                values[mechanics_offset + 36 + member_index] = 1.0
            if member_index >= 6:
                opponent_index = member_index - 6
                fakeout_ages[opponent_index] = (
                    int(age) if isinstance(age, (int, float))
                    and not isinstance(age, bool) and age >= 0 else None
                )
                revealed_fakeout[opponent_index] = any(
                    to_id_str(str(move)) == "fakeout"
                    for move in member.get("revealed_moves", ())
                )
                revealed_move_counts[opponent_index] = len(set(
                    to_id_str(str(move)) for move in member.get("revealed_moves", ())
                ))
                action_count = member.get("move_actions_since_switch")
                if isinstance(action_count, int) and not isinstance(action_count, bool):
                    move_actions_since_switch[opponent_index] = max(action_count, 0)
        for pokemon_index, member in enumerate(mechanics["opponent"][:6]):
            if not isinstance(member, dict):
                continue
            if include_recent_opponent_moves:
                for recent_index, move in enumerate(list(member.get("recent_moves", ()))[:2]):
                    model_inputs["opponent_recent_move_ids"][pokemon_index, recent_index] = token_id(
                        "move", str(move), vocab_version=token_vocab_version
                    )
            revealed_moves = member.get("revealed_moves", ())
            for move_index, move in enumerate(list(revealed_moves)[:4]):
                model_inputs["opponent_move_ids"][pokemon_index, move_index] = token_id(
                    "move", str(move), vocab_version=token_vocab_version
                )

    # Species-level learnability is a legal possibility prior, not hidden-set
    # truth. The threat flags require a public switch age of zero or one, the
    # current active-roster map, and no observed move action since entry.
    # Age one can be the entrant's first move choice because switches resolve
    # during the prior turn. Legacy records without the action count retain
    # the previous age-only behavior.
    base_fields = model_inputs["field_features"]
    active_rows = _numpy_float32(base_fields[76:88].reshape(2, 6))
    active_opponents = np.max(active_rows, axis=0) > 0.5
    fakeout_offset = mechanics_offset + 48
    for pokemon_index, pokemon in enumerate(list(opponent_members or ())[:6]):
        species = (
            pokemon.get("species") or pokemon.get("name", "")
            if isinstance(pokemon, dict) else str(pokemon)
        )
        can_learn = to_id_str(str(species)) in learnable_fakeout
        can_still_have = revealed_fakeout[pokemon_index] or (
            can_learn and revealed_move_counts[pokemon_index] < 4
        )
        active_entry_window = (
            bool(active_opponents[pokemon_index])
            and fakeout_ages[pokemon_index] is not None
            and fakeout_ages[pokemon_index] <= 1
        )
        no_prior_action = move_actions_since_switch[pokemon_index] in (None, 0)
        confirmed = revealed_fakeout[pokemon_index]
        values[fakeout_offset + pokemon_index] = float(can_learn)
        values[fakeout_offset + 6 + pokemon_index] = float(
            active_entry_window and no_prior_action and can_still_have
        )
        values[fakeout_offset + 12 + pokemon_index] = float(
            active_entry_window and no_prior_action and confirmed
        )
    # Keep the pre-extension evaluator layout as an exact prefix: current
    # 100-wide state, legal choices, own decision history, then public
    # mechanics. New volatile bits are appended after that legacy prefix.
    volatile_offset = mechanics_offset + STATE_VALUE_MECHANICS_DIM
    field_features = _numpy_float32(model_inputs["field_features"].reshape(-1))
    if field_features.size >= 460:
        values[volatile_offset:volatile_offset + OBSERVATION_VOLATILE_WIDTH] = (
            field_features[100:460]
        )
    current_observation = trajectory[index].get("observations", {}).get(agent)
    if prepared_observations is not None:
        current_observation = prepared_observations[index]
    if current_observation is not None:
        current_state = _numpy_float32(current_observation).reshape(-1)
        if current_state.size >= 106:
            # ShowdownDoublesEnv stores own Reflect/Light Screen/Aurora Veil,
            # then the opponent's three screens, in observation[100:106].
            # Preserve them after the legacy 1,331-value field prefix.
            screen_offset = STATE_VALUE_VOLATILE_FIELD_DIM - STATE_VALUE_BASE_FIELD_DIM
            values[screen_offset:screen_offset + STATE_VALUE_SCREEN_FEATURE_DIM] = (
                np.clip(current_state[100:106], 0, 1)
            )
    if prepared_condition_ages is None:
        observations = prepared_observations or [
            step.get("observations", {}).get(agent) for step in trajectory
        ]
        prepared_condition_ages = _condition_age_history(observations)
    age_offset = STATE_VALUE_SCREEN_FIELD_DIM - STATE_VALUE_BASE_FIELD_DIM
    values[age_offset:age_offset + STATE_VALUE_CONDITION_AGE_FEATURE_DIM] = _numpy_float32(
        prepared_condition_ages[index]
    )
    return torch.from_numpy(values)


def _condition_age_history(
    observations: list[torch.Tensor | None],
    *,
    maximum_age: int = 8,
) -> torch.Tensor:
    """Encode consecutive observed-present age and its knownness per match.

    This is an observation-resolution lower bound, not an exact remaining
    duration: conditions may be refreshed between decisions, and hidden items
    can extend duration. A transition from observed absent to present makes
    the age clock known; conditions already active at the first observation
    retain an unknown age flag.
    """
    if not observations:
        return torch.zeros((0, STATE_VALUE_CONDITION_AGE_FEATURE_DIM), dtype=torch.float32)
    rows: list[list[float]] = []
    run_age = [0] * STATE_VALUE_CONDITION_AGE_COUNT
    age_known = [False] * STATE_VALUE_CONDITION_AGE_COUNT
    previous: list[bool] | None = None
    maximum_observation_index = max(STATE_VALUE_CONDITION_OBSERVATION_INDICES)
    for observation in observations:
        if observation is None:
            run_age = [0] * STATE_VALUE_CONDITION_AGE_COUNT
            age_known = [False] * STATE_VALUE_CONDITION_AGE_COUNT
            previous = None
            rows.append([0.0] * STATE_VALUE_CONDITION_AGE_FEATURE_DIM)
            continue
        state = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        if state.numel() <= maximum_observation_index:
            run_age = [0] * STATE_VALUE_CONDITION_AGE_COUNT
            age_known = [False] * STATE_VALUE_CONDITION_AGE_COUNT
            previous = None
            rows.append([0.0] * STATE_VALUE_CONDITION_AGE_FEATURE_DIM)
            continue
        current = [
            bool(state[observation_index] > 0.5)
            for observation_index in STATE_VALUE_CONDITION_OBSERVATION_INDICES
        ]
        for feature_index, active in enumerate(current):
            if not active:
                run_age[feature_index] = 0
                age_known[feature_index] = False
            elif previous is None:
                run_age[feature_index] = 0
                age_known[feature_index] = False
            elif not previous[feature_index]:
                run_age[feature_index] = 0
                age_known[feature_index] = True
            else:
                run_age[feature_index] = min(
                    run_age[feature_index] + 1, maximum_age
                )
        rows.append(
            [age / maximum_age for age in run_age]
            + [float(known) for known in age_known]
        )
        previous = current
    return torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class StateValueExample:
    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    team_ids: tuple[str, str]
    own_team_id: str
    game_id: str
    record_index: int = -1
    perspective_agent: str = ""
    step_index: int = -1
    policy_inputs: dict[str, torch.Tensor] | None = None


class StateValueDataset(Dataset[dict[str, torch.Tensor]]):
    """Per-state win/loss labels built only from validated player-view rows."""

    def __init__(self, examples: list[StateValueExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {**example.inputs, "win_target": example.target}

    @staticmethod
    def _team_id(team: Any) -> str:
        members = team.get("pokemon", team.get("team", [])) if isinstance(team, dict) else team
        species = sorted(
            str(pokemon.get("species") or pokemon.get("name", "")).lower()
            for pokemon in members
            if isinstance(pokemon, dict)
        )
        return hashlib.sha256("|".join(species).encode()).hexdigest()

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        token_vocab_version: int = 1,
        include_recent_opponent_moves: bool = True,
        include_policy_inputs: bool = False,
        _records_already_validated: bool = False,
    ) -> "StateValueDataset":
        examples: list[StateValueExample] = []
        static_team_cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        for record_index, record in enumerate(records):
            if (
                (not _records_already_validated and not is_perspective_safe_record(record))
                or record.get("timed_out")
            ):
                continue
            agent = record["perspective_agent"]
            side = record["perspective_team"]
            opponent_side = "player2" if side == "player1" else "player1"
            own_team = record["teams"][side]
            opponent_team = record["teams"][opponent_side]
            terminal_outcome = None
            for step in reversed(record["trajectory"]):
                components = step.get("reward_components", {}).get(agent, {})
                value = components.get("terminal_outcome")
                if isinstance(value, (int, float)) and value in (-1, 1):
                    terminal_outcome = float(value)
                    break
            if terminal_outcome is None:
                continue

            team_ids = (
                cls._team_id(record["teams"]["player1"]),
                cls._team_id(record["teams"]["player2"]),
            )
            own_team_id = team_ids[0] if side == "player1" else team_ids[1]
            pair_id = hashlib.sha256("|".join(sorted(team_ids)).encode()).hexdigest()
            game_id = (
                f"{record.get('format', 'unknown')}:{record.get('episode', record_index)}:{pair_id}"
            )
            # JSON trajectory values are reused as the current observation and
            # again in the next two states' history features. Convert each
            # observation/action once per match instead of rebuilding tensors
            # for every perspective-state example.
            prepared_observations: list[torch.Tensor | None] = []
            prepared_actions: list[torch.Tensor] = []
            for trajectory_step in record["trajectory"]:
                raw_observation = trajectory_step.get("observations", {}).get(agent)
                prepared_observations.append(
                    None if raw_observation is None
                    else torch.as_tensor(raw_observation, dtype=torch.float32)
                )
                raw_action = trajectory_step.get("actions", {}).get(agent, ())
                prepared_actions.append(
                    torch.as_tensor(raw_action, dtype=torch.long).reshape(-1)
                )
            prepared_condition_ages = _condition_age_history(prepared_observations)
            prepared_history_contexts = _prepare_state_value_history_contexts(
                prepared_observations, prepared_actions
            )
            for step_index, step in enumerate(record["trajectory"]):
                observation = prepared_observations[step_index]
                if observation is None:
                    continue
                # Rollouts snapshot the opponent species in the exact public
                # roster order used by live inference. Preserve that order
                # when rebuilding offline evaluator inputs as well.
                roster_snapshot = step.get("opponent_species", {}).get(agent)
                state_opponent_team = (
                    [{"species": species} for species in roster_snapshot]
                    if isinstance(roster_snapshot, list) and len(roster_snapshot) == 6
                    else opponent_team
                )
                action_mask = step.get("action_masks", {}).get(agent, [0] * 126)
                inputs = build_model_inputs(
                    own_team,
                    state_opponent_team,
                    observation,
                    action_mask,
                    token_vocab_version=token_vocab_version,
                    _static_team_cache=static_team_cache,
                )
                # The policy and evaluator consume the same live inputs before
                # the evaluator-only public-history context is appended. Keep
                # a shallow mapping when requested so PPO can reuse the tensor
                # features instead of rebuilding them for each rollout state.
                if include_policy_inputs:
                    policy_inputs = dict(inputs)
                    # The evaluator context writes revealed move tokens into
                    # these two tensors below. Isolate them so cached live
                    # policy inputs retain the exact pre-context observation.
                    for key in ("opponent_move_ids", "opponent_recent_move_ids"):
                        policy_inputs[key] = inputs[key].clone()
                else:
                    policy_inputs = None
                context = _state_value_context(
                    record["trajectory"],
                    step_index,
                    agent,
                    inputs["action_mask"],
                    inputs,
                    token_vocab_version,
                    state_opponent_team,
                    str(record.get("format", "")),
                    include_recent_opponent_moves,
                    prepared_observations,
                    prepared_actions,
                    prepared_condition_ages,
                    prepared_history_contexts,
                )
                inputs["field_features"] = torch.cat(
                    [inputs["field_features"][:STATE_VALUE_BASE_FIELD_DIM], context], dim=0
                )
                inputs.pop("action_mask", None)
                inputs.pop("opponent_move_indices", None)
                inputs.pop("opponent_item_indices", None)
                examples.append(
                    StateValueExample(
                        inputs=inputs,
                        target=torch.tensor([float(terminal_outcome > 0)], dtype=torch.float32),
                        team_ids=team_ids,
                        own_team_id=own_team_id,
                        game_id=game_id,
                        record_index=record_index,
                        perspective_agent=agent,
                        step_index=step_index,
                        policy_inputs=policy_inputs,
                    )
                )
        return cls(examples)


def iter_player_view_records(
    paths: Iterable[str | Path], *, validate_perspective: bool = True
) -> Iterator[dict[str, Any]]:
    """Yield player-view JSONL rows, validating the closed-sheet contract by default.

    Set ``validate_perspective=False`` only when an earlier pass has already
    validated the same immutable source files and the caller verifies their
    signatures before publishing derived data.
    """
    for path in paths:
        # Keep rows as bytes: orjson avoids the intermediate UTF-8 decode and
        # is materially faster on the large trajectory records used by the
        # out-of-core cache builder.
        with open_jsonl_binary(path) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = orjson.loads(line)
                if validate_perspective and not is_perspective_safe_record(record):
                    raise ValueError(
                        f"{path}:{line_number} is not a perspective-safe schema-v2 record"
                    )
                yield record


def read_player_view_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read valid JSONL player views; use the iterator for large corpora."""
    return list(iter_player_view_records(paths))


def split_by_heldout_teams(
    dataset: StateValueDataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 20261019,
) -> tuple[list[int], list[int], list[str]]:
    """Split whole battles and reserve complete team rosters for evaluation."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    teams = sorted({team_id for example in dataset.examples for team_id in example.team_ids})
    if len(teams) < 3:
        raise ValueError("held-out team evaluation requires at least three distinct rosters")
    # Deterministic hash ordering avoids depending on input record order.
    teams.sort(key=lambda team: hashlib.sha256(f"{seed}:{team}".encode()).digest())
    holdout_count = min(len(teams) - 2, max(1, round(len(teams) * test_fraction)))
    heldout = set(teams[:holdout_count])
    train_indices, test_indices = [], []
    for index, example in enumerate(dataset.examples):
        has_heldout_team = any(team_id in heldout for team_id in example.team_ids)
        if example.own_team_id in heldout:
            test_indices.append(index)
        elif has_heldout_team:
            # Drop the opposite perspective for a held-out match so its
            # non-held-out own roster cannot leak across the partition.
            continue
        else:
            train_indices.append(index)
    if not train_indices or not test_indices:
        raise ValueError("team split produced an empty train or held-out partition")
    return train_indices, test_indices, sorted(heldout)


def split_by_train_calibration_test_teams(
    dataset: StateValueDataset,
    *,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.15,
    seed: int = 20261019,
) -> tuple[list[int], list[int], list[int], list[str], list[str]]:
    """Partition battles by whole rosters into train, calibration, and test.

    A battle touching a held-out roster contributes only the perspective whose
    own roster belongs to that partition. Its opposite perspective is dropped,
    so the held-out roster cannot enter training through the visible preview.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    teams = {team_id for example in dataset.examples for team_id in example.team_ids}
    calibration_teams, test_teams = split_roster_ids(
        teams,
        test_fraction=test_fraction,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    train_indices, calibration_indices, test_indices = [], [], []
    for index, example in enumerate(dataset.examples):
        touches_test = any(team_id in test_teams for team_id in example.team_ids)
        touches_calibration = any(
            team_id in calibration_teams for team_id in example.team_ids
        )
        if touches_test:
            if example.own_team_id in test_teams:
                test_indices.append(index)
        elif touches_calibration:
            if example.own_team_id in calibration_teams:
                calibration_indices.append(index)
        else:
            train_indices.append(index)
    if not train_indices or not calibration_indices or not test_indices:
        raise ValueError("team split produced an empty train, calibration, or test partition")
    return (
        train_indices,
        calibration_indices,
        test_indices,
        sorted(calibration_teams),
        sorted(test_teams),
    )


def split_roster_ids(
    roster_ids: Iterable[str],
    *,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.15,
    seed: int = 20261019,
) -> tuple[set[str], set[str]]:
    """Return deterministic calibration and test roster sets."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    teams = sorted(set(roster_ids))
    if len(teams) < 5:
        raise ValueError("train/calibration/test split requires at least five distinct rosters")
    teams.sort(key=lambda team: hashlib.sha256(f"{seed}:{team}".encode()).digest())
    test_count = min(len(teams) - 3, max(1, round(len(teams) * test_fraction)))
    test_teams = set(teams[:test_count])
    remaining = teams[test_count:]
    remaining.sort(
        key=lambda team: hashlib.sha256(f"{seed}:calibration:{team}".encode()).digest()
    )
    calibration_count = min(len(remaining) - 2, max(1, round(len(remaining) * calibration_fraction)))
    calibration_teams = set(remaining[:calibration_count])
    return calibration_teams, test_teams


def calibration_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = probabilities.detach().float().cpu().reshape(-1).clamp(0.0, 1.0)
    targets = targets.detach().float().cpu().reshape(-1)
    if probabilities.numel() != targets.numel() or not probabilities.numel():
        raise ValueError("calibration requires equally sized, non-empty tensors")
    ece = 0.0
    bin_rows = []
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            win_rate = float(targets[mask].mean())
            ece += count / probabilities.numel() * abs(mean_probability - win_rate)
            bin_rows.append(
                {"low": float(low), "high": float(high), "count": count,
                 "mean_probability": mean_probability, "win_rate": win_rate}
            )
    return {
        "count": int(probabilities.numel()),
        "brier_score": float(torch.mean((probabilities - targets) ** 2)),
        "log_loss": float(torch.nn.functional.binary_cross_entropy(
            probabilities.clamp(1e-7, 1.0 - 1e-7), targets
        )),
        "accuracy": float(((probabilities >= 0.5) == (targets >= 0.5)).float().mean()),
        "expected_calibration_error": ece,
        "calibration_bins": bin_rows,
    }
