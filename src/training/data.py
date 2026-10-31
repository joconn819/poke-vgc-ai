"""Dataset utilities for supervised training on synthetic battles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.features.tensors import (
    ABILITY_INDEX_MAP,
    ITEM_INDEX_MAP,
    MOVE_INDEX_MAP,
    POKEMON_INDEX_MAP,
)
from src.features.vocab import token_id
from src.features.volatile_effects import (
    EVALUATOR_BASE_FIELD_WIDTH,
    OBSERVATION_VOLATILE_OFFSET,
    OBSERVATION_VOLATILE_WIDTH,
)
from src.training.synthetic_battles import BattleGenerator, BattleRecord
from src.utils.config import RegulationConfig
from src.utils.team_pool import TeamPool


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace(" ", "-")


def _team_members(team: Dict[str, Any] | Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(team, dict):
        return list(team.get("pokemon", []))
    return list(team)


def _team_token_signature(
    members: Sequence[Dict[str, Any]], *, include_set: bool,
) -> tuple[Any, ...]:
    rows = []
    for pokemon in members[:6]:
        row: tuple[Any, ...] = (str(pokemon.get("species") or pokemon.get("name") or ""),)
        if include_set:
            moves = pokemon.get("moves", [])
            move_names = (
                tuple(str(move) for move in list(moves)[:4])
                if isinstance(moves, (list, tuple)) else ()
            )
            row += (
                move_names,
                str(pokemon.get("item") or ""),
                str(pokemon.get("ability") or ""),
            )
        rows.append(row)
    return tuple(rows)


def _static_team_token_cache_key(
    player_team: Dict[str, Any] | Sequence[Dict[str, Any]],
    opponent_team: Dict[str, Any] | Sequence[Dict[str, Any]],
    token_vocab_version: int,
) -> tuple[Any, ...]:
    """Build the cache key for static tokens, using public opponent species only."""
    player_team = _team_members(player_team)
    opponent_team = _team_members(opponent_team)
    return (
        int(token_vocab_version),
        _team_token_signature(player_team, include_set=True),
        _team_token_signature(opponent_team, include_set=False),
    )


def _species_id(name: str | None) -> int:
    return POKEMON_INDEX_MAP.get(_normalize_name(name), 0)


def _move_id(name: str | None) -> int:
    normalized = _normalize_name(name)
    if normalized == "protect":
        normalized = "protected"
    return MOVE_INDEX_MAP.get(normalized, 0)


def _item_id(name: str | None) -> int:
    return ITEM_INDEX_MAP.get(_normalize_name(name), 0)


def _ability_id(name: str | None) -> int:
    return ABILITY_INDEX_MAP.get(_normalize_name(name), 0)


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary statistics for a supervised dataset."""

    num_examples: int
    num_battles: int
    player_win_rate: float
    mean_battle_length: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "num_examples": self.num_examples,
            "num_battles": self.num_battles,
            "player_win_rate": self.player_win_rate,
            "mean_battle_length": self.mean_battle_length,
        }


class SyntheticBattleDataset(Dataset[Dict[str, torch.Tensor]]):
    """Flattened step-level dataset built from synthetic battle trajectories."""

    def __init__(
        self,
        examples: List[Dict[str, torch.Tensor]],
        statistics: DatasetStatistics,
    ):
        self._examples = examples
        self._statistics = statistics

    @classmethod
    def from_battle_records(
        cls,
        records: Sequence[BattleRecord],
        regulation_config: RegulationConfig,
    ) -> "SyntheticBattleDataset":
        examples: List[Dict[str, torch.Tensor]] = []
        total_steps = 0
        player_wins = 0

        for record in records:
            total_steps += len(record.trajectory)
            if record.winner == "player":
                player_wins += 1

            for step in record.trajectory:
                examples.append(_build_example(record, step))

        statistics = DatasetStatistics(
            num_examples=len(examples),
            num_battles=len(records),
            player_win_rate=(player_wins / len(records)) if records else 0.0,
            mean_battle_length=(total_steps / len(records)) if records else 0.0,
        )
        _ = regulation_config
        return cls(examples=examples, statistics=statistics)

    @classmethod
    def from_generator(
        cls,
        generator: BattleGenerator,
        team_pool: TeamPool,
        num_battles: int,
        regulation_config: RegulationConfig,
    ) -> "SyntheticBattleDataset":
        records = generate_battle_records(generator, team_pool, num_battles)
        return cls.from_battle_records(records, regulation_config=regulation_config)

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticBattleDataset":
        payload = torch.load(Path(path), map_location="cpu")
        statistics = DatasetStatistics(**payload["statistics"])
        return cls(examples=payload["examples"], statistics=statistics)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "examples": self._examples,
                "statistics": self._statistics.to_dict(),
            },
            target,
        )

    def get_statistics(self) -> Dict[str, float | int]:
        return self._statistics.to_dict()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._examples[index]


def build_model_inputs(
    player_team: Sequence[Dict[str, Any]],
    opponent_team: Sequence[Dict[str, Any]],
    state: Any,
    legal_actions_mask: Any,
    token_vocab_version: int = 1,
    _static_team_cache: Dict[tuple[Any, ...], Dict[str, torch.Tensor]] | None = None,
    _clone_static_cache: bool = True,
    _static_team_cache_key: tuple[Any, ...] | None = None,
) -> Dict[str, torch.Tensor]:
    """Encode a single live observation into TransformerPolicy model inputs.

    Shared by both offline dataset construction (`_build_example`) and
    self-play rollout generation, so encoding stays consistent between
    supervised and RL training.
    """
    player_team = _team_members(player_team)
    opponent_team = _team_members(opponent_team)

    # Team identity/set tokens are constant across every state in one match.
    # Reuse their encoded form during offline trajectory conversion; clone on
    # return so callers retain the prior independent-tensor behavior.
    cache_key = _static_team_cache_key or _static_team_token_cache_key(
        player_team, opponent_team, token_vocab_version
    )
    static = _static_team_cache.get(cache_key) if _static_team_cache is not None else None
    if static is None:
        pokemon_ids = torch.zeros(6, dtype=torch.long)
        move_ids = torch.zeros((6, 4), dtype=torch.long)
        item_ids = torch.zeros(6, dtype=torch.long)
        ability_ids = torch.zeros(6, dtype=torch.long)
        team_mask = torch.zeros(6, dtype=torch.bool)
        opponent_pokemon_ids = torch.zeros(6, dtype=torch.long)
        opponent_team_mask = torch.zeros(6, dtype=torch.bool)

        for idx, pokemon in enumerate(player_team[:6]):
            team_mask[idx] = True
            pokemon_ids[idx] = token_id(
                "pokemon", pokemon.get("species") or pokemon.get("name"),
                vocab_version=token_vocab_version,
            )
            item_ids[idx] = token_id("item", pokemon.get("item"), vocab_version=token_vocab_version)
            ability_ids[idx] = token_id(
                "ability", pokemon.get("ability"), vocab_version=token_vocab_version
            )
            for move_index, move_name in enumerate(list(pokemon.get("moves", []))[:4]):
                move_ids[idx, move_index] = token_id(
                    "move", move_name, vocab_version=token_vocab_version
                )

        for idx, pokemon in enumerate(opponent_team[:6]):
            opponent_team_mask[idx] = True
            # Team preview reveals species only. Never copy hidden opponent
            # moves, item, or ability into model inputs.
            opponent_pokemon_ids[idx] = token_id(
                "pokemon", pokemon.get("species") or pokemon.get("name"),
                vocab_version=token_vocab_version,
            )
        static = {
            "pokemon_ids": pokemon_ids,
            "move_ids": move_ids,
            "item_ids": item_ids,
            "ability_ids": ability_ids,
            "team_mask": team_mask,
            "opponent_pokemon_ids": opponent_pokemon_ids,
            "opponent_team_mask": opponent_team_mask,
        }
        if _static_team_cache is not None:
            _static_team_cache[cache_key] = static
    if _clone_static_cache:
        static = {key: value.clone() for key, value in static.items()}

    hp_fractions = torch.ones(6, dtype=torch.float32)
    status = torch.zeros(6, dtype=torch.long)

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    mask_tensor = torch.as_tensor(legal_actions_mask, dtype=torch.bool)
    field_features = _extract_field_features(state_tensor)
    if state_tensor.numel() >= 7:
        hp_fractions.copy_(state_tensor[1:7].clamp(0.0, 1.0))
    else:
        hp_fractions[: min(6, state_tensor.shape[0])] = state_tensor[:6].clamp(0.0, 1.0)
    if state_tensor.numel() >= 70:
        status.copy_((state_tensor[64:70] * 6).round().long().clamp(0, 7))

    opponent_move_indices = torch.zeros((6, 4), dtype=torch.long)
    opponent_recent_move_ids = torch.zeros((6, 2), dtype=torch.long)
    opponent_item_indices = torch.zeros(6, dtype=torch.long)
    opponent_ability_indices = torch.zeros(6, dtype=torch.long)
    opponent_hp_fractions = torch.ones(6, dtype=torch.float32)
    opponent_status = torch.zeros(6, dtype=torch.long)
    if state_tensor.numel() >= 13:
        opponent_hp_fractions.copy_(state_tensor[7:13].clamp(0.0, 1.0))
    if state_tensor.numel() >= 100:
        opponent_status.copy_((state_tensor[94:100] * 6).round().long().clamp(0, 7))

    return {
        **static,
        "field_features": field_features,
        "hp_fractions": hp_fractions,
        "status": status,
        "opponent_move_ids": opponent_move_indices,
        "opponent_recent_move_ids": opponent_recent_move_ids,
        "opponent_item_ids": opponent_item_indices,
        "opponent_ability_ids": opponent_ability_indices,
        "opponent_hp_fractions": opponent_hp_fractions,
        "opponent_status": opponent_status,
        # Compatibility targets for the legacy auxiliary prediction losses.
        "opponent_move_indices": opponent_move_indices,
        "opponent_item_indices": opponent_item_indices,
        "action_mask": mask_tensor,
    }


def _build_example(record: BattleRecord, step: Any) -> Dict[str, torch.Tensor]:
    battle_length_target = min(max(len(record.trajectory) - 1, 0), 49)

    inputs = build_model_inputs(
        player_team=record.player_team,
        opponent_team=record.opponent_team,
        state=step.state,
        legal_actions_mask=step.legal_actions_mask,
    )

    inputs["action_index"] = torch.tensor(step.action, dtype=torch.long)
    inputs["win_target"] = torch.tensor(
        [1.0 if record.winner == "player" else 0.0], dtype=torch.float32
    )
    inputs["battle_length_target"] = torch.tensor(battle_length_target, dtype=torch.long)
    return inputs


def _extract_field_features(state: torch.Tensor) -> torch.Tensor:
    # Preserve the 64 base/tactical values and 24 active-lineup indicators,
    # then append three volatile features for each active slot: Perish Song
    # present, normalized countdown, and trapped. The source state stores
    # volatile triples for six own and six opposing team members at 106:142;
    # use the active-lineup one-hots so bench volatile state does not get
    # confused with the current decision's active Pokemon.
    if state.device.type == "cpu" and state.dtype == torch.float32 and not state.requires_grad:
        # Observations reach this encoder as CPU float32 tensors. NumPy can
        # gather the four active volatile rows without per-slot tensor .item()
        # calls, which otherwise add Python dispatch to every recorded state.
        values = state.numpy()
        features_array = np.zeros(EVALUATOR_BASE_FIELD_WIDTH, dtype=np.float32)
        take = min(64, values.shape[0])
        features_array[:take] = values[:take]
        if values.size >= 94:
            features_array[64:88] = values[70:94]
        if values.size >= 142 and values.size >= 94:
            active = values[70:94].reshape(4, 6)
            volatile = values[106:142].reshape(12, 3)
            active_max = active.max(axis=1)
            roster_indices = active.argmax(axis=1)
            roster_indices[2:] += 6
            features_array[88:100] = np.where(
                (active_max > 0.5)[:, None],
                volatile[roster_indices],
                0.0,
            ).reshape(-1)
        volatile_end = OBSERVATION_VOLATILE_OFFSET + OBSERVATION_VOLATILE_WIDTH
        if values.size >= volatile_end:
            features_array[100:] = np.clip(
                values[OBSERVATION_VOLATILE_OFFSET:volatile_end], 0.0, 1.0
            )
        return torch.from_numpy(features_array)

    # Keep the tensor implementation for device tensors, non-float32 inputs,
    # and autograd callers.
    features = torch.zeros(EVALUATOR_BASE_FIELD_WIDTH, dtype=torch.float32)
    take = min(64, state.shape[0])
    features[:take] = state[:take]
    if state.numel() >= 94:
        features[64:88] = state[70:94]
    if state.numel() >= 142 and state.numel() >= 94:
        active = state[70:94].reshape(4, 6)
        volatile = state[106:142].reshape(12, 3)
        for active_slot, row in enumerate(active):
            roster_index = int(row.argmax().item()) if row.max().item() > 0.5 else None
            if roster_index is None:
                continue
            member_index = roster_index + (6 if active_slot >= 2 else 0)
            feature_offset = 88 + active_slot * 3
            features[feature_offset:feature_offset + 3] = volatile[member_index]
    volatile_end = OBSERVATION_VOLATILE_OFFSET + OBSERVATION_VOLATILE_WIDTH
    if state.numel() >= volatile_end:
        features[100:] = state[OBSERVATION_VOLATILE_OFFSET:volatile_end].clamp(0.0, 1.0)
    return features


def _collate_examples(examples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    return {key: torch.stack([example[key] for example in examples], dim=0) for key in examples[0]}


def build_dataloader(
    dataset: Dataset[Dict[str, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_examples)


def generate_battle_records(
    generator: BattleGenerator,
    team_pool: TeamPool,
    num_battles: int,
) -> List[BattleRecord]:
    records: List[BattleRecord] = []
    for _ in range(num_battles):
        player_team = _team_members(team_pool.sample_team())
        opponent_team = _team_members(team_pool.sample_team())
        records.append(
            generator.generate_battle(
                player_team=player_team,
                opponent_team=opponent_team,
            )
        )
    return records
