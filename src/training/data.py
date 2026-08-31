"""Dataset utilities for supervised training on synthetic battles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.features.tensors import (
    ABILITY_INDEX_MAP,
    ITEM_INDEX_MAP,
    MOVE_INDEX_MAP,
    POKEMON_INDEX_MAP,
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
) -> Dict[str, torch.Tensor]:
    """Encode a single live observation into TransformerPolicy model inputs.

    Shared by both offline dataset construction (`_build_example`) and
    self-play rollout generation, so encoding stays consistent between
    supervised and RL training.
    """
    player_team = _team_members(player_team)
    opponent_team = _team_members(opponent_team)

    pokemon_ids = torch.zeros(6, dtype=torch.long)
    move_ids = torch.zeros((6, 4), dtype=torch.long)
    item_ids = torch.zeros(6, dtype=torch.long)
    ability_ids = torch.zeros(6, dtype=torch.long)
    team_mask = torch.zeros(6, dtype=torch.bool)
    hp_fractions = torch.ones(6, dtype=torch.float32)
    status = torch.zeros(6, dtype=torch.long)

    for idx, pokemon in enumerate(player_team[:6]):
        team_mask[idx] = True
        pokemon_ids[idx] = _species_id(pokemon.get("species") or pokemon.get("name"))
        item_ids[idx] = _item_id(pokemon.get("item"))
        ability_ids[idx] = _ability_id(pokemon.get("ability"))

        for move_index, move_name in enumerate(list(pokemon.get("moves", []))[:4]):
            move_ids[idx, move_index] = _move_id(move_name)

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    mask_tensor = torch.as_tensor(legal_actions_mask, dtype=torch.bool)
    field_features = _extract_field_features(state_tensor)
    hp_fractions[: min(4, state_tensor.shape[0])] = state_tensor[: min(4, state_tensor.shape[0])].clamp(0.0, 1.0)

    opponent_move_indices = torch.zeros((6, 4), dtype=torch.long)
    opponent_item_indices = torch.zeros(6, dtype=torch.long)
    for idx, pokemon in enumerate(opponent_team[:6]):
        opponent_item_indices[idx] = _item_id(pokemon.get("item"))
        for move_index, move_name in enumerate(list(pokemon.get("moves", []))[:4]):
            opponent_move_indices[idx, move_index] = _move_id(move_name)

    return {
        "pokemon_ids": pokemon_ids,
        "move_ids": move_ids,
        "item_ids": item_ids,
        "ability_ids": ability_ids,
        "team_mask": team_mask,
        "field_features": field_features,
        "hp_fractions": hp_fractions,
        "status": status,
        "action_mask": mask_tensor,
        "opponent_move_indices": opponent_move_indices,
        "opponent_item_indices": opponent_item_indices,
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
    features = torch.zeros(32, dtype=torch.float32)
    take = min(32, state.shape[0])
    features[:take] = state[:take]
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

