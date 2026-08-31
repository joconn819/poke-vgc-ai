"""Tests for supervised training dataset construction."""

import torch
from torch.utils.data import DataLoader

from src.training.data import SyntheticBattleDataset, build_dataloader
from src.training.synthetic_battles import BattleRecord, TrajectoryStep
from src.utils.config import REGULATION_SV2024_1


def _sample_team() -> list[dict]:
    return [
        {
            "name": "pikachu",
            "item": "choice-band",
            "ability": "static",
            "moves": ["earthquake", "surf", "thunderbolt", "ice-beam"],
        },
        {
            "name": "charizard",
            "item": "life-orb",
            "ability": "pressure",
            "moves": ["heat-wave", "focus-blast", "shadow-ball", "protect"],
        },
        {
            "name": "dragonite",
            "item": "assault-vest",
            "ability": "pressure",
            "moves": ["earthquake", "outrage", "ice-beam", "protect"],
        },
        {
            "name": "garchomp",
            "item": "choice-scarf",
            "ability": "rough-skin",
            "moves": ["earthquake", "stone-edge", "outrage", "protect"],
        },
        {
            "name": "landorus",
            "item": "choice-specs",
            "ability": "intimidate",
            "moves": ["earth-power", "stone-edge", "focus-blast", "protect"],
        },
        {
            "name": "tornadus",
            "item": "heavy-duty-boots",
            "ability": "pressure",
            "moves": ["thunderbolt", "heat-wave", "shadow-ball", "protect"],
        },
    ]


def _sample_record(winner: str = "player") -> BattleRecord:
    team = _sample_team()
    trajectory = [
        TrajectoryStep(
            state=torch.linspace(0.0, 1.0, 512).numpy(),
            legal_actions_mask=torch.tensor([1] * 10 + [0] * 116, dtype=torch.uint8).numpy(),
            action=2,
            reward=0.0,
            done=False,
        ),
        TrajectoryStep(
            state=torch.linspace(1.0, 0.0, 512).numpy(),
            legal_actions_mask=torch.tensor([1] * 8 + [0] * 118, dtype=torch.uint8).numpy(),
            action=4,
            reward=1.0 if winner == "player" else -1.0,
            done=True,
        ),
    ]
    return BattleRecord(
        player_team=team,
        opponent_team=team,
        winner=winner,
        trajectory=trajectory,
    )


class TestSyntheticBattleDataset:
    """Test supervised dataset creation from battle records."""

    def test_dataset_flattens_trajectory_steps(self) -> None:
        dataset = SyntheticBattleDataset.from_battle_records(
            [_sample_record("player"), _sample_record("opponent")],
            regulation_config=REGULATION_SV2024_1,
        )

        assert len(dataset) == 4

    def test_dataset_item_contains_model_inputs_and_targets(self) -> None:
        dataset = SyntheticBattleDataset.from_battle_records(
            [_sample_record()],
            regulation_config=REGULATION_SV2024_1,
        )

        sample = dataset[0]

        assert set(sample.keys()) == {
            "pokemon_ids",
            "move_ids",
            "item_ids",
            "ability_ids",
            "team_mask",
            "field_features",
            "hp_fractions",
            "status",
            "action_mask",
            "action_index",
            "win_target",
            "opponent_move_indices",
            "opponent_item_indices",
            "battle_length_target",
        }
        assert sample["pokemon_ids"].shape == (6,)
        assert sample["move_ids"].shape == (6, 4)
        assert sample["team_mask"].dtype == torch.bool
        assert sample["field_features"].shape == (32,)
        assert sample["action_mask"].shape == (126,)
        assert sample["action_index"].dtype == torch.long

    def test_dataloader_collates_batch(self) -> None:
        dataset = SyntheticBattleDataset.from_battle_records(
            [_sample_record(), _sample_record()],
            regulation_config=REGULATION_SV2024_1,
        )

        loader = build_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch["pokemon_ids"].shape == (2, 6)
        assert batch["move_ids"].shape == (2, 6, 4)
        assert batch["action_mask"].shape == (2, 126)
        assert batch["opponent_move_indices"].shape == (2, 6, 4)
        assert batch["battle_length_target"].shape == (2,)

    def test_dataset_statistics_capture_target_distribution(self) -> None:
        dataset = SyntheticBattleDataset.from_battle_records(
            [_sample_record("player"), _sample_record("opponent")],
            regulation_config=REGULATION_SV2024_1,
        )

        stats = dataset.get_statistics()

        assert stats["num_examples"] == 4
        assert stats["num_battles"] == 2
        assert stats["player_win_rate"] == 0.5
        assert stats["mean_battle_length"] == 2.0


def test_build_dataloader_returns_torch_dataloader() -> None:
    dataset = SyntheticBattleDataset.from_battle_records(
        [_sample_record()],
        regulation_config=REGULATION_SV2024_1,
    )

    loader = build_dataloader(dataset, batch_size=1, shuffle=False)

    assert isinstance(loader, DataLoader)
