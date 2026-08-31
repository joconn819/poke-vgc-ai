"""Tests for the supervised training loop."""

from pathlib import Path

import torch

from src.models.policy import TransformerPolicy
from src.training.data import SyntheticBattleDataset, build_dataloader
from src.training.synthetic_battles import BattleRecord, TrajectoryStep
from src.training.train import SupervisedTrainingConfig, SupervisedTrainer, load_checkpoint
from src.utils.config import REGULATION_SV2024_1


def _team() -> list[dict]:
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


def _record(action: int, winner: str) -> BattleRecord:
    trajectory = [
        TrajectoryStep(
            state=torch.full((512,), 0.25).numpy(),
            legal_actions_mask=torch.tensor([1] * 12 + [0] * 114, dtype=torch.uint8).numpy(),
            action=action,
            reward=1.0 if winner == "player" else -1.0,
            done=True,
        )
    ]
    return BattleRecord(
        player_team=_team(),
        opponent_team=_team(),
        winner=winner,
        trajectory=trajectory,
    )


def _loader() -> tuple:
    dataset = SyntheticBattleDataset.from_battle_records(
        [_record(1, "player"), _record(2, "opponent"), _record(3, "player"), _record(4, "opponent")],
        regulation_config=REGULATION_SV2024_1,
    )
    train_loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    val_loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    return dataset, train_loader, val_loader


class TestSupervisedTrainer:
    """Tests for training and checkpoint flow."""

    def test_train_epoch_returns_metrics(self, tmp_path: Path) -> None:
        _, train_loader, val_loader = _loader()
        model = TransformerPolicy(
            embedding_dim=32,
            num_transformer_layers=2,
            num_attention_heads=2,
            ff_dim=128,
        )
        trainer = SupervisedTrainer(
            model=model,
            config=SupervisedTrainingConfig(
                epochs=1,
                learning_rate=1e-3,
                checkpoint_dir=tmp_path,
            ),
        )

        metrics = trainer.train(train_loader, val_loader)

        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert "val_action_accuracy" in metrics
        assert (tmp_path / "latest.pt").exists()
        assert (tmp_path / "best.pt").exists()

    def test_checkpoint_round_trip_restores_model_state(self, tmp_path: Path) -> None:
        _, train_loader, val_loader = _loader()
        model = TransformerPolicy(
            embedding_dim=32,
            num_transformer_layers=2,
            num_attention_heads=2,
            ff_dim=128,
        )
        trainer = SupervisedTrainer(
            model=model,
            config=SupervisedTrainingConfig(
                epochs=1,
                learning_rate=1e-3,
                checkpoint_dir=tmp_path,
            ),
        )
        trainer.train(train_loader, val_loader)

        restored_model = TransformerPolicy(
            embedding_dim=32,
            num_transformer_layers=2,
            num_attention_heads=2,
            ff_dim=128,
        )
        checkpoint = load_checkpoint(tmp_path / "best.pt", restored_model)

        assert checkpoint["metrics"]["val_loss"] >= 0.0
        batch = next(iter(val_loader))
        with torch.no_grad():
            output = restored_model(
                pokemon_ids=batch["pokemon_ids"],
                move_ids=batch["move_ids"],
                item_ids=batch["item_ids"],
                ability_ids=batch["ability_ids"],
                team_mask=batch["team_mask"],
                field_features=batch["field_features"],
                hp_fractions=batch["hp_fractions"],
                status=batch["status"],
            )
        assert output.action_logits.shape[0] == 2

    def test_resume_training_continues_from_latest_epoch(self, tmp_path: Path) -> None:
        _, train_loader, val_loader = _loader()
        first_model = TransformerPolicy(
            embedding_dim=32,
            num_transformer_layers=2,
            num_attention_heads=2,
            ff_dim=128,
        )
        first_trainer = SupervisedTrainer(
            model=first_model,
            config=SupervisedTrainingConfig(
                epochs=1,
                learning_rate=1e-3,
                checkpoint_dir=tmp_path,
            ),
        )
        first_metrics = first_trainer.train(train_loader, val_loader)
        assert first_metrics["epoch"] == 1.0

        resumed_model = TransformerPolicy(
            embedding_dim=32,
            num_transformer_layers=2,
            num_attention_heads=2,
            ff_dim=128,
        )
        resumed_trainer = SupervisedTrainer(
            model=resumed_model,
            config=SupervisedTrainingConfig(
                epochs=3,
                learning_rate=1e-3,
                checkpoint_dir=tmp_path,
            ),
        )

        resume_metrics = resumed_trainer.train(
            train_loader,
            val_loader,
            resume_from=tmp_path / "latest.pt",
        )

        assert resume_metrics["epoch"] == 3.0
        checkpoint = load_checkpoint(tmp_path / "latest.pt", resumed_model)
        assert checkpoint["epoch"] == 3
