"""Tests for self-play rollout generation and the PPO self-play trainer."""

from pathlib import Path

import torch

from src.models.policy import TransformerPolicy
from src.training.self_play import OpponentPool, PolicyAgent
from src.training.self_play_train import (
    RolloutBatch,
    SelfPlayConfig,
    SelfPlayTrainer,
    generate_rollouts,
)
from src.utils.config import REGULATION_CHAMPIONS_MB
from src.utils.team_pool import TeamPool


def _team_pool() -> TeamPool:
    return TeamPool(
        teams_file="data/teams/champions_mb.json",
        regulation=REGULATION_CHAMPIONS_MB,
    )


class TestGenerateRollouts:
    def test_generates_requested_number_of_episodes(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        pool = _team_pool()
        opponent_pool = OpponentPool(heuristic_agent_types=["random", "max_damage"], seed=0)

        batch = generate_rollouts(
            model=model,
            team_pool=pool,
            opponent_pool=opponent_pool,
            num_episodes=2,
            max_steps=5,
            seed=0,
        )

        assert isinstance(batch, RolloutBatch)
        assert batch.rewards.shape[0] == 2
        assert batch.rewards.ndim == 2
        assert batch.action_mask.shape[-1] == 126

    def test_terminal_reward_matches_winner(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        pool = _team_pool()
        opponent_pool = OpponentPool(heuristic_agent_types=["random"], seed=0)

        batch = generate_rollouts(
            model=model,
            team_pool=pool,
            opponent_pool=opponent_pool,
            num_episodes=3,
            max_steps=5,
            seed=1,
        )

        # Every episode should end with a nonzero terminal reward (+1/-1) since
        # BattleGenerator forces a decision at max_steps.
        for episode_index in range(batch.rewards.shape[0]):
            mask = batch.step_mask[episode_index]
            last_step = int(mask.sum().item()) - 1
            assert batch.rewards[episode_index, last_step].item() in (1.0, -1.0)


class TestSelfPlayTrainer:
    def test_train_step_returns_loss_metrics(self, tmp_path: Path) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        config = SelfPlayConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
            episodes_per_update=2,
            max_steps_per_episode=5,
            epochs_per_update=1,
        )
        trainer = SelfPlayTrainer(model=model, config=config, team_pool=_team_pool())

        metrics = trainer.train_iteration(iteration=0)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "mean_reward" in metrics

    def test_checkpoint_is_saved_after_iteration(self, tmp_path: Path) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        config = SelfPlayConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
            episodes_per_update=2,
            max_steps_per_episode=5,
            epochs_per_update=1,
        )
        trainer = SelfPlayTrainer(model=model, config=config, team_pool=_team_pool())

        trainer.train_iteration(iteration=0)

        assert (tmp_path / "ckpt" / "latest.pt").exists()

    def test_resume_from_supervised_checkpoint(self, tmp_path: Path) -> None:
        from src.training.train import SupervisedTrainer, SupervisedTrainingConfig

        base_model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        sup_config = SupervisedTrainingConfig(
            checkpoint_dir=str(tmp_path / "sup_ckpt"),
            epochs=1,
            device="cpu",
        )
        sup_trainer = SupervisedTrainer(model=base_model, config=sup_config)
        sup_checkpoint_path = tmp_path / "sup_ckpt" / "seed.pt"
        sup_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        sup_trainer._save_checkpoint(
            sup_checkpoint_path, epoch=0, metrics={}, best_metrics={}, best_val_loss=0.0
        )

        rl_model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        config = SelfPlayConfig(
            checkpoint_dir=str(tmp_path / "rl_ckpt"),
            device="cpu",
            episodes_per_update=2,
            max_steps_per_episode=5,
            epochs_per_update=1,
            init_from_checkpoint=str(sup_checkpoint_path),
        )
        trainer = SelfPlayTrainer(model=rl_model, config=config, team_pool=_team_pool())

        for name, param in rl_model.named_parameters():
            base_param = dict(base_model.named_parameters())[name]
            assert torch.allclose(param, base_param)
