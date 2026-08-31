"""Tests for the self-play PolicyAgent wrapper and opponent pool."""

from pathlib import Path

import numpy as np
import torch

from src.models.policy import TransformerPolicy
from src.training.self_play import OpponentPool, PolicyAgent
from src.training.train import SupervisedTrainer, SupervisedTrainingConfig


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
    ]


class TestPolicyAgent:
    def test_predict_returns_legal_action(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        agent = PolicyAgent(model=model, player_team=_team(), opponent_team=_team())

        observation = np.full(512, 0.25, dtype=np.float32)
        legal_actions_mask = np.zeros(126, dtype=np.uint8)
        legal_actions_mask[[3, 7, 20]] = 1

        action = agent.predict(observation, legal_actions_mask)

        assert legal_actions_mask[action] == 1

    def test_predict_with_log_prob_returns_log_prob_of_selected_action(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        agent = PolicyAgent(model=model, player_team=_team(), opponent_team=_team())

        observation = np.full(512, 0.25, dtype=np.float32)
        legal_actions_mask = np.zeros(126, dtype=np.uint8)
        legal_actions_mask[[3, 7, 20]] = 1

        action, log_prob, value = agent.predict_with_value(observation, legal_actions_mask)

        assert legal_actions_mask[action] == 1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_predict_native_supports_variable_doubles_mask(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        agent = PolicyAgent(model=model, player_team=_team(), opponent_team=_team())

        observation = np.full(512, 0.25, dtype=np.float32)
        legal_actions_mask = np.zeros(214, dtype=np.uint8)
        legal_actions_mask[[2, 7, 107 + 4, 107 + 12]] = 1

        actions, log_prob, value = agent.predict_native_with_value(
            observation, legal_actions_mask
        )

        assert actions.shape == (2,)
        assert legal_actions_mask[actions[0]] == 1
        assert legal_actions_mask[107 + actions[1]] == 1
        assert isinstance(log_prob, float)
        assert 0.0 <= value <= 1.0

    def test_deterministic_mode_is_argmax(self) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        model.eval()
        agent = PolicyAgent(
            model=model, player_team=_team(), opponent_team=_team(), deterministic=True
        )

        observation = np.full(512, 0.25, dtype=np.float32)
        legal_actions_mask = np.zeros(126, dtype=np.uint8)
        legal_actions_mask[[3, 7, 20]] = 1

        action_a = agent.predict(observation, legal_actions_mask)
        action_b = agent.predict(observation, legal_actions_mask)

        assert action_a == action_b


class TestOpponentPool:
    def test_sample_returns_heuristic_agent_when_empty(self) -> None:
        pool = OpponentPool(heuristic_agent_types=["random", "max_damage"], seed=0)

        agent = pool.sample()

        assert agent is not None

    def test_add_checkpoint_increases_pool_size(self, tmp_path: Path) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        config = SupervisedTrainingConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            epochs=1,
            learning_rate=1e-3,
            device="cpu",
        )
        trainer = SupervisedTrainer(model=model, config=config)
        checkpoint_path = tmp_path / "ckpt" / "seed.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer._save_checkpoint(
            checkpoint_path, epoch=0, metrics={}, best_metrics={}, best_val_loss=0.0
        )

        pool = OpponentPool(heuristic_agent_types=["random"], seed=0)
        pool.add_checkpoint(str(checkpoint_path))

        assert pool.num_checkpoints() == 1

    def test_sample_prefers_checkpoints_when_available(self, tmp_path: Path) -> None:
        model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
        config = SupervisedTrainingConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            epochs=1,
            learning_rate=1e-3,
            device="cpu",
        )
        trainer = SupervisedTrainer(model=model, config=config)
        checkpoint_path = tmp_path / "ckpt" / "seed.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer._save_checkpoint(
            checkpoint_path, epoch=0, metrics={}, best_metrics={}, best_val_loss=0.0
        )

        pool = OpponentPool(
            heuristic_agent_types=["random"],
            seed=0,
            checkpoint_sample_probability=1.0,
        )
        pool.add_checkpoint(str(checkpoint_path))

        agent = pool.sample(
            player_team=_team(),
            opponent_team=_team(),
            model_factory=lambda: TransformerPolicy(
                embedding_dim=16, num_transformer_layers=1, ff_dim=32
            ),
        )

        assert isinstance(agent, PolicyAgent)
