import pytest
import torch

from src.belief.history_encoder import BattleHistoryEncoder


def test_history_encoder_returns_final_state() -> None:
    encoder = BattleHistoryEncoder(observation_dim=8, belief_dim=4, hidden_dim=12)
    result = encoder(torch.zeros(2, 3, 8), torch.ones(2, 3, 4))
    assert result.shape == (2, 12)


def test_history_encoder_defaults_beliefs_to_zero() -> None:
    encoder = BattleHistoryEncoder(observation_dim=8, belief_dim=4, hidden_dim=12)
    assert encoder(torch.zeros(2, 3, 8)).shape == (2, 12)


def test_history_encoder_rejects_misaligned_inputs() -> None:
    encoder = BattleHistoryEncoder(observation_dim=8, belief_dim=4)
    with pytest.raises(ValueError):
        encoder(torch.zeros(2, 8))
