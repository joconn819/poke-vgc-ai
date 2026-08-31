"""Focused tests for team-preview action encoding and masking."""

import pytest
import torch

from src.training.team_selection import (
    ACTION_COUNT,
    TeamSelectionAction,
    TeamSelectionPolicy,
    action_mask,
    decode_action,
    encode_action,
)


def test_action_space_is_four_of_six_with_ordered_leads():
    assert ACTION_COUNT == 180
    for index in (0, 37, ACTION_COUNT - 1):
        action = decode_action(index)
        assert len(action.team) == 4
        assert len(action.leads) == 2
        assert set(action.leads) <= set(action.team)
        assert encode_action(action) == index


def test_encoding_normalizes_team_order_but_preserves_lead_order():
    assert encode_action([3, 1, 5, 0], [5, 1]) == encode_action(
        TeamSelectionAction((0, 1, 3, 5), (5, 1))
    )
    assert encode_action([0, 1, 2, 3], [0, 1]) != encode_action([0, 1, 2, 3], [1, 0])


@pytest.mark.parametrize("team, leads", [([0, 1, 2], [0, 1]), ([0, 1, 2, 3], [0, 4])])
def test_invalid_selection_is_rejected(team, leads):
    with pytest.raises(ValueError):
        encode_action(team, leads)


def test_action_mask_and_policy_mask():
    available = torch.tensor([[True, True, True, True, False, True]])
    mask = action_mask(available)
    assert mask.shape == (1, ACTION_COUNT)
    assert mask.sum() == 60  # C(5, 4) * P(4, 2)

    policy = TeamSelectionPolicy(feature_dim=3, hidden_dim=16)
    logits = policy(torch.randn(1, 6, 3), available)
    assert logits.shape == (1, ACTION_COUNT)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()
