"""Tests for policy loss functions."""

import pytest
import torch
from src.models.losses import PolicyLoss


class TestPolicyLoss:
    """Test policy loss computation."""

    @pytest.fixture
    def loss_fn(self):
        """Create a policy loss function."""
        return PolicyLoss(
            action_weight=1.0,
            win_weight=0.1,
            opponent_moves_weight=0.1,
            opponent_items_weight=0.1,
            battle_length_weight=0.1,
        )

    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        batch_size = 4
        team_size = 6

        return {
            "action_logits": torch.randn(batch_size, 126),
            "win_probability": torch.sigmoid(torch.randn(batch_size, 1)),
            "opponent_moves": torch.randn(batch_size, team_size, 4, 656),
            "opponent_items": torch.randn(batch_size, team_size, 376),
            "battle_length": torch.randn(batch_size, 50),
        }

    @pytest.fixture
    def targets(self):
        """Create sample targets."""
        batch_size = 4
        team_size = 6

        return {
            "action_indices": torch.randint(0, 126, (batch_size,)),
            "win_target": torch.randint(0, 2, (batch_size, 1)).float(),
            "opponent_move_indices": torch.randint(0, 656, (batch_size, team_size, 4)),
            "opponent_item_indices": torch.randint(0, 376, (batch_size, team_size)),
            "battle_length_targets": torch.randint(0, 50, (batch_size,)),
        }

    def test_loss_computation(self, loss_fn, predictions, targets):
        """Test that loss can be computed."""
        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_action_loss_component(self, loss_fn, predictions, targets):
        """Test action loss is computed correctly."""
        with torch.no_grad():
            action_loss = torch.nn.functional.cross_entropy(
                predictions["action_logits"], targets["action_indices"]
            )

        full_loss = loss_fn(predictions, targets)

        assert action_loss < full_loss  # Full loss should be larger with auxiliary tasks

    def test_auxiliary_loss_weighting(self, loss_fn, predictions, targets):
        """Test auxiliary loss weights are applied."""
        loss_high_aux = PolicyLoss(
            action_weight=1.0,
            win_weight=1.0,
            opponent_moves_weight=1.0,
            opponent_items_weight=1.0,
            battle_length_weight=1.0,
        )(predictions, targets)

        loss_low_aux = PolicyLoss(
            action_weight=1.0,
            win_weight=0.01,
            opponent_moves_weight=0.01,
            opponent_items_weight=0.01,
            battle_length_weight=0.01,
        )(predictions, targets)

        assert loss_high_aux > loss_low_aux

    def test_masked_action_loss(self, loss_fn, predictions, targets):
        """Test that masked actions are handled correctly."""
        batch_size = predictions["action_logits"].shape[0]
        action_mask = torch.ones(batch_size, 126, dtype=torch.bool)
        action_mask[:, 100:126] = False  # Mask last 26 actions

        loss = loss_fn(predictions, targets, action_mask=action_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_loss_zero_with_perfect_predictions(self, loss_fn, predictions, targets):
        """Test loss approaches zero with perfect predictions."""
        batch_size = 4
        team_size = 6

        perfect_predictions = {
            "action_logits": torch.zeros(batch_size, 126),
            "win_probability": targets["win_target"].float(),
            "opponent_moves": torch.zeros(batch_size, team_size, 4, 656),
            "opponent_items": torch.zeros(batch_size, team_size, 376),
            "battle_length": torch.zeros(batch_size, 50),
        }

        for i in range(batch_size):
            perfect_predictions["action_logits"][i, targets["action_indices"][i]] = 10.0
            for j in range(team_size):
                for k in range(4):
                    perfect_predictions["opponent_moves"][
                        i, j, k, targets["opponent_move_indices"][i, j, k]
                    ] = 10.0
                perfect_predictions["opponent_items"][
                    i, j, targets["opponent_item_indices"][i, j]
                ] = 10.0
            perfect_predictions["battle_length"][i, targets["battle_length_targets"][i]] = 10.0

        loss = loss_fn(perfect_predictions, targets)

        assert loss.item() < 0.5  # Loss should be very small

    def test_loss_backward_pass(self, loss_fn, predictions, targets):
        """Test that loss is differentiable."""
        predictions = {k: v.requires_grad_(True) for k, v in predictions.items()}
        loss = loss_fn(predictions, targets)
        loss.backward()

        for v in predictions.values():
            assert v.grad is not None

    def test_batch_loss_consistency(self, loss_fn, predictions, targets):
        """Test loss is consistent across batches."""
        batch_size = predictions["action_logits"].shape[0]

        full_loss = loss_fn(predictions, targets)

        individual_losses = []
        for i in range(batch_size):
            single_pred = {
                k: v[i : i + 1] if isinstance(v, torch.Tensor) else v
                for k, v in predictions.items()
            }
            single_target = {
                k: v[i : i + 1] if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()
            }
            individual_losses.append(loss_fn(single_pred, single_target).item())

        expected_loss = sum(individual_losses) / batch_size
        assert abs(full_loss.item() - expected_loss) < 0.01


class TestPolicyLossEdgeCases:
    """Test edge cases in policy loss."""

    def test_empty_batch(self):
        """Test loss handles empty batch gracefully."""
        loss_fn = PolicyLoss()

        predictions = {
            "action_logits": torch.randn(0, 126),
            "win_probability": torch.randn(0, 1),
            "opponent_moves": torch.randn(0, 6, 4, 656),
            "opponent_items": torch.randn(0, 6, 376),
            "battle_length": torch.randn(0, 50),
        }

        targets = {
            "action_indices": torch.randint(0, 126, (0,)),
            "win_target": torch.randint(0, 2, (0, 1)).float(),
            "opponent_move_indices": torch.randint(0, 656, (0, 6, 4)),
            "opponent_item_indices": torch.randint(0, 376, (0, 6)),
            "battle_length_targets": torch.randint(0, 50, (0,)),
        }

        with pytest.raises((RuntimeError, ValueError)):
            loss_fn(predictions, targets)

    def test_all_masked_actions(self):
        """Test loss with all actions masked."""
        loss_fn = PolicyLoss()
        batch_size = 4

        predictions = {
            "action_logits": torch.randn(batch_size, 126),
            "win_probability": torch.sigmoid(torch.randn(batch_size, 1)),
            "opponent_moves": torch.randn(batch_size, 6, 4, 656),
            "opponent_items": torch.randn(batch_size, 6, 376),
            "battle_length": torch.randn(batch_size, 50),
        }

        targets = {
            "action_indices": torch.randint(0, 126, (batch_size,)),
            "win_target": torch.randint(0, 2, (batch_size, 1)).float(),
            "opponent_move_indices": torch.randint(0, 656, (batch_size, 6, 4)),
            "opponent_item_indices": torch.randint(0, 376, (batch_size, 6)),
            "battle_length_targets": torch.randint(0, 50, (batch_size,)),
        }

        action_mask = torch.zeros(batch_size, 126, dtype=torch.bool)

        with pytest.raises(ValueError):
            loss_fn(predictions, targets, action_mask=action_mask)
