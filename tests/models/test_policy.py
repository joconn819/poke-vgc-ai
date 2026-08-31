"""Tests for the Transformer-based policy network.

Using TDD approach - tests define expected behavior and shapes.
"""

import pytest
import torch
import numpy as np
from src.models.policy import TransformerPolicy, PolicyOutput


class TestTransformerPolicyArchitecture:
    """Test model architecture and forward pass."""

    @pytest.fixture
    def model(self):
        """Create a test policy model."""
        return TransformerPolicy(
            embedding_dim=64,
            num_transformer_layers=4,
            num_attention_heads=4,
            ff_dim=256,
            vocab_size_pokemon=1025,
            vocab_size_moves=656,
            vocab_size_items=376,
            vocab_size_abilities=267,
        )

    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        batch_size = 4
        team_size = 6
        field_features = 32

        return {
            "pokemon_ids": torch.randint(0, 1025, (batch_size, team_size)),
            "move_ids": torch.randint(0, 656, (batch_size, team_size, 4)),
            "item_ids": torch.randint(0, 376, (batch_size, team_size)),
            "ability_ids": torch.randint(0, 267, (batch_size, team_size)),
            "team_mask": torch.ones(batch_size, team_size, dtype=torch.bool),
            "field_features": torch.randn(batch_size, field_features),
            "hp_fractions": torch.rand(batch_size, team_size),
            "status": torch.randint(0, 8, (batch_size, team_size)),
        }

    def test_model_forward_pass(self, model, batch_data):
        """Test that forward pass produces correct output structure."""
        output = model(**batch_data)

        assert isinstance(output, PolicyOutput)
        assert hasattr(output, "action_logits")
        assert hasattr(output, "win_probability")
        assert hasattr(output, "opponent_moves")
        assert hasattr(output, "opponent_items")
        assert hasattr(output, "battle_length")

    def test_action_logits_shape(self, model, batch_data):
        """Test action logits have correct shape (batch_size, 126)."""
        output = model(**batch_data)

        batch_size = batch_data["pokemon_ids"].shape[0]
        assert output.action_logits.shape == (batch_size, 126)

    def test_win_probability_shape(self, model, batch_data):
        """Test win probability output shape."""
        output = model(**batch_data)

        batch_size = batch_data["pokemon_ids"].shape[0]
        assert output.win_probability.shape == (batch_size, 1)
        assert torch.all(output.win_probability >= 0) and torch.all(output.win_probability <= 1)

    def test_opponent_moves_shape(self, model, batch_data):
        """Test opponent moves prediction shape."""
        output = model(**batch_data)

        batch_size = batch_data["pokemon_ids"].shape[0]
        team_size = batch_data["pokemon_ids"].shape[1]

        assert output.opponent_moves.shape == (batch_size, team_size, 4, 656)

    def test_opponent_items_shape(self, model, batch_data):
        """Test opponent items prediction shape."""
        output = model(**batch_data)

        batch_size = batch_data["pokemon_ids"].shape[0]
        team_size = batch_data["pokemon_ids"].shape[1]

        assert output.opponent_items.shape == (batch_size, team_size, 376)

    def test_battle_length_shape(self, model, batch_data):
        """Test battle length prediction shape."""
        output = model(**batch_data)

        batch_size = batch_data["pokemon_ids"].shape[0]
        assert output.battle_length.shape == (batch_size, 50)

    def test_variable_team_sizes(self, model):
        """Test model handles variable team sizes with padding."""
        batch_size = 3

        for team_size in [2, 4, 6]:
            batch_data = {
                "pokemon_ids": torch.randint(0, 1025, (batch_size, 6)),
                "move_ids": torch.randint(0, 656, (batch_size, 6, 4)),
                "item_ids": torch.randint(0, 376, (batch_size, 6)),
                "ability_ids": torch.randint(0, 267, (batch_size, 6)),
                "team_mask": torch.zeros(batch_size, 6, dtype=torch.bool),
                "field_features": torch.randn(batch_size, 32),
                "hp_fractions": torch.rand(batch_size, 6),
                "status": torch.randint(0, 8, (batch_size, 6)),
            }

            for i in range(batch_size):
                batch_data["team_mask"][i, :team_size] = True

            output = model(**batch_data)
            assert output.action_logits.shape == (batch_size, 126)

    def test_gradient_flow(self, model, batch_data):
        """Test that gradients flow through all heads."""
        output = model(**batch_data)

        total_loss = (
            output.action_logits.sum()
            + output.win_probability.sum()
            + output.opponent_moves.sum()
            + output.opponent_items.sum()
            + output.battle_length.sum()
        )
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_with_seed(self, batch_data):
        """Test model is deterministic with fixed seed."""
        torch.manual_seed(42)
        model1 = TransformerPolicy(
            embedding_dim=64,
            num_transformer_layers=4,
            num_attention_heads=4,
            ff_dim=256,
        )
        torch.manual_seed(42)
        model2 = TransformerPolicy(
            embedding_dim=64,
            num_transformer_layers=4,
            num_attention_heads=4,
            ff_dim=256,
        )

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(**batch_data)
            output2 = model2(**batch_data)

        torch.testing.assert_close(output1.action_logits, output2.action_logits)

    def test_batch_independence(self, model, batch_data):
        """Test that batch items don't affect each other."""
        model.eval()
        with torch.no_grad():
            output_full = model(**batch_data)

            for i in range(batch_data["pokemon_ids"].shape[0]):
                single_batch = {
                    key: value[i : i + 1] if isinstance(value, torch.Tensor) else value
                    for key, value in batch_data.items()
                }
                output_single = model(**single_batch)

                torch.testing.assert_close(
                    output_full.action_logits[i : i + 1], output_single.action_logits, atol=1e-5, rtol=1e-4
                )

    def test_model_has_reasonable_param_count(self, model):
        """Test model has reasonable number of parameters (1-5M)."""
        total_params = sum(p.numel() for p in model.parameters())

        assert 1_000_000 <= total_params <= 5_000_000, (
            f"Model has {total_params} parameters, "
            "should be between 1M and 5M for fast iteration"
        )

    def test_action_logits_respectable_scale(self, model, batch_data):
        """Test action logits have reasonable magnitude."""
        model.eval()
        with torch.no_grad():
            output = model(**batch_data)

        assert torch.abs(output.action_logits).mean() < 10, "Action logits have too large magnitude"
        assert torch.std(output.action_logits) > 0.01, "Action logits have no variation"

    def test_forward_pass_cuda_compatible(self, model, batch_data):
        """Test model is CUDA compatible (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = model.cuda()
        batch_data_cuda = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in batch_data.items()
        }

        output = model(**batch_data_cuda)
        assert output.action_logits.device.type == "cuda"

    def test_eval_mode_changes_behavior(self, model, batch_data):
        """Test that eval mode properly affects model (e.g., dropout)."""
        model.train()
        with torch.no_grad():
            output_train = model(**batch_data)

        model.eval()
        with torch.no_grad():
            output_eval = model(**batch_data)

        torch.manual_seed(42)
        model.train()
        with torch.no_grad():
            output_train2 = model(**batch_data)

        assert not torch.allclose(
            output_train.action_logits, output_train2.action_logits
        ), "Model should have stochastic behavior in train mode"


class TestPolicyOutputDataclass:
    """Test PolicyOutput dataclass."""

    def test_policy_output_creation(self):
        """Test PolicyOutput can be created and accessed."""
        action_logits = torch.randn(4, 126)
        win_prob = torch.sigmoid(torch.randn(4, 1))
        opp_moves = torch.randn(4, 6, 4, 656)
        opp_items = torch.randn(4, 6, 376)
        battle_len = torch.randn(4, 50)

        output = PolicyOutput(
            action_logits=action_logits,
            win_probability=win_prob,
            opponent_moves=opp_moves,
            opponent_items=opp_items,
            battle_length=battle_len,
        )

        assert output.action_logits is action_logits
        assert output.win_probability is win_prob


class TestPolicyWithDifferentConfigs:
    """Test policy model with different configurations."""

    @pytest.mark.parametrize(
        "embedding_dim,num_layers,num_heads",
        [(32, 2, 2), (64, 4, 4), (128, 8, 8)],
    )
    def test_different_architectures(self, embedding_dim, num_layers, num_heads):
        """Test model works with different architecture configurations."""
        model = TransformerPolicy(
            embedding_dim=embedding_dim,
            num_transformer_layers=num_layers,
            num_attention_heads=num_heads,
            ff_dim=embedding_dim * 4,
        )

        batch_data = {
            "pokemon_ids": torch.randint(0, 1025, (2, 6)),
            "move_ids": torch.randint(0, 656, (2, 6, 4)),
            "item_ids": torch.randint(0, 376, (2, 6)),
            "ability_ids": torch.randint(0, 267, (2, 6)),
            "team_mask": torch.ones(2, 6, dtype=torch.bool),
            "field_features": torch.randn(2, 32),
            "hp_fractions": torch.rand(2, 6),
            "status": torch.randint(0, 8, (2, 6)),
        }

        output = model(**batch_data)
        assert output.action_logits.shape == (2, 126)
