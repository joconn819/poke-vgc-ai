"""Tests for policy inference utilities."""

import pytest
import torch
from src.models.inference import PolicyInference
from src.models.policy import PolicyOutput


class TestPolicyInference:
    """Test policy inference utilities."""

    @pytest.fixture
    def inference(self):
        """Create a policy inference instance."""
        return PolicyInference()

    @pytest.fixture
    def policy_output(self):
        """Create sample policy output."""
        batch_size = 4
        team_size = 6

        return PolicyOutput(
            action_logits=torch.randn(batch_size, 126),
            win_probability=torch.sigmoid(torch.randn(batch_size, 1)),
            opponent_moves=torch.randn(batch_size, team_size, 4, 656),
            opponent_items=torch.randn(batch_size, team_size, 376),
            battle_length=torch.randn(batch_size, 50),
        )

    @pytest.fixture
    def action_masks(self):
        """Create action masks."""
        batch_size = 4
        masks = torch.ones(batch_size, 126, dtype=torch.bool)
        masks[:, 100:126] = False
        return masks

    def test_apply_action_mask(self, inference, policy_output, action_masks):
        """Test action mask is properly applied."""
        masked_logits = inference.apply_action_mask(policy_output.action_logits, action_masks)

        assert masked_logits.shape == policy_output.action_logits.shape
        assert not torch.isnan(masked_logits).any()

    def test_masked_logits_negative_infinity(self, inference, policy_output, action_masks):
        """Test masked actions are set to negative infinity."""
        masked_logits = inference.apply_action_mask(policy_output.action_logits, action_masks)

        batch_size = policy_output.action_logits.shape[0]
        for i in range(batch_size):
            masked_indices = ~action_masks[i]
            assert (masked_logits[i, masked_indices] == float("-inf")).all()

    def test_sample_actions_from_distribution(self, inference, policy_output, action_masks):
        """Test sampling actions from masked distribution."""
        torch.manual_seed(42)
        sampled_actions = inference.sample_actions(policy_output.action_logits, action_masks)

        assert sampled_actions.shape == (policy_output.action_logits.shape[0],)
        assert sampled_actions.dtype == torch.long
        assert torch.all(sampled_actions >= 0)
        assert torch.all(sampled_actions < 126)

    def test_sampled_actions_respect_mask(self, inference, policy_output, action_masks):
        """Test sampled actions respect the mask."""
        torch.manual_seed(42)
        sampled_actions = inference.sample_actions(policy_output.action_logits, action_masks)

        batch_size = policy_output.action_logits.shape[0]
        for i in range(batch_size):
            assert action_masks[i, sampled_actions[i]]

    def test_argmax_actions(self, inference, policy_output, action_masks):
        """Test selecting actions via argmax."""
        argmax_actions = inference.argmax_actions(policy_output.action_logits, action_masks)

        assert argmax_actions.shape == (policy_output.action_logits.shape[0],)
        assert argmax_actions.dtype == torch.long
        assert torch.all(argmax_actions >= 0)
        assert torch.all(argmax_actions < 126)

    def test_argmax_actions_respect_mask(self, inference, policy_output, action_masks):
        """Test argmax actions respect the mask."""
        argmax_actions = inference.argmax_actions(policy_output.action_logits, action_masks)

        batch_size = policy_output.action_logits.shape[0]
        for i in range(batch_size):
            assert action_masks[i, argmax_actions[i]]

    def test_decode_action_pair(self, inference):
        """Test decoding action pair from flattened index."""
        action_index = 42

        move1, move2, target1, target2 = inference.decode_action_pair(action_index)

        assert 0 <= move1 < 4
        assert 0 <= move2 < 4
        assert 0 <= target1 < 3
        assert 0 <= target2 < 3

    def test_decode_action_pair_invertible(self, inference):
        """Test encoding and decoding are inverse operations."""
        for move1 in range(4):
            for move2 in range(4):
                for target1 in range(3):
                    for target2 in range(3):
                        action_index = (
                            move1 * 36 + move2 * 9 + target1 * 3 + target2
                        )

                        decoded = inference.decode_action_pair(action_index)
                        assert decoded == (move1, move2, target1, target2)

    def test_batch_inference(self, inference, policy_output, action_masks):
        """Test efficient batch inference."""
        batch_size = policy_output.action_logits.shape[0]

        actions, probs = inference.batch_inference(
            policy_output.action_logits, action_masks, sample=False
        )

        assert actions.shape == (batch_size,)
        assert probs.shape == (batch_size,)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_batch_inference_deterministic(self, inference, policy_output, action_masks):
        """Test batch inference is deterministic with argmax."""
        torch.manual_seed(42)
        actions1, _ = inference.batch_inference(
            policy_output.action_logits.clone(), action_masks, sample=False
        )

        torch.manual_seed(42)
        actions2, _ = inference.batch_inference(
            policy_output.action_logits.clone(), action_masks, sample=False
        )

        torch.testing.assert_close(actions1, actions2)

    def test_batch_inference_stochastic(self, inference, policy_output, action_masks):
        """Test batch inference is stochastic with sampling."""
        torch.manual_seed(42)
        actions1, _ = inference.batch_inference(
            policy_output.action_logits.clone(), action_masks, sample=True
        )

        torch.manual_seed(43)
        actions2, _ = inference.batch_inference(
            policy_output.action_logits.clone(), action_masks, sample=True
        )

        assert not torch.equal(actions1, actions2)

    def test_predict_opponent_moves(self, inference, policy_output):
        """Test predicting opponent movesets."""
        batch_size = policy_output.opponent_moves.shape[0]
        team_size = policy_output.opponent_moves.shape[1]

        predicted_moves = inference.predict_opponent_moves(policy_output.opponent_moves)

        assert predicted_moves.shape == (batch_size, team_size, 4)
        assert predicted_moves.dtype == torch.long

    def test_predict_opponent_items(self, inference, policy_output):
        """Test predicting opponent items."""
        batch_size = policy_output.opponent_items.shape[0]
        team_size = policy_output.opponent_items.shape[1]

        predicted_items = inference.predict_opponent_items(policy_output.opponent_items)

        assert predicted_items.shape == (batch_size, team_size)
        assert predicted_items.dtype == torch.long

    def test_predict_battle_length(self, inference, policy_output):
        """Test predicting battle length."""
        batch_size = policy_output.battle_length.shape[0]

        predicted_length = inference.predict_battle_length(policy_output.battle_length)

        assert predicted_length.shape == (batch_size,)
        assert predicted_length.dtype == torch.long

    def test_predict_win_probability(self, inference, policy_output):
        """Test extracting win probability."""
        batch_size = policy_output.win_probability.shape[0]

        win_prob = inference.predict_win_probability(policy_output.win_probability)

        assert win_prob.shape == (batch_size,)
        assert torch.all(win_prob >= 0) and torch.all(win_prob <= 1)

    def test_inference_output_consistency(self, inference, policy_output, action_masks):
        """Test inference output is consistent across calls."""
        policy_output.action_logits = policy_output.action_logits.clone().detach()

        output1 = inference.batch_inference(
            policy_output.action_logits, action_masks, sample=False
        )
        output2 = inference.batch_inference(
            policy_output.action_logits.clone(), action_masks, sample=False
        )

        torch.testing.assert_close(output1[0], output2[0])
        torch.testing.assert_close(output1[1], output2[1])

    def test_temperature_scaling(self, inference):
        """Test temperature scaling of logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        probs_temp1 = torch.softmax(logits / 1.0, dim=-1)
        probs_temp2 = torch.softmax(logits / 2.0, dim=-1)

        assert torch.all(probs_temp1[0, -1] > probs_temp2[0, -1])


class TestPolicyInferenceEdgeCases:
    """Test edge cases in policy inference."""

    def test_all_masked_actions_sampling(self):
        """Test handling when all actions are masked (should fail gracefully)."""
        inference = PolicyInference()
        logits = torch.randn(1, 126)
        mask = torch.zeros(1, 126, dtype=torch.bool)

        with pytest.raises((RuntimeError, ValueError)):
            inference.sample_actions(logits, mask)

    def test_single_valid_action(self):
        """Test when only one action is valid."""
        inference = PolicyInference()
        logits = torch.randn(1, 126)
        mask = torch.zeros(1, 126, dtype=torch.bool)
        mask[0, 42] = True

        action = inference.argmax_actions(logits, mask)
        assert action[0] == 42

    def test_extreme_logit_values(self):
        """Test handling extreme logit values."""
        inference = PolicyInference()
        logits = torch.tensor([[1e6, -1e6, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        action = inference.argmax_actions(logits, mask)
        assert action[0] == 0

    def test_nan_handling(self):
        """Test handling NaN values in logits."""
        inference = PolicyInference()
        logits = torch.tensor([[float("nan"), 1.0, 2.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        action = inference.argmax_actions(logits, mask)
        assert action[0] >= 0 and action[0] < 3
