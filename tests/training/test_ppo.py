"""Tests for PPO advantage estimation and loss functions."""

import torch

from src.training.ppo import compute_gae, ppo_policy_loss, ppo_value_loss


class TestComputeGAE:
    def test_single_step_episode_advantage_equals_reward_minus_value(self) -> None:
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.0]])
        dones = torch.tensor([[1.0]])

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

        assert torch.allclose(advantages, torch.tensor([[1.0]]))
        assert torch.allclose(returns, torch.tensor([[1.0]]))

    def test_multi_step_episode_discounts_future_rewards(self) -> None:
        # Two-step episode, zero reward until terminal win.
        rewards = torch.tensor([[0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5]])
        dones = torch.tensor([[0.0, 1.0]])

        advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

        # Return at t=1 is 1.0 (terminal). Return at t=0 is 0 + 1.0*1.0 = 1.0.
        assert torch.allclose(returns, torch.tensor([[1.0, 1.0]]), atol=1e-5)
        assert torch.allclose(advantages, returns - values, atol=1e-5)

    def test_batched_episodes_are_independent(self) -> None:
        rewards = torch.tensor([[1.0], [-1.0]])
        values = torch.zeros((2, 1))
        dones = torch.ones((2, 1))

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

        assert torch.allclose(returns, torch.tensor([[1.0], [-1.0]]))


class TestPPOPolicyLoss:
    def test_zero_when_ratio_is_one_and_advantage_positive(self) -> None:
        log_probs = torch.tensor([0.1, -0.2])
        old_log_probs = torch.tensor([0.1, -0.2])
        advantages = torch.tensor([1.0, 1.0])

        loss = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)

        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)

    def test_clips_large_positive_ratio(self) -> None:
        old_log_probs = torch.tensor([0.0])
        log_probs = torch.tensor([2.0])  # ratio = e^2 >> 1 + clip
        advantages = torch.tensor([1.0])

        loss = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)

        # Clipped surrogate caps benefit at (1+eps)*advantage.
        assert torch.isclose(loss, torch.tensor(-1.2), atol=1e-5)

    def test_gradient_flows_through_log_probs(self) -> None:
        log_probs = torch.tensor([0.0], requires_grad=True)
        old_log_probs = torch.tensor([0.0])
        advantages = torch.tensor([1.0])

        loss = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)
        loss.backward()

        assert log_probs.grad is not None


class TestPPOValueLoss:
    def test_zero_when_prediction_matches_return(self) -> None:
        values = torch.tensor([1.0, 2.0])
        returns = torch.tensor([1.0, 2.0])

        loss = ppo_value_loss(values, returns)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_when_prediction_differs(self) -> None:
        values = torch.tensor([0.0])
        returns = torch.tensor([1.0])

        loss = ppo_value_loss(values, returns)

        assert loss.item() > 0.0
