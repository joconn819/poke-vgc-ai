"""Proximal Policy Optimization (PPO) building blocks.

These are pure-tensor utilities, kept independent of the environment/rollout
machinery so they are easy to unit test and reuse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    Args:
        rewards: (batch_size, time_steps) per-step rewards.
        values: (batch_size, time_steps) value estimates for each step.
        dones: (batch_size, time_steps) 1.0 if the episode terminated at that
            step (bootstrapping stops there), else 0.0.
        gamma: Discount factor.
        lam: GAE lambda smoothing factor.

    Returns:
        Tuple of (advantages, returns), both shaped (batch_size, time_steps).
    """
    batch_size, time_steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)

    next_value = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(time_steps)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_value * not_done - values[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae
        next_value = values[:, t]

    returns = advantages + values
    return advantages, returns


def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate policy loss (to be minimized).

    Args:
        log_probs: Current policy log-probabilities of taken actions.
        old_log_probs: Log-probabilities under the policy that generated the
            rollout (detached, no gradient).
        advantages: Advantage estimates for the taken actions.
        clip_epsilon: PPO clipping range.

    Returns:
        Scalar loss (negative of the clipped surrogate objective).
    """
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    surrogate = torch.min(unclipped, clipped)
    return -surrogate.mean()


def ppo_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error value function loss."""
    return F.mse_loss(values, returns)
