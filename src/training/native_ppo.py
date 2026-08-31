"""Utilities for turning native Showdown trajectories into PPO batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from src.training.ppo import compute_gae, ppo_policy_loss, ppo_value_loss


@dataclass(frozen=True)
class NativeTransition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: np.ndarray
    reward: float
    old_log_prob: float
    value: float
    done: bool


def valid_native_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only complete, non-timeout records with distinct teams."""
    valid = []
    for record in records:
        if record.get("timed_out") or record.get("teams_differ") is not True:
            continue
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            continue
        if any("log_probs" not in step or "values" not in step for step in trajectory):
            continue
        valid.append(record)
    return valid


def extract_player_transitions(record: dict[str, Any], player: str) -> list[NativeTransition]:
    """Extract one player's factorized native transitions from a record."""
    result = []
    for index, step in enumerate(record["trajectory"]):
        mask = np.asarray(step["action_masks"][player], dtype=np.int8)
        action = np.asarray(step["actions"][player], dtype=np.int64)
        if mask.ndim != 1 or mask.size % 2 or action.shape != (2,):
            raise ValueError("native transition has invalid mask or action shape")
        result.append(
            NativeTransition(
                observation=np.asarray(step["observations"][player], dtype=np.float32),
                action_mask=mask,
                action=action,
                reward=float(step["rewards"].get(player, 0.0)),
                old_log_prob=float(step["log_probs"][player]),
                value=float(step["values"][player]),
                done=index == len(record["trajectory"]) - 1,
            )
        )
    return result


def factorized_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    value_weight: float = 0.5,
) -> torch.Tensor:
    """Apply PPO to joint doubles actions represented by summed slot log-probs."""
    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError("policy tensors must have matching shapes")
    return ppo_policy_loss(log_probs, old_log_probs, advantages) + value_weight * ppo_value_loss(
        values, returns
    )


def compute_native_gae(transitions: list[NativeTransition]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for one extracted player's trajectory."""
    if not transitions:
        raise ValueError("cannot compute GAE for an empty trajectory")
    rewards = torch.tensor([[item.reward for item in transitions]], dtype=torch.float32)
    values = torch.tensor([[item.value for item in transitions]], dtype=torch.float32)
    dones = torch.tensor([[float(item.done) for item in transitions]], dtype=torch.float32)
    return compute_gae(rewards, values, dones)
