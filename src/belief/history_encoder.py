"""Recurrent encoding of battle observations and hidden-information beliefs."""

from __future__ import annotations

import torch
from torch import nn


class BattleHistoryEncoder(nn.Module):
    """Encode a sequence of observations plus belief features into one state."""

    def __init__(self, observation_dim: int = 512, belief_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        if observation_dim <= 0 or belief_dim < 0 or hidden_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.observation_dim = observation_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(observation_dim + belief_dim, hidden_dim, batch_first=True)

    def forward(
        self, observations: torch.Tensor, belief_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the final recurrent state for ``(batch, time, feature)`` inputs."""
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError("observations must have shape (batch, time, observation_dim)")
        if self.belief_dim:
            if belief_features is None:
                belief_features = torch.zeros(
                    (*observations.shape[:2], self.belief_dim),
                    dtype=observations.dtype,
                    device=observations.device,
                )
            if belief_features.shape != (*observations.shape[:2], self.belief_dim):
                raise ValueError("belief_features must align with observations")
            observations = torch.cat([observations, belief_features], dim=-1)
        _, hidden = self.gru(observations)
        return hidden[-1]
