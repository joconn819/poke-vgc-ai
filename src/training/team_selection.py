"""Team-preview action encoding and policy.

The policy treats a VGC team-preview decision as one atomic action: choose an
unordered set of four Pokemon and an ordered pair of leads from that set.
There are ``C(6, 4) * P(4, 2) = 180`` such actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable, Sequence

import torch
from torch import nn

TEAM_SIZE = 6
SELECTED_SIZE = 4
LEAD_SIZE = 2
ACTION_COUNT = 180


@dataclass(frozen=True)
class TeamSelectionAction:
    """A legal team selection, with leads in battlefield order."""

    team: tuple[int, ...]
    leads: tuple[int, ...]


def _all_actions() -> tuple[TeamSelectionAction, ...]:
    return tuple(
        TeamSelectionAction(team, leads)
        for team in combinations(range(TEAM_SIZE), SELECTED_SIZE)
        for leads in permutations(team, LEAD_SIZE)
    )


ACTIONS = _all_actions()
_ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def encode_action(
    team: TeamSelectionAction | Iterable[int],
    leads: Iterable[int] | None = None,
) -> int:
    """Encode a team and ordered leads as an integer in ``[0, ACTION_COUNT)``."""
    if isinstance(team, TeamSelectionAction):
        if leads is not None:
            raise ValueError("leads must be omitted when team is an action")
        action = team
    else:
        if leads is None:
            raise ValueError("leads are required when team is a sequence")
        action = TeamSelectionAction(tuple(sorted(team)), tuple(leads))

    if len(action.team) != SELECTED_SIZE or len(set(action.team)) != SELECTED_SIZE:
        raise ValueError("team must contain four distinct Pokemon indices")
    if any(index < 0 or index >= TEAM_SIZE for index in action.team):
        raise ValueError("team indices must be between 0 and 5")
    if len(action.leads) != LEAD_SIZE or len(set(action.leads)) != LEAD_SIZE:
        raise ValueError("leads must contain two distinct Pokemon indices")
    if not set(action.leads).issubset(action.team):
        raise ValueError("leads must be selected team members")
    try:
        return _ACTION_TO_INDEX[action]
    except KeyError as error:
        raise ValueError("invalid team selection") from error


def decode_action(index: int) -> TeamSelectionAction:
    """Decode an integer action into its selected team and ordered leads."""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index must be an integer in [0, {ACTION_COUNT})")
    return ACTIONS[index]


def action_mask(available: torch.Tensor | Sequence[bool]) -> torch.Tensor:
    """Return legal-action masks for available Pokemon.

    ``available`` may be shaped ``(6,)`` or ``(batch, 6)``.  An action is legal
    only when all four selected Pokemon are available.
    """
    available_tensor = torch.as_tensor(available, dtype=torch.bool)
    if available_tensor.shape[-1] != TEAM_SIZE:
        raise ValueError("availability must have a final dimension of six")
    flat = available_tensor.reshape(-1, TEAM_SIZE)
    selected = torch.tensor([action.team for action in ACTIONS], device=flat.device)
    mask = flat[:, selected].all(dim=2)
    return mask.reshape(*available_tensor.shape[:-1], ACTION_COUNT)


class TeamSelectionPolicy(nn.Module):
    """Small masked policy network for team-preview decisions."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(TEAM_SIZE * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_COUNT),
        )

    def forward(
        self,
        team_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits, with illegal actions set to ``-inf``.

        ``team_features`` has shape ``(batch, 6, feature_dim)``.  ``mask`` may
        be an action mask shaped ``(batch, 180)`` or an availability mask
        shaped ``(batch, 6)``.
        """
        if team_features.ndim != 3 or team_features.shape[1:] != (TEAM_SIZE, self.feature_dim):
            raise ValueError(
                f"team_features must have shape (batch, {TEAM_SIZE}, {self.feature_dim})"
            )
        logits = self.network(team_features.reshape(team_features.shape[0], -1))
        if mask is None:
            return logits
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if mask_tensor.shape[-1] == TEAM_SIZE:
            mask_tensor = action_mask(mask_tensor)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] == 1 and logits.shape[0] != 1:
            mask_tensor = mask_tensor.expand(logits.shape[0], -1)
        if mask_tensor.shape != logits.shape:
            raise ValueError(f"mask must have shape {tuple(logits.shape)} or (batch, 6)")
        return logits.masked_fill(~mask_tensor, -torch.inf)

    def sample(
        self, team_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return ``(indices, log probabilities)``."""
        logits = self(team_features, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        indices = distribution.sample()
        return indices, distribution.log_prob(indices)
