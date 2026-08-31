"""Regulation-aware replay mixing and legality helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.utils.config import RegulationConfig


@dataclass(frozen=True)
class ReplayExampleMetadata:
    """Metadata required to safely reuse a trajectory across regulations."""

    regulation_id: str
    legal_action_ids: tuple[int, ...]


class RegulationAdapter:
    """Apply regulation-specific masks while retaining historical examples."""

    def __init__(self, regulation: RegulationConfig, historical_weight: float = 0.25):
        if not 0.0 <= historical_weight <= 1.0:
            raise ValueError("historical_weight must be between 0 and 1")
        self.regulation = regulation
        self.historical_weight = historical_weight

    def filter_action_mask(
        self, action_mask: Sequence[int], legal_action_ids: Iterable[int]
    ) -> np.ndarray:
        """Intersect a stored mask with actions legal in the current regulation."""
        mask = np.asarray(action_mask, dtype=np.int8).copy()
        allowed = {int(action) for action in legal_action_ids}
        mask[[index for index in range(mask.size) if index not in allowed]] = 0
        if not mask.any():
            raise ValueError("Regulation removed every action from the mask")
        return mask

    def replay_weight(self, metadata: ReplayExampleMetadata) -> float:
        """Weight current-regulation data fully and historical data conservatively."""
        return 1.0 if metadata.regulation_id == self.regulation.regulation_id else self.historical_weight

    def weighted_indices(self, metadata: Sequence[ReplayExampleMetadata]) -> np.ndarray:
        """Return normalized sampling probabilities for a replay manifest."""
        weights = np.asarray([self.replay_weight(item) for item in metadata], dtype=np.float64)
        if weights.size == 0 or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Replay metadata must contain at least one valid example")
        return weights / weights.sum()
