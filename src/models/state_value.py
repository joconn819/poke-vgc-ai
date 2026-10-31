"""Scalar win-probability interface over the shared battle-state encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from src.features.volatile_effects import FIELD_FEATURE_SCHEMA
from src.models.policy import TransformerPolicy
from src.training.state_value import (
    STATE_VALUE_FIELD_DIM,
    STATE_VALUE_FIELD_SCHEMA,
    STATE_VALUE_VOLATILE_FIELD_DIM,
)


class StateValueEvaluator(nn.Module):
    """Return only the acting player's probability of winning the battle.

    Inputs mirror the policy's state representation so the evaluator can use
    the same legal, perspective-filtered features as the action policy. The
    wrapped network may have auxiliary heads internally, but this interface
    exposes exactly one probability per state, shaped ``(batch,)``.
    """

    def __init__(
        self,
        state_model: TransformerPolicy,
        *,
        calibration_scale: float = 1.0,
        calibration_bias: float = 0.0,
        calibration_by_turn: list[dict[str, Any]] | None = None,
    ):
        super().__init__()
        if calibration_scale <= 0:
            raise ValueError("calibration_scale must be positive")
        self.state_model = state_model
        self.calibration_scale = float(calibration_scale)
        self.calibration_bias = float(calibration_bias)
        self.calibration_by_turn = calibration_by_turn
        if calibration_by_turn:
            if any(float(row.get("scale", 0.0)) <= 0 for row in calibration_by_turn):
                raise ValueError("turn calibration scales must be positive")
            if calibration_by_turn[-1].get("max_turn") is not None:
                raise ValueError("the final turn-calibration group must be an open-ended bucket")

    def forward(
        self,
        pokemon_ids: torch.Tensor,
        move_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ability_ids: torch.Tensor,
        team_mask: torch.Tensor,
        field_features: torch.Tensor,
        hp_fractions: torch.Tensor,
        status: torch.Tensor,
        history_observations: Optional[torch.Tensor] = None,
        belief_features: Optional[torch.Tensor] = None,
        opponent_pokemon_ids: Optional[torch.Tensor] = None,
        opponent_move_ids: Optional[torch.Tensor] = None,
        opponent_item_ids: Optional[torch.Tensor] = None,
        opponent_ability_ids: Optional[torch.Tensor] = None,
        opponent_team_mask: Optional[torch.Tensor] = None,
        opponent_hp_fractions: Optional[torch.Tensor] = None,
        opponent_status: Optional[torch.Tensor] = None,
        opponent_recent_move_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate win probability from a batch of acting-player observations.

        Opponent inputs must contain only preview-public or battle-revealed
        information. The standard training feature builder enforces this for
        closed-team-sheet VGC records.
        """
        output = self.state_model(
            pokemon_ids=pokemon_ids,
            move_ids=move_ids,
            item_ids=item_ids,
            ability_ids=ability_ids,
            team_mask=team_mask,
            field_features=field_features,
            hp_fractions=hp_fractions,
            status=status,
            history_observations=history_observations,
            belief_features=belief_features,
            opponent_pokemon_ids=opponent_pokemon_ids,
            opponent_move_ids=opponent_move_ids,
            opponent_recent_move_ids=opponent_recent_move_ids,
            opponent_item_ids=opponent_item_ids,
            opponent_ability_ids=opponent_ability_ids,
            opponent_team_mask=opponent_team_mask,
            opponent_hp_fractions=opponent_hp_fractions,
            opponent_status=opponent_status,
        )
        probability = output.win_probability
        if probability.ndim != 2 or probability.shape[1] != 1:
            raise RuntimeError("state model must return one win probability per state")
        # The policy's legacy probability head has a tiny auxiliary critic
        # connection for training compatibility. Clamp its numerical output so
        # this public evaluator always honors the [0, 1] probability contract.
        probability = probability.squeeze(1).clamp(min=0.0, max=1.0)
        if self.calibration_by_turn:
            epsilon = torch.finfo(probability.dtype).eps
            logits = torch.logit(probability.clamp(min=epsilon, max=1.0 - epsilon))
            turns = field_features[:, 0].to(dtype=probability.dtype)
            calibrated = torch.empty_like(probability)
            assigned = torch.zeros_like(turns, dtype=torch.bool)
            for row in self.calibration_by_turn:
                max_turn = row.get("max_turn")
                mask = ~assigned if max_turn is None else (
                    (~assigned) & (turns <= float(max_turn))
                )
                if mask.any():
                    scale = float(row["scale"])
                    bias = float(row["bias"])
                    calibrated[mask] = torch.sigmoid(logits[mask] * scale + bias)
                    assigned |= mask
            if not assigned.all():
                raise RuntimeError("turn calibration does not cover all inputs")
            return calibrated
        if self.calibration_scale == 1.0 and self.calibration_bias == 0.0:
            return probability
        epsilon = torch.finfo(probability.dtype).eps
        logits = torch.logit(probability.clamp(min=epsilon, max=1.0 - epsilon))
        return torch.sigmoid(logits * self.calibration_scale + self.calibration_bias)


def load_state_value_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> StateValueEvaluator:
    """Load a versioned state-value checkpoint emitted by its trainer."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("model_kind") != "closed_sheet_state_value":
        raise ValueError("checkpoint is not a closed-sheet state-value model")
    config = checkpoint.get("model_config") or {}
    field_feature_dim = int(config.get("field_feature_dim", 0) or 0)
    if field_feature_dim >= STATE_VALUE_FIELD_DIM:
        if checkpoint.get("field_feature_schema") != STATE_VALUE_FIELD_SCHEMA:
            raise ValueError(
                "condition-age state-value checkpoint has an unknown field feature schema"
            )
    elif field_feature_dim >= STATE_VALUE_VOLATILE_FIELD_DIM:
        if checkpoint.get("field_feature_schema") != FIELD_FEATURE_SCHEMA:
            raise ValueError(
                "expanded state-value checkpoint has an unknown field feature schema"
            )
    state_model = TransformerPolicy.from_config_dict(config)
    state_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    calibration = checkpoint.get("calibration", {})
    evaluator = StateValueEvaluator(
        state_model,
        calibration_scale=float(calibration.get("scale", 1.0)),
        calibration_bias=float(calibration.get("bias", 0.0)),
        calibration_by_turn=calibration.get("by_turn"),
    ).to(map_location)
    evaluator.eval()
    return evaluator
