"""Metrics for supervised policy training and evaluation."""

from __future__ import annotations

from typing import Dict

import torch


def action_accuracy(
    action_logits: torch.Tensor,
    action_indices: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> float:
    logits = action_logits
    if action_mask is not None:
        logits = logits.clone()
        logits[~action_mask] = float("-inf")
    predictions = logits.argmax(dim=1)
    return (predictions == action_indices).float().mean().item()


def binary_brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((probabilities.view(-1) - targets.view(-1)) ** 2).item()


def opponent_moves_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


def opponent_items_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


def battle_length_mae(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1).float()
    return torch.mean(torch.abs(predictions - targets.float())).item()


def summarize_metrics(
    action_logits: torch.Tensor,
    win_probability: torch.Tensor,
    opponent_moves: torch.Tensor,
    opponent_items: torch.Tensor,
    battle_length: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        "action_accuracy": action_accuracy(
            action_logits,
            batch["action_index"],
            batch.get("action_mask"),
        ),
        "win_brier_score": binary_brier_score(
            win_probability,
            batch["win_target"],
        ),
        "opponent_moves_accuracy": opponent_moves_accuracy(
            opponent_moves,
            batch["opponent_move_indices"],
        ),
        "opponent_items_accuracy": opponent_items_accuracy(
            opponent_items,
            batch["opponent_item_indices"],
        ),
        "battle_length_mae": battle_length_mae(
            battle_length,
            batch["battle_length_target"],
        ),
    }

