"""Policy loss functions for multi-task learning.

Combines action prediction loss with auxiliary task losses:
- Win probability prediction
- Opponent moveset prediction
- Opponent item prediction
- Battle length estimation
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyLoss(nn.Module):
    """Combined loss for policy network with multi-task learning.

    Weights are used to balance the main task (action prediction) against
    auxiliary tasks.
    """

    def __init__(
        self,
        action_weight: float = 1.0,
        win_weight: float = 0.1,
        opponent_moves_weight: float = 0.1,
        opponent_items_weight: float = 0.1,
        battle_length_weight: float = 0.1,
    ):
        """Initialize loss function.

        Args:
            action_weight: Weight for main action prediction loss
            win_weight: Weight for win probability loss
            opponent_moves_weight: Weight for opponent moves prediction
            opponent_items_weight: Weight for opponent items prediction
            battle_length_weight: Weight for battle length prediction
        """
        super().__init__()

        self.action_weight = action_weight
        self.win_weight = win_weight
        self.opponent_moves_weight = opponent_moves_weight
        self.opponent_items_weight = opponent_items_weight
        self.battle_length_weight = battle_length_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Dictionary with keys:
                - action_logits: (batch_size, 126)
                - win_probability: (batch_size, 1)
                - opponent_moves: (batch_size, team_size, 4, vocab_size_moves)
                - opponent_items: (batch_size, team_size, vocab_size_items)
                - battle_length: (batch_size, 50)
            targets: Dictionary with keys:
                - action_indices: (batch_size,)
                - win_target: (batch_size, 1)
                - opponent_move_indices: (batch_size, team_size, 4)
                - opponent_item_indices: (batch_size, team_size)
                - battle_length_targets: (batch_size,)
            action_mask: Optional (batch_size, 126) boolean mask for valid actions

        Returns:
            Combined loss tensor (scalar)
        """
        batch_size = predictions["action_logits"].shape[0]

        if batch_size == 0:
            raise ValueError("Cannot compute loss for empty batch")

        action_loss = self._compute_action_loss(
            predictions["action_logits"],
            targets["action_indices"],
            action_mask,
        )

        win_loss = self._compute_win_loss(
            predictions["win_probability"],
            targets["win_target"],
        )

        opponent_moves_loss = self._compute_opponent_moves_loss(
            predictions["opponent_moves"],
            targets["opponent_move_indices"],
        )

        opponent_items_loss = self._compute_opponent_items_loss(
            predictions["opponent_items"],
            targets["opponent_item_indices"],
        )

        battle_length_loss = self._compute_battle_length_loss(
            predictions["battle_length"],
            targets["battle_length_targets"],
        )

        total_loss = (
            self.action_weight * action_loss
            + self.win_weight * win_loss
            + self.opponent_moves_weight * opponent_moves_loss
            + self.opponent_items_weight * opponent_items_loss
            + self.battle_length_weight * battle_length_loss
        )

        return total_loss

    @staticmethod
    def _compute_action_loss(
        action_logits: torch.Tensor,
        action_indices: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute action prediction loss with optional masking.

        Args:
            action_logits: (batch_size, 126)
            action_indices: (batch_size,)
            action_mask: Optional (batch_size, 126) boolean mask

        Returns:
            Scalar loss

        Raises:
            ValueError: If all actions are masked for any batch item
        """
        if action_mask is None:
            return F.cross_entropy(action_logits, action_indices)

        if torch.any(torch.all(~action_mask, dim=1)):
            raise ValueError("Cannot compute loss when all actions are masked for a batch item")

        masked_logits = action_logits.clone()
        masked_logits[~action_mask] = float("-inf")

        log_probs = F.log_softmax(masked_logits, dim=1)
        action_loss = F.nll_loss(log_probs, action_indices)

        return action_loss

    @staticmethod
    def _compute_win_loss(
        win_probability: torch.Tensor,
        win_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute win probability loss.

        Args:
            win_probability: (batch_size, 1) in [0, 1]
            win_target: (batch_size, 1) in {0, 1}

        Returns:
            Scalar loss
        """
        return F.binary_cross_entropy(win_probability, win_target)

    @staticmethod
    def _compute_opponent_moves_loss(
        opponent_moves: torch.Tensor,
        opponent_move_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute opponent moves prediction loss.

        Args:
            opponent_moves: (batch_size, team_size, 4, vocab_size_moves)
            opponent_move_indices: (batch_size, team_size, 4)

        Returns:
            Scalar loss
        """
        batch_size, team_size, num_slots, num_moves = opponent_moves.shape

        opponent_moves_flat = opponent_moves.reshape(batch_size * team_size * num_slots, num_moves)
        opponent_move_indices_flat = opponent_move_indices.reshape(batch_size * team_size * num_slots)

        return F.cross_entropy(opponent_moves_flat, opponent_move_indices_flat)

    @staticmethod
    def _compute_opponent_items_loss(
        opponent_items: torch.Tensor,
        opponent_item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute opponent items prediction loss.

        Args:
            opponent_items: (batch_size, team_size, vocab_size_items)
            opponent_item_indices: (batch_size, team_size)

        Returns:
            Scalar loss
        """
        batch_size, team_size, num_items = opponent_items.shape

        opponent_items_flat = opponent_items.reshape(batch_size * team_size, num_items)
        opponent_item_indices_flat = opponent_item_indices.reshape(batch_size * team_size)

        return F.cross_entropy(opponent_items_flat, opponent_item_indices_flat)

    @staticmethod
    def _compute_battle_length_loss(
        battle_length: torch.Tensor,
        battle_length_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute battle length prediction loss.

        Args:
            battle_length: (batch_size, 50) logits for turn buckets
            battle_length_targets: (batch_size,) target turn bucket index

        Returns:
            Scalar loss
        """
        return F.cross_entropy(battle_length, battle_length_targets)
