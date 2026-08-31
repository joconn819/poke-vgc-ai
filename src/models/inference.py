"""Inference utilities for the policy network.

Handles action sampling, masking, decoding action pairs, and multi-task predictions.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


class PolicyInference:
    """Inference-time utilities for policy predictions."""

    @staticmethod
    def apply_action_mask(
        action_logits: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Apply action mask to logits.

        Args:
            action_logits: (batch_size, 126)
            action_mask: (batch_size, 126) boolean mask, True for valid actions
            temperature: Temperature for softmax (>1 = softer, <1 = sharper)

        Returns:
            Masked logits with invalid actions set to -inf
        """
        masked_logits = action_logits.clone()
        masked_logits[~action_mask] = float("-inf")

        if temperature != 1.0:
            masked_logits = masked_logits / temperature

        return masked_logits

    @staticmethod
    def sample_actions(
        action_logits: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample actions from masked distribution.

        Args:
            action_logits: (batch_size, 126)
            action_mask: (batch_size, 126) boolean mask
            temperature: Temperature for sampling

        Returns:
            Sampled action indices (batch_size,)

        Raises:
            RuntimeError: If no valid actions in mask
        """
        masked_logits = PolicyInference.apply_action_mask(action_logits, action_mask, temperature)

        if torch.any(torch.all(~action_mask, dim=1)):
            raise RuntimeError("Batch contains entries with no valid actions")

        probs = F.softmax(masked_logits, dim=1)

        if torch.any(torch.isnan(probs)):
            raise ValueError("Probability distribution contains NaN values")

        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

        return actions

    @staticmethod
    def argmax_actions(
        action_logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select actions via argmax over masked distribution.

        Args:
            action_logits: (batch_size, 126)
            action_mask: (batch_size, 126) boolean mask

        Returns:
            Argmax action indices (batch_size,)

        Raises:
            RuntimeError: If no valid actions in mask
        """
        masked_logits = PolicyInference.apply_action_mask(action_logits, action_mask)

        if torch.any(torch.all(~action_mask, dim=1)):
            raise RuntimeError("Batch contains entries with no valid actions")

        actions = torch.argmax(masked_logits, dim=1)

        return actions

    @staticmethod
    def decode_action_pair(action_index: int) -> Tuple[int, int, int, int]:
        """Decode flattened action index into components.

        Action space: 4 moves for player 1 × 4 moves for player 2 ×
                     3 targets for player 1 × 3 targets for player 2 = 4 × 4 × 3 × 3 = 144

        We use only 126 valid combinations (excluding some invalid dual-targeting scenarios).

        Args:
            action_index: Index in [0, 126)

        Returns:
            Tuple of (move1, move2, target1, target2)
        """
        move1 = action_index // 36
        remainder = action_index % 36

        move2 = remainder // 9
        remainder = remainder % 9

        target1 = remainder // 3
        target2 = remainder % 3

        return move1, move2, target1, target2

    @staticmethod
    def batch_inference(
        action_logits: torch.Tensor,
        action_mask: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient batch inference for action selection.

        Args:
            action_logits: (batch_size, 126)
            action_mask: (batch_size, 126)
            sample: Whether to sample (True) or use argmax (False)
            temperature: Temperature for sampling

        Returns:
            Tuple of (actions, probabilities)
            - actions: (batch_size,)
            - probabilities: (batch_size,) probability of selected action
        """
        masked_logits = PolicyInference.apply_action_mask(
            action_logits, action_mask, temperature
        )

        probs = F.softmax(masked_logits, dim=1)

        if sample:
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(masked_logits, dim=1)

        selected_probs = probs[torch.arange(probs.shape[0]), actions]

        return actions, selected_probs

    @staticmethod
    def predict_opponent_moves(opponent_moves: torch.Tensor) -> torch.Tensor:
        """Predict opponent movesets.

        Args:
            opponent_moves: (batch_size, team_size, 4, vocab_size_moves)

        Returns:
            Predicted move indices (batch_size, team_size, 4)
        """
        predicted_moves = torch.argmax(opponent_moves, dim=-1)

        return predicted_moves

    @staticmethod
    def predict_opponent_items(opponent_items: torch.Tensor) -> torch.Tensor:
        """Predict opponent items.

        Args:
            opponent_items: (batch_size, team_size, vocab_size_items)

        Returns:
            Predicted item indices (batch_size, team_size)
        """
        predicted_items = torch.argmax(opponent_items, dim=-1)

        return predicted_items

    @staticmethod
    def predict_battle_length(battle_length: torch.Tensor) -> torch.Tensor:
        """Predict battle length.

        Args:
            battle_length: (batch_size, 50) logits for turn buckets

        Returns:
            Predicted turn bucket indices (batch_size,)
        """
        predicted_length = torch.argmax(battle_length, dim=-1)

        return predicted_length

    @staticmethod
    def predict_win_probability(win_probability: torch.Tensor) -> torch.Tensor:
        """Extract win probability predictions.

        Args:
            win_probability: (batch_size, 1) in [0, 1]

        Returns:
            Win probability (batch_size,)
        """
        return win_probability.squeeze(1)
