"""Battle replayer for iterating through and extracting training data from battle records.

Provides efficient iteration over battle trajectories, extracting states, actions,
and outcomes for supervised learning.
"""

from typing import Iterator, Tuple, Optional
import numpy as np

from src.training.synthetic_battles import BattleRecord


class BattleReplayer:
    """Iterates through a battle trajectory and extracts training data.
    
    Provides a clean interface for iterating through recorded battles and
    extracting (state, action, done, reward) tuples for training models.
    """

    def __init__(self, battle_record: BattleRecord):
        """Initialize BattleReplayer with a battle record.
        
        Args:
            battle_record: BattleRecord instance to replay.
        """
        self.battle_record = battle_record
        self._current_index = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, bool, float]]:
        """Iterate through battle trajectory steps.
        
        Yields:
            Tuples of (state, action, done, reward) for each step in trajectory.
        """
        self._current_index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, int, bool, float]:
        """Get next step in battle trajectory.
        
        Returns:
            Tuple of (state, action, done, reward).
        
        Raises:
            StopIteration: When trajectory is exhausted.
        """
        if self._current_index >= len(self.battle_record.trajectory):
            raise StopIteration
        
        step = self.battle_record.trajectory[self._current_index]
        self._current_index += 1
        
        return (
            step.state,
            step.action,
            step.done,
            step.reward,
        )

    def get_trajectory_length(self) -> int:
        """Get the length of the battle trajectory.
        
        Returns:
            Number of steps in the trajectory.
        """
        return len(self.battle_record.trajectory)

    def get_winner(self) -> str:
        """Get the winner of the battle.
        
        Returns:
            "player" or "opponent".
        """
        return self.battle_record.winner

    def get_states(self) -> np.ndarray:
        """Get all states from the trajectory.
        
        Returns:
            Array of shape (num_steps, state_dim).
        """
        states = [step.state for step in self.battle_record.trajectory]
        return np.array(states)

    def get_actions(self) -> np.ndarray:
        """Get all actions from the trajectory.
        
        Returns:
            Array of shape (num_steps,) with action indices.
        """
        actions = [step.action for step in self.battle_record.trajectory]
        return np.array(actions)

    def get_rewards(self) -> np.ndarray:
        """Get all rewards from the trajectory.
        
        Returns:
            Array of shape (num_steps,) with rewards.
        """
        rewards = [step.reward for step in self.battle_record.trajectory]
        return np.array(rewards)

    def get_dones(self) -> np.ndarray:
        """Get done flags from the trajectory.
        
        Returns:
            Array of shape (num_steps,) with boolean done flags.
        """
        dones = [step.done for step in self.battle_record.trajectory]
        return np.array(dones)

    def get_legal_actions_masks(self) -> np.ndarray:
        """Get legal action masks from the trajectory.
        
        Returns:
            Array of shape (num_steps, num_actions).
        """
        masks = [step.legal_actions_mask for step in self.battle_record.trajectory]
        return np.array(masks)
