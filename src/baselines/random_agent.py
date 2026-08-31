"""Random agent that samples uniformly from legal actions."""

import numpy as np
from src.baselines.base_agent import Agent


class RandomAgent(Agent):
    """Selects actions uniformly at random from legal actions.
    
    Useful for baseline comparison and exploration.
    """

    def __init__(self, seed: int | None = None):
        """Initialize RandomAgent.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    def predict(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Select a uniformly random legal action.
        
        Args:
            observation: Current state observation (unused).
            legal_actions_mask: Binary mask of legal actions.
        
        Returns:
            Index of a randomly selected legal action.
        """
        legal_actions = np.where(legal_actions_mask == 1)[0]
        return int(self.rng.choice(legal_actions))
