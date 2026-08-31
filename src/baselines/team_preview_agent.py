"""Agent for team preview phase selection."""

import numpy as np
from src.baselines.base_agent import Agent


class TeamPreviewAgent(Agent):
    """Selects Pokemon during team preview phase.
    
    During team preview, the agent sees opponent team and must order
    its own team. This agent uses simple heuristics to select which
    Pokemon to lead with.
    
    Strategy:
    - Randomly select from available Pokemon
    - Simple baseline for team preview optimization
    """

    def __init__(self, seed: int | None = None):
        """Initialize TeamPreviewAgent.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    def predict(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Select a Pokemon during team preview.
        
        Args:
            observation: Team preview observation (opponent team info, etc.).
            legal_actions_mask: Binary mask of available Pokemon (typically 6 for team size).
        
        Returns:
            Index of selected Pokemon (0-5 for team preview).
        """
        legal_pokemon = np.where(legal_actions_mask == 1)[0]
        
        # Simple heuristic: randomly select from available Pokemon
        return int(self.rng.choice(legal_pokemon))
