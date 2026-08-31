"""Agent that prefers using moves over switching when safe."""

import numpy as np
from src.baselines.base_agent import Agent


class SwitchPreservingAgent(Agent):
    """Selects actions with preference for moves over switches.
    
    Strategy:
    - Strongly prefer attacking moves (actions 0-3) when available
    - Switch only when moves are not legal
    - Use random selection among preferred actions for variety
    
    This agent tries to preserve team by avoiding unnecessary switches.
    """

    def __init__(self, seed: int | None = None):
        """Initialize SwitchPreservingAgent.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    def predict(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Select action preferring moves over switches.
        
        Args:
            observation: Current state observation (unused).
            legal_actions_mask: Binary mask of legal actions.
        
        Returns:
            Index of action (prefer move over switch).
        """
        legal_actions = np.where(legal_actions_mask == 1)[0]
        
        # Separate moves and switches
        legal_moves = legal_actions[legal_actions < 4]
        legal_switches = legal_actions[legal_actions >= 4]
        
        # Prefer moves: use them 80% of the time if available
        if len(legal_moves) > 0:
            if self.rng.rand() < 0.8:
                # Pick random move among legal moves
                return int(self.rng.choice(legal_moves))
            elif len(legal_switches) > 0:
                # Occasionally switch if available
                return int(self.rng.choice(legal_switches))
            else:
                # Only moves available
                return int(self.rng.choice(legal_moves))
        else:
            # No moves available, switch
            if len(legal_switches) > 0:
                return int(self.rng.choice(legal_switches))
            else:
                # Fallback: pick any legal action
                return int(self.rng.choice(legal_actions))
