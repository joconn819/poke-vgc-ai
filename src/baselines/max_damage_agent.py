"""Agent that heuristically maximizes expected damage."""

import numpy as np
from src.baselines.base_agent import Agent


class MaxDamageAgent(Agent):
    """Selects actions to maximize expected damage heuristically.
    
    Strategy:
    - Prefer attacking moves (actions 0-3) over switching (actions 4-5)
    - Among legal actions, select the lowest-indexed one (simple heuristic)
    - If no moves available, switch to random legal Pokemon
    
    This agent uses a simple damage heuristic without full state understanding.
    """

    def predict(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Select action to maximize damage heuristically.
        
        Args:
            observation: Current state observation (unused for simple heuristic).
            legal_actions_mask: Binary mask of legal actions.
        
        Returns:
            Index of action that maximizes expected damage.
        """
        legal_actions = np.where(legal_actions_mask == 1)[0]
        
        # Prefer moves (indices 0-3) over switches (4-5)
        legal_moves = legal_actions[legal_actions < 4]
        
        if len(legal_moves) > 0:
            # Among legal moves, pick the first one
            # (This is a simple heuristic; could be improved with damage calculation)
            return int(legal_moves[0])
        else:
            # No moves available, pick a switch
            legal_switches = legal_actions[legal_actions >= 4]
            if len(legal_switches) > 0:
                return int(legal_switches[0])
            else:
                # Fallback: pick any legal action
                return int(legal_actions[0])
