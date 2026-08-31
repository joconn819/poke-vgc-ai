"""Abstract base class for Pokemon battle agents."""

from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """Abstract base class for agents that select actions in Pokemon battles.
    
    Agents work with Gymnasium interface where:
    - observation: continuous or discrete state representation
    - legal_actions_mask: binary mask indicating which actions are legal (1=legal, 0=illegal)
    - action: integer index into the legal action space
    """

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Select an action based on observation and legal actions.
        
        Args:
            observation: Current state observation (shape varies by format).
            legal_actions_mask: Binary mask of legal actions (1 for legal, 0 for illegal).
                Shape: (num_actions,)
        
        Returns:
            Action index (must be legal according to legal_actions_mask).
        """
        pass
