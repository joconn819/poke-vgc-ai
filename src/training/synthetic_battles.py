"""Synthetic battle generator for creating training data.

This module generates Pokemon VGC doubles battles using heuristic agents
to produce realistic training trajectories for supervised learning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from src.utils.config import RegulationConfig
from src.env.vgc_env import VGCDoublesEnv
from src.baselines.base_agent import Agent
from src.baselines.random_agent import RandomAgent
from src.baselines.max_damage_agent import MaxDamageAgent


@dataclass
class TrajectoryStep:
    """Single step in a battle trajectory.
    
    Attributes:
        state: Observation at this step (encoded battle state).
        legal_actions_mask: Binary mask of legal actions at this step.
        action: Action taken by the agent.
        reward: Immediate reward for taking this action.
        done: Whether the episode terminated after this step.
    """
    state: np.ndarray
    legal_actions_mask: np.ndarray
    action: int
    reward: float
    done: bool


@dataclass
class BattleRecord:
    """Complete record of a generated battle.
    
    Attributes:
        player_team: Player's Pokemon team specifications.
        opponent_team: Opponent's Pokemon team specifications.
        winner: "player" if player won, "opponent" if opponent won.
        trajectory: List of TrajectoryStep objects from battle start to end.
    """
    player_team: List[Dict[str, Any]]
    opponent_team: List[Dict[str, Any]]
    winner: str
    trajectory: List[TrajectoryStep]


class BattleGenerator:
    """Generates synthetic Pokemon battles for training data.
    
    Uses VGCDoublesEnv with heuristic agents (MaxDamageAgent, RandomAgent)
    to simulate battles and record complete trajectories for supervised learning.
    """

    def __init__(
        self,
        regulation_config: RegulationConfig,
        seed: Optional[int] = None,
        player_agent_type: str = "max_damage",
        opponent_agent_type: str = "random",
        max_steps: int = 500,
    ):
        """Initialize BattleGenerator.
        
        Args:
            regulation_config: VGC regulation defining legal teams.
            seed: Random seed for reproducibility.
            player_agent_type: Type of agent for player ("random" or "max_damage").
            opponent_agent_type: Type of agent for opponent ("random" or "max_damage").
            max_steps: Maximum steps per battle before forced termination.
        """
        self.regulation_config = regulation_config
        self.seed = seed
        self.player_agent_type = player_agent_type
        self.opponent_agent_type = opponent_agent_type
        self.max_steps = max_steps
        
        # Random number generator for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Statistics tracking
        self._battle_count = 0
        self._player_wins = 0
        self._opponent_wins = 0
        self._total_steps = 0

    def _get_agent(self, agent_type: str, seed: Optional[int] = None) -> Agent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent ("random" or "max_damage").
            seed: Random seed for the agent.
        
        Returns:
            Agent instance.
        """
        if agent_type == "random":
            return RandomAgent(seed=seed)
        elif agent_type == "max_damage":
            return MaxDamageAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def generate_battle(
        self,
        player_team: List[Dict[str, Any]],
        opponent_team: List[Dict[str, Any]],
    ) -> BattleRecord:
        """Generate a single battle between two teams.
        
        Args:
            player_team: Player's team specification (list of Pokemon dicts).
            opponent_team: Opponent's team specification.
        
        Returns:
            BattleRecord containing the complete battle trajectory.
        """
        # Create environment
        env = VGCDoublesEnv(
            regulation_config=self.regulation_config,
            player_team=player_team,
            opponent_team=opponent_team,
        )
        
        # Create agents with different seeds to ensure different behavior
        player_seed = None if self.seed is None else self.seed + self._battle_count * 2
        opponent_seed = None if self.seed is None else self.seed + self._battle_count * 2 + 1
        
        player_agent = self._get_agent(self.player_agent_type, seed=player_seed)
        opponent_agent = self._get_agent(self.opponent_agent_type, seed=opponent_seed)
        
        # Run battle
        trajectory = []
        observation, info = env.reset()
        
        for step_count in range(self.max_steps):
            legal_mask, _ = env._get_legal_moves()
            
            # Player action
            player_action = player_agent.predict(observation, legal_mask)
            
            # Opponent action (using same observation for simplicity)
            # In a real implementation, would maintain separate belief states
            opponent_action = opponent_agent.predict(observation, legal_mask)
            
            # Take step (here we simulate with player action for simplicity)
            # The environment would normally handle simultaneous action resolution
            action = player_action
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record trajectory step
            step = TrajectoryStep(
                state=observation.copy(),
                legal_actions_mask=legal_mask.astype(np.uint8),
                action=action,
                reward=float(reward),
                done=done,
            )
            trajectory.append(step)
            
            if done:
                break
        
        # Determine winner
        winner = "player" if reward > 0 else "opponent"
        
        # Update statistics
        self._battle_count += 1
        self._total_steps += len(trajectory)
        if winner == "player":
            self._player_wins += 1
        else:
            self._opponent_wins += 1
        
        # Create and return battle record
        record = BattleRecord(
            player_team=player_team,
            opponent_team=opponent_team,
            winner=winner,
            trajectory=trajectory,
        )
        
        return record

    def get_battle_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated battles.
        
        Returns:
            Dictionary with statistics:
            - total_battles: Number of battles generated.
            - player_wins: Number of battles won by player.
            - opponent_wins: Number of battles won by opponent.
            - average_battle_length: Average number of steps per battle.
        """
        avg_length = (
            self._total_steps / self._battle_count
            if self._battle_count > 0
            else 0
        )
        
        return {
            "total_battles": self._battle_count,
            "player_wins": self._player_wins,
            "opponent_wins": self._opponent_wins,
            "average_battle_length": avg_length,
        }
