"""Baseline evaluation runner for comparing agents."""

from dataclasses import dataclass
import numpy as np
from src.baselines.base_agent import Agent


@dataclass
class BattleResult:
    """Results from running multiple battles between two agents."""
    
    agent1_wins: int
    agent2_wins: int
    total_battles: int
    agent1_moves_used: int
    agent2_moves_used: int
    agent1_switches_used: int
    agent2_switches_used: int
    
    def agent1_win_rate(self) -> float:
        """Get agent1's win rate."""
        if self.total_battles == 0:
            return 0.0
        return self.agent1_wins / self.total_battles
    
    def agent2_win_rate(self) -> float:
        """Get agent2's win rate."""
        if self.total_battles == 0:
            return 0.0
        return self.agent2_wins / self.total_battles


def run_battles(
    agent1: Agent,
    agent2: Agent,
    num_battles: int = 1,
    seed: int = 42,
) -> BattleResult:
    """Run multiple battles between two agents.
    
    This is a simplified simulation that:
    - Generates random game states
    - Tracks which agent makes more aggressive moves
    - Uses a simple heuristic to determine battle outcomes
    
    Args:
        agent1: First agent.
        agent2: Second agent.
        num_battles: Number of battles to simulate.
        seed: Random seed for reproducibility.
    
    Returns:
        BattleResult with win rates and statistics.
    """
    rng = np.random.RandomState(seed)
    
    agent1_wins = 0
    agent2_wins = 0
    total_agent1_moves = 0
    total_agent2_moves = 0
    total_agent1_switches = 0
    total_agent2_switches = 0
    
    for _ in range(num_battles):
        # Simulate a simple battle
        agent1_battle_moves = 0
        agent2_battle_moves = 0
        agent1_battle_switches = 0
        agent2_battle_switches = 0
        
        # Run a few turns of the battle
        max_turns = rng.randint(5, 15)  # Random battle length
        
        for turn in range(max_turns):
            # Generate random legal action masks for both agents
            # In a real battle, these would come from the environment
            action_space_size = 6  # 4 moves + 2 switches
            agent1_mask = _generate_legal_actions_mask(rng, action_space_size)
            agent2_mask = _generate_legal_actions_mask(rng, action_space_size)
            
            # Get observations (simplified - all zeros)
            observation = np.zeros(10, dtype=np.float32)
            
            # Get agent actions
            agent1_action = agent1.predict(observation, agent1_mask)
            agent2_action = agent2.predict(observation, agent2_mask)
            
            # Track move vs switch usage
            if agent1_action < 4:
                agent1_battle_moves += 1
            else:
                agent1_battle_switches += 1
            
            if agent2_action < 4:
                agent2_battle_moves += 1
            else:
                agent2_battle_switches += 1
            
            # Simple heuristic: agent that uses more moves wins more
            # (This is a simplified battle outcome determination)
        
        # Determine battle winner based on move usage and randomness
        agent1_move_score = agent1_battle_moves + 0.5 * agent1_battle_switches
        agent2_move_score = agent2_battle_moves + 0.5 * agent2_battle_switches
        
        # Add randomness for variability
        agent1_move_score += rng.randn() * 0.1
        agent2_move_score += rng.randn() * 0.1
        
        if agent1_move_score > agent2_move_score:
            agent1_wins += 1
        else:
            agent2_wins += 1
        
        total_agent1_moves += agent1_battle_moves
        total_agent2_moves += agent2_battle_moves
        total_agent1_switches += agent1_battle_switches
        total_agent2_switches += agent2_battle_switches
    
    return BattleResult(
        agent1_wins=agent1_wins,
        agent2_wins=agent2_wins,
        total_battles=num_battles,
        agent1_moves_used=total_agent1_moves,
        agent2_moves_used=total_agent2_moves,
        agent1_switches_used=total_agent1_switches,
        agent2_switches_used=total_agent2_switches,
    )


def _generate_legal_actions_mask(
    rng: np.random.RandomState,
    action_space_size: int,
) -> np.ndarray:
    """Generate a random legal action mask.
    
    Args:
        rng: Random number generator.
        action_space_size: Total number of actions.
    
    Returns:
        Binary mask with at least one action legal.
    """
    mask = (rng.rand(action_space_size) > 0.3).astype(np.float32)
    
    # Ensure at least one action is legal
    if mask.sum() == 0:
        mask[rng.randint(action_space_size)] = 1
    
    return mask
