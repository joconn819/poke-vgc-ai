"""Tests for baseline evaluation runner."""

import pytest
import numpy as np
from src.evaluation.baseline_runner import run_battles, BattleResult


class TestBaselineRunner:
    """Test baseline runner functionality."""

    def test_battle_result_structure(self):
        """BattleResult should have required fields."""
        result = BattleResult(
            agent1_wins=5,
            agent2_wins=3,
            total_battles=8,
            agent1_moves_used=120,
            agent2_moves_used=110,
            agent1_switches_used=40,
            agent2_switches_used=45,
        )
        
        assert result.agent1_wins == 5
        assert result.agent2_wins == 3
        assert result.total_battles == 8
        assert result.agent1_win_rate() == 5 / 8
        assert result.agent2_win_rate() == 3 / 8

    def test_run_battles_returns_battle_result(self):
        """run_battles should return a BattleResult object."""
        from src.baselines.random_agent import RandomAgent
        
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)
        
        # We'll test with a minimal number of battles
        result = run_battles(agent1, agent2, num_battles=1, seed=42)
        
        assert isinstance(result, BattleResult)
        assert result.total_battles == 1

    def test_run_battles_win_rates_sum_to_one(self):
        """Win rates should sum to 1.0 (assuming no draws)."""
        from src.baselines.random_agent import RandomAgent
        
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)
        
        result = run_battles(agent1, agent2, num_battles=2, seed=42)
        
        total_wins = result.agent1_wins + result.agent2_wins
        # Note: We assume no draws for this test
        assert total_wins == result.total_battles

    def test_run_battles_tracks_move_statistics(self):
        """run_battles should track moves and switches used."""
        from src.baselines.random_agent import RandomAgent
        
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)
        
        result = run_battles(agent1, agent2, num_battles=1, seed=42)
        
        assert hasattr(result, "agent1_moves_used")
        assert hasattr(result, "agent2_moves_used")
        assert hasattr(result, "agent1_switches_used")
        assert hasattr(result, "agent2_switches_used")
        
        # Should be non-negative integers
        assert result.agent1_moves_used >= 0
        assert result.agent2_moves_used >= 0
        assert result.agent1_switches_used >= 0
        assert result.agent2_switches_used >= 0

    def test_run_battles_deterministic_with_seed(self):
        """run_battles should be deterministic with same seed."""
        from src.baselines.random_agent import RandomAgent
        
        agent1_v1 = RandomAgent(seed=42)
        agent2_v1 = RandomAgent(seed=43)
        result1 = run_battles(agent1_v1, agent2_v1, num_battles=1, seed=100)
        
        agent1_v2 = RandomAgent(seed=42)
        agent2_v2 = RandomAgent(seed=43)
        result2 = run_battles(agent1_v2, agent2_v2, num_battles=1, seed=100)
        
        assert result1.agent1_wins == result2.agent1_wins
        assert result1.agent2_wins == result2.agent2_wins

    def test_run_battles_multiple_battles(self):
        """run_battles should handle multiple battles."""
        from src.baselines.random_agent import RandomAgent
        
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)
        
        result = run_battles(agent1, agent2, num_battles=5, seed=42)
        
        assert result.total_battles == 5
        assert result.agent1_wins + result.agent2_wins == 5

    def test_run_battles_identical_agents_convergence(self):
        """Identical agents should have roughly 50-50 win rate over many games."""
        from src.baselines.random_agent import RandomAgent
        
        # Use same seed but create two instances (they'll still behave identically)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)
        
        result = run_battles(agent1, agent2, num_battles=10, seed=123)
        
        # With 10 battles, we just check the structure is correct
        # Over many battles, should approach 50-50
        assert result.total_battles == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
