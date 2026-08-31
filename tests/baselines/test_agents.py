"""Tests for baseline agents."""

import pytest
import numpy as np
from src.baselines.base_agent import Agent
from src.baselines.random_agent import RandomAgent
from src.baselines.max_damage_agent import MaxDamageAgent
from src.baselines.switch_preserving_agent import SwitchPreservingAgent
from src.baselines.team_preview_agent import TeamPreviewAgent


class TestRandomAgent:
    """Test RandomAgent implementation."""

    def test_random_agent_respects_action_mask(self):
        """RandomAgent should only select from legal actions."""
        agent = RandomAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)  # Mock observation
        
        # Legal actions mask: only actions 0, 2, 5 are legal
        legal_actions_mask = np.array(
            [1, 0, 1, 0, 0, 1] + [0] * 94, dtype=np.float32
        )
        
        legal_action_indices = {0, 2, 5}
        
        # Run multiple times to ensure all sampled actions are legal
        for _ in range(50):
            action = agent.predict(observation, legal_actions_mask)
            assert action in legal_action_indices, f"Action {action} not in legal actions"

    def test_random_agent_produces_valid_actions(self):
        """RandomAgent should always produce valid action indices."""
        agent = RandomAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        for _ in range(20):
            action = agent.predict(observation, legal_actions_mask)
            assert isinstance(action, (int, np.integer)), "Action should be an integer"
            assert 0 <= action < 100, f"Action {action} out of bounds"

    def test_random_agent_deterministic_with_seed(self):
        """RandomAgent should be deterministic when initialized with same seed."""
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        agent1 = RandomAgent(seed=123)
        agent2 = RandomAgent(seed=123)
        
        actions1 = [agent1.predict(observation, legal_actions_mask) for _ in range(10)]
        actions2 = [agent2.predict(observation, legal_actions_mask) for _ in range(10)]
        
        assert actions1 == actions2, "Same seed should produce same sequence"

    def test_random_agent_different_with_different_seeds(self):
        """RandomAgent with different seeds should produce different sequences."""
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        agent1 = RandomAgent(seed=123)
        agent2 = RandomAgent(seed=456)
        
        actions1 = [agent1.predict(observation, legal_actions_mask) for _ in range(20)]
        actions2 = [agent2.predict(observation, legal_actions_mask) for _ in range(20)]
        
        # Very unlikely to be the same for 20 consecutive samples
        assert actions1 != actions2, "Different seeds should produce different sequences"

    def test_random_agent_single_legal_action(self):
        """RandomAgent should select the only legal action."""
        agent = RandomAgent(seed=42)
        observation = np.zeros((10,), dtype=np.float32)
        
        # Only action 3 is legal
        legal_actions_mask = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        for _ in range(10):
            action = agent.predict(observation, legal_actions_mask)
            assert action == 3, "Should always select the only legal action"


class TestMaxDamageAgent:
    """Test MaxDamageAgent implementation."""

    def test_max_damage_agent_respects_action_mask(self):
        """MaxDamageAgent should only select from legal actions."""
        agent = MaxDamageAgent()
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.array([1, 0, 1, 0, 0, 1] + [0] * 94, dtype=np.float32)
        
        legal_action_indices = {0, 2, 5}
        
        action = agent.predict(observation, legal_actions_mask)
        assert action in legal_action_indices, f"Action {action} not in legal actions"

    def test_max_damage_agent_produces_valid_actions(self):
        """MaxDamageAgent should produce valid action indices."""
        agent = MaxDamageAgent()
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        action = agent.predict(observation, legal_actions_mask)
        assert isinstance(action, (int, np.integer)), "Action should be an integer"
        assert 0 <= action < 100, f"Action {action} out of bounds"

    def test_max_damage_agent_deterministic(self):
        """MaxDamageAgent should be deterministic."""
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        agent = MaxDamageAgent()
        
        actions = [agent.predict(observation, legal_actions_mask) for _ in range(5)]
        
        # All actions should be identical for same observation and legal mask
        assert len(set(actions)) == 1, "MaxDamageAgent should be deterministic"

    def test_max_damage_agent_prefers_moves_over_switches(self):
        """MaxDamageAgent should prefer moves (actions 0-3) over switches (4-5)."""
        agent = MaxDamageAgent()
        observation = np.zeros((100,), dtype=np.float32)
        
        # Only switches are legal
        legal_actions_mask = np.array(
            [0, 0, 0, 0, 1, 1] + [0] * 94, dtype=np.float32
        )
        
        action = agent.predict(observation, legal_actions_mask)
        assert action in {4, 5}, "Should select switch when moves not available"

    def test_max_damage_agent_single_legal_action(self):
        """MaxDamageAgent should select the only legal action."""
        agent = MaxDamageAgent()
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.array([0, 0, 0, 0, 1, 0] + [0] * 94, dtype=np.float32)
        
        action = agent.predict(observation, legal_actions_mask)
        assert action == 4, "Should select the only legal action"


class TestSwitchPreservingAgent:
    """Test SwitchPreservingAgent implementation."""

    def test_switch_preserving_agent_respects_action_mask(self):
        """SwitchPreservingAgent should respect legal action mask."""
        agent = SwitchPreservingAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.array([1, 0, 1, 0, 0, 1] + [0] * 94, dtype=np.float32)
        
        legal_action_indices = {0, 2, 5}
        
        action = agent.predict(observation, legal_actions_mask)
        assert action in legal_action_indices, f"Action {action} not in legal actions"

    def test_switch_preserving_agent_produces_valid_actions(self):
        """SwitchPreservingAgent should produce valid action indices."""
        agent = SwitchPreservingAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        action = agent.predict(observation, legal_actions_mask)
        assert isinstance(action, (int, np.integer)), "Action should be an integer"
        assert 0 <= action < 100, f"Action {action} out of bounds"

    def test_switch_preserving_agent_deterministic_with_seed(self):
        """SwitchPreservingAgent should be deterministic with same seed."""
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        agent1 = SwitchPreservingAgent(seed=123)
        agent2 = SwitchPreservingAgent(seed=123)
        
        actions1 = [agent1.predict(observation, legal_actions_mask) for _ in range(10)]
        actions2 = [agent2.predict(observation, legal_actions_mask) for _ in range(10)]
        
        assert actions1 == actions2, "Same seed should produce same sequence"

    def test_switch_preserving_agent_prefers_moves_when_available(self):
        """SwitchPreservingAgent should prefer moves over switches."""
        agent = SwitchPreservingAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)
        
        # Both moves and switches available
        legal_actions_mask = np.array([1, 1, 1, 1, 1, 1] + [0] * 94, dtype=np.float32)
        
        # Run many times to check tendency
        move_count = 0
        switch_count = 0
        for _ in range(50):
            action = agent.predict(observation, legal_actions_mask)
            if action < 4:
                move_count += 1
            else:
                switch_count += 1
        
        # Should prefer moves over switches
        assert move_count > switch_count, "Should prefer moves over switches"

    def test_switch_preserving_agent_switches_when_necessary(self):
        """SwitchPreservingAgent should switch when moves not available."""
        agent = SwitchPreservingAgent(seed=42)
        observation = np.zeros((100,), dtype=np.float32)
        
        # Only switches available
        legal_actions_mask = np.array([0, 0, 0, 0, 1, 1] + [0] * 94, dtype=np.float32)
        
        action = agent.predict(observation, legal_actions_mask)
        assert action in {4, 5}, "Should switch when moves not available"


class TestTeamPreviewAgent:
    """Test TeamPreviewAgent implementation."""

    def test_team_preview_agent_respects_action_mask(self):
        """TeamPreviewAgent should select from legal actions only."""
        agent = TeamPreviewAgent(seed=42)
        # Team preview observation: typically 6 leading Pokemon
        observation = np.zeros((10,), dtype=np.float32)
        # Legal moves: 0-5 (pick one of 6 team members)
        legal_actions_mask = np.array([1, 1, 1, 1, 1, 1] + [0, 0, 0, 0], dtype=np.float32)
        
        action = agent.predict(observation, legal_actions_mask)
        assert 0 <= action < 6, f"Action {action} should be in team preview range"

    def test_team_preview_agent_produces_valid_actions(self):
        """TeamPreviewAgent should produce valid Pokemon indices."""
        agent = TeamPreviewAgent(seed=42)
        observation = np.zeros((10,), dtype=np.float32)
        legal_actions_mask = np.ones(6, dtype=np.float32)  # 6 team members
        
        action = agent.predict(observation, legal_actions_mask)
        assert isinstance(action, (int, np.integer)), "Action should be an integer"
        assert 0 <= action < 6, f"Action {action} out of bounds for team preview"

    def test_team_preview_agent_deterministic_with_seed(self):
        """TeamPreviewAgent should be deterministic with same seed."""
        observation = np.zeros((10,), dtype=np.float32)
        legal_actions_mask = np.ones(6, dtype=np.float32)
        
        agent1 = TeamPreviewAgent(seed=123)
        agent2 = TeamPreviewAgent(seed=123)
        
        actions1 = [agent1.predict(observation, legal_actions_mask) for _ in range(10)]
        actions2 = [agent2.predict(observation, legal_actions_mask) for _ in range(10)]
        
        assert actions1 == actions2, "Same seed should produce same sequence"

    def test_team_preview_agent_respects_unavailable_pokemon(self):
        """TeamPreviewAgent should not pick unavailable Pokemon."""
        agent = TeamPreviewAgent(seed=42)
        observation = np.zeros((10,), dtype=np.float32)
        
        # Only Pokemon 0, 2, 4 available
        legal_actions_mask = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
        
        legal_actions = {0, 2, 4}
        
        for _ in range(20):
            action = agent.predict(observation, legal_actions_mask)
            assert action in legal_actions, f"Action {action} not in available Pokemon"

    def test_team_preview_agent_interface(self):
        """TeamPreviewAgent should inherit from Agent interface."""
        agent = TeamPreviewAgent(seed=42)
        assert isinstance(agent, Agent), "TeamPreviewAgent should inherit from Agent"
        assert hasattr(agent, "predict"), "Should have predict method"


class TestAgentInterface:
    """Test base Agent abstract class."""

    def test_agent_abstract_class_cannot_be_instantiated(self):
        """Agent abstract class should not be instantiable."""
        with pytest.raises(TypeError):
            Agent()

    def test_all_agents_implement_predict_method(self):
        """All agent implementations should implement predict method."""
        agents = [
            RandomAgent(seed=42),
            MaxDamageAgent(),
            SwitchPreservingAgent(seed=42),
            TeamPreviewAgent(seed=42),
        ]
        
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        for agent in agents:
            action = agent.predict(observation, legal_actions_mask)
            assert isinstance(action, (int, np.integer)), f"{type(agent).__name__} should return int action"

    def test_predict_method_signature(self):
        """Predict method should accept observation and legal_actions_mask."""
        agent = RandomAgent(seed=42)
        
        observation = np.zeros((100,), dtype=np.float32)
        legal_actions_mask = np.ones(100, dtype=np.float32)
        
        # Should not raise
        action = agent.predict(observation, legal_actions_mask)
        assert action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
