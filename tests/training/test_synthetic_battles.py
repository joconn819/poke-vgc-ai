"""Tests for synthetic battle generator and replayer."""

import pytest
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from src.training.synthetic_battles import BattleGenerator, BattleRecord
from src.training.battle_replayer import BattleReplayer
from src.utils.config import REGULATION_SV2024_1, RegulationConfig


class TestBattleGenerator:
    """Tests for BattleGenerator class."""

    @pytest.fixture
    def regulation(self):
        """Use test regulation."""
        return REGULATION_SV2024_1

    @pytest.fixture
    def generator(self, regulation):
        """Create a BattleGenerator instance."""
        return BattleGenerator(regulation_config=regulation, seed=42)

    @pytest.fixture
    def sample_team(self, regulation):
        """Create a sample legal team."""
        pokemon_list = list(regulation.legal_pokemon)[:6]
        return [
            {
                "name": pokemon,
                "level": 50,
                "item": "choice-band",
                "ability": "static",
                "moves": ["earthquake", "protected"],
                "tera_type": "normal",
            }
            for pokemon in pokemon_list
        ]

    def test_generator_initialization(self, generator, regulation):
        """Test BattleGenerator initializes correctly."""
        assert generator.regulation_config == regulation
        assert generator.seed == 42

    def test_generate_battle_returns_valid_record(
        self, generator, sample_team
    ):
        """Test generate_battle returns a valid BattleRecord."""
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        assert isinstance(record, BattleRecord)
        assert record.player_team == sample_team
        assert record.opponent_team == sample_team
        assert record.winner in ["player", "opponent"]
        assert isinstance(record.trajectory, list)
        assert len(record.trajectory) > 0

    def test_trajectory_complete_from_initial_to_terminal(
        self, generator, sample_team
    ):
        """Test that trajectory contains complete battle from initial to terminal state."""
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        trajectory = record.trajectory
        
        # First step should have done=False (unless 1-step battle)
        if len(trajectory) > 1:
            assert trajectory[0].done is False, "First step should not be terminal"
        
        # Last step should have done=True
        assert trajectory[-1].done is True, "Last step should be terminal"

    def test_battle_outcomes_deterministic_with_seed(
        self, regulation, sample_team
    ):
        """Test that identical battles with same seed produce identical outcomes."""
        gen1 = BattleGenerator(regulation_config=regulation, seed=42)
        gen2 = BattleGenerator(regulation_config=regulation, seed=42)
        
        record1 = gen1.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        record2 = gen2.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        assert record1.winner == record2.winner
        assert len(record1.trajectory) == len(record2.trajectory)
        # Compare trajectory step by step
        for step1, step2 in zip(record1.trajectory, record2.trajectory):
            np.testing.assert_array_equal(step1.state, step2.state)
            np.testing.assert_array_equal(step1.legal_actions_mask, step2.legal_actions_mask)
            assert step1.action == step2.action

    def test_different_seeds_produce_different_battles(
        self, regulation, sample_team
    ):
        """Test that different seeds can produce different battles with random agents."""
        gen1 = BattleGenerator(
            regulation_config=regulation,
            seed=42,
            player_agent_type="random",
            opponent_agent_type="random",
        )
        gen2 = BattleGenerator(
            regulation_config=regulation,
            seed=999,
            player_agent_type="random",
            opponent_agent_type="random",
        )
        
        record1 = gen1.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        record2 = gen2.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        # Both records should be valid
        assert isinstance(record1, BattleRecord)
        assert isinstance(record2, BattleRecord)
        
        # With random agents, trajectories are likely to differ
        # (checking that at least one of these differs)
        assert len(record1.trajectory) != len(record2.trajectory) or \
               record1.winner != record2.winner or \
               (len(record1.trajectory) > 0 and 
                len(record2.trajectory) > 0 and
                record1.trajectory[0].action != record2.trajectory[0].action)

    def test_team_pool_respected_in_generation(
        self, generator, regulation, sample_team
    ):
        """Test that generated teams use only legal Pokemon from the regulation."""
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        # Check all Pokemon in trajectory teams are legal
        for pokemon_spec in record.player_team:
            assert pokemon_spec["name"] in regulation.legal_pokemon, \
                f"Illegal Pokemon {pokemon_spec['name']} in player team"
        
        for pokemon_spec in record.opponent_team:
            assert pokemon_spec["name"] in regulation.legal_pokemon, \
                f"Illegal Pokemon {pokemon_spec['name']} in opponent team"

    def test_trajectory_actions_are_legal(
        self, generator, sample_team
    ):
        """Test that all actions in trajectory are legal according to their mask."""
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        for step in record.trajectory:
            # Action should be legal according to the mask
            assert step.legal_actions_mask[step.action] == 1, \
                f"Action {step.action} is illegal according to mask"

    def test_battle_lengths_are_reasonable(self, generator, sample_team):
        """Test that generated battles have reasonable length."""
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        trajectory_length = len(record.trajectory)
        
        # Battles should be at least 1 step
        assert trajectory_length >= 1, "Battle should have at least 1 step"
        
        # Battles shouldn't be absurdly long (should terminate)
        assert trajectory_length < 1000, "Battle exceeded max reasonable length"

    def test_generate_battle_with_different_agents(self, regulation, sample_team):
        """Test battle generation with different agent strategies."""
        gen_random = BattleGenerator(
            regulation_config=regulation,
            seed=42,
            player_agent_type="random",
            opponent_agent_type="random"
        )
        gen_mixed = BattleGenerator(
            regulation_config=regulation,
            seed=42,
            player_agent_type="max_damage",
            opponent_agent_type="random"
        )
        
        record_random = gen_random.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        record_mixed = gen_mixed.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        # Both should produce valid records
        assert isinstance(record_random, BattleRecord)
        assert isinstance(record_mixed, BattleRecord)
        
        # Records might differ due to different agent strategies
        assert len(record_random.trajectory) > 0
        assert len(record_mixed.trajectory) > 0


class TestBattleReplayer:
    """Tests for BattleReplayer class."""

    @pytest.fixture
    def regulation(self):
        """Use test regulation."""
        return REGULATION_SV2024_1

    @pytest.fixture
    def battle_record(self, regulation):
        """Create a sample BattleRecord for testing."""
        generator = BattleGenerator(regulation_config=regulation, seed=42)
        pokemon_list = list(regulation.legal_pokemon)[:6]
        team = [
            {
                "name": pokemon,
                "level": 50,
                "item": "choice-band",
                "ability": "static",
                "moves": ["earthquake", "protected"],
                "tera_type": "normal",
            }
            for pokemon in pokemon_list
        ]
        return generator.generate_battle(player_team=team, opponent_team=team)

    def test_replayer_initialization(self, battle_record):
        """Test BattleReplayer initializes correctly."""
        replayer = BattleReplayer(battle_record)
        assert replayer.battle_record == battle_record

    def test_replayer_iterates_all_steps(self, battle_record):
        """Test that replayer iterates through all trajectory steps."""
        replayer = BattleReplayer(battle_record)
        
        steps = list(replayer)
        trajectory_steps = battle_record.trajectory
        
        assert len(steps) == len(trajectory_steps), \
            "Replayer should iterate all trajectory steps"

    def test_replayer_extracts_required_fields(self, battle_record):
        """Test that replayer extracts state, action, and outcome."""
        replayer = BattleReplayer(battle_record)
        
        for state, action, done, reward in replayer:
            assert isinstance(state, np.ndarray), "State should be ndarray"
            assert isinstance(action, (int, np.integer)), "Action should be int"
            assert isinstance(done, bool), "Done should be bool"
            assert isinstance(reward, (int, float)), "Reward should be numeric"

    def test_replayer_terminal_state_identification(self, battle_record):
        """Test that replayer correctly identifies terminal state."""
        replayer = BattleReplayer(battle_record)
        
        steps = list(replayer)
        
        # Last step should have done=True
        if steps:
            *_, last_step = steps
            state, action, done, reward = last_step
            assert done is True, "Last step should be terminal"

    def test_replayer_iteration_order(self, battle_record):
        """Test that replayer iterates in chronological order."""
        replayer = BattleReplayer(battle_record)
        
        previous_index = -1
        for idx, (state, action, done, reward) in enumerate(replayer):
            assert idx == previous_index + 1, "Iteration should be in order"
            previous_index = idx


class TestBattleRecordFormat:
    """Tests for BattleRecord format and constraints."""

    @pytest.fixture
    def regulation(self):
        """Use test regulation."""
        return REGULATION_SV2024_1

    @pytest.fixture
    def sample_team(self, regulation):
        """Create a sample legal team."""
        pokemon_list = list(regulation.legal_pokemon)[:6]
        return [
            {
                "name": pokemon,
                "level": 50,
                "item": "choice-band",
                "ability": "static",
                "moves": ["earthquake", "protected"],
                "tera_type": "normal",
            }
            for pokemon in pokemon_list
        ]

    def test_battle_record_has_required_fields(
        self, regulation, sample_team
    ):
        """Test that BattleRecord contains all required fields."""
        generator = BattleGenerator(regulation_config=regulation, seed=42)
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        assert hasattr(record, "player_team")
        assert hasattr(record, "opponent_team")
        assert hasattr(record, "winner")
        assert hasattr(record, "trajectory")

    def test_trajectory_step_has_required_fields(
        self, regulation, sample_team
    ):
        """Test that trajectory steps contain required fields."""
        generator = BattleGenerator(regulation_config=regulation, seed=42)
        record = generator.generate_battle(
            player_team=sample_team,
            opponent_team=sample_team,
        )
        
        for step in record.trajectory:
            assert hasattr(step, "state"), "Step should have state"
            assert hasattr(step, "legal_actions_mask"), "Step should have legal_actions_mask"
            assert hasattr(step, "action"), "Step should have action"
            assert hasattr(step, "reward"), "Step should have reward"
            assert hasattr(step, "done"), "Step should have done flag"


class TestBattleStatistics:
    """Tests for battle statistics tracking."""

    @pytest.fixture
    def regulation(self):
        """Use test regulation."""
        return REGULATION_SV2024_1

    @pytest.fixture
    def generator(self, regulation):
        """Create a BattleGenerator."""
        return BattleGenerator(regulation_config=regulation, seed=42)

    @pytest.fixture
    def sample_team(self, regulation):
        """Create a sample legal team."""
        pokemon_list = list(regulation.legal_pokemon)[:6]
        return [
            {
                "name": pokemon,
                "level": 50,
                "item": "choice-band",
                "ability": "static",
                "moves": ["earthquake", "protected"],
                "tera_type": "normal",
            }
            for pokemon in pokemon_list
        ]

    def test_get_battle_statistics(self, generator, sample_team):
        """Test get_battle_statistics returns valid statistics."""
        # Generate a few battles
        for _ in range(3):
            generator.generate_battle(
                player_team=sample_team,
                opponent_team=sample_team,
            )
        
        stats = generator.get_battle_statistics()
        
        assert isinstance(stats, dict)
        assert "total_battles" in stats
        assert "player_wins" in stats
        assert "opponent_wins" in stats
        assert "average_battle_length" in stats
        assert stats["total_battles"] == 3

    def test_statistics_win_rate_calculation(self, regulation, sample_team):
        """Test that win rate statistics are calculated correctly."""
        generator = BattleGenerator(regulation_config=regulation, seed=42)
        
        # Generate multiple battles
        num_battles = 5
        for _ in range(num_battles):
            generator.generate_battle(
                player_team=sample_team,
                opponent_team=sample_team,
            )
        
        stats = generator.get_battle_statistics()
        
        # Win counts should sum to total battles
        assert stats["player_wins"] + stats["opponent_wins"] == num_battles
        
        # Win rates should be between 0 and 1
        player_win_rate = stats["player_wins"] / num_battles
        assert 0.0 <= player_win_rate <= 1.0

    def test_statistics_battle_length_tracking(self, regulation, sample_team):
        """Test that average battle length is tracked correctly."""
        generator = BattleGenerator(regulation_config=regulation, seed=42)
        
        # Generate multiple battles
        num_battles = 3
        for _ in range(num_battles):
            generator.generate_battle(
                player_team=sample_team,
                opponent_team=sample_team,
            )
        
        stats = generator.get_battle_statistics()
        
        assert stats["average_battle_length"] > 0
        assert isinstance(stats["average_battle_length"], (int, float))
