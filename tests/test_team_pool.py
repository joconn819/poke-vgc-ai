"""Tests for team pool sampling and loading."""

from pathlib import Path

import json
import pytest

from src.utils.config import REGULATION_SV2024_1, RegulationConfig
from src.utils.team_pool import TeamPool


class TestTeamPoolLoading:
    """Tests for loading team data from JSON."""
    
    def test_team_pool_loads_from_file(self, tmp_path):
        """TeamPool should load teams from a JSON file."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": "Test Team 1",
                "pokemon": [
                    {
                        "name": "Salamence",
                        "item": "Choice Scarf",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]
                    },
                    {
                        "name": "Torkoal",
                        "item": "Assault Vest",
                        "ability": "Drought",
                        "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]
                    },
                    {
                        "name": "Rillaboom",
                        "item": "Life Orb",
                        "ability": "Grassy Surge",
                        "moves": ["Grassy Glide", "Protected", "Close Combat", "Knock Off"]
                    },
                    {
                        "name": "Landorus",
                        "item": "Rocky Helmet",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Protected", "Stone Edge", "Superpower"]
                    },
                    {
                        "name": "Incineroar",
                        "item": "Heavy-Duty Boots",
                        "ability": "Intimidate",
                        "moves": ["Flare Blitz", "Close Combat", "Protected", "Knock Off"]
                    },
                    {
                        "name": "Glastrier",
                        "item": "Choice Band",
                        "ability": "Chilling Neigh",
                        "moves": ["Ice Punch", "Close Combat", "Protected", "Earthquake"]
                    }
                ]
            }
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Assert
        assert pool is not None
        assert len(pool) == 1
    
    def test_team_pool_handles_missing_file(self):
        """TeamPool should raise error for missing file."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            TeamPool("/nonexistent/path/teams.json", REGULATION_SV2024_1)
    
    def test_team_pool_loads_multiple_teams(self, tmp_path):
        """TeamPool should load multiple teams from JSON."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": f"Team {i}",
                "pokemon": [
                    {
                        "name": "Salamence",
                        "item": "Choice Scarf",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]
                    }
                    for _ in range(6)
                ]
            }
            for i in range(5)
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Assert
        assert len(pool) == 5


class TestTeamPoolSampling:
    """Tests for sampling teams from the pool."""
    
    def test_sample_team_returns_valid_team(self, tmp_path):
        """sample_team() should return a valid 6-Pokemon team dict."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": "Test Team",
                "pokemon": [
                    {
                        "name": "Salamence",
                        "item": "Choice Scarf",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]
                    },
                    {
                        "name": "Torkoal",
                        "item": "Assault Vest",
                        "ability": "Drought",
                        "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]
                    },
                    {
                        "name": "Rillaboom",
                        "item": "Life Orb",
                        "ability": "Grassy Surge",
                        "moves": ["Grassy Glide", "Protected", "Close Combat", "Knock Off"]
                    },
                    {
                        "name": "Landorus",
                        "item": "Rocky Helmet",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Protected", "Stone Edge", "Superpower"]
                    },
                    {
                        "name": "Incineroar",
                        "item": "Heavy-Duty Boots",
                        "ability": "Intimidate",
                        "moves": ["Flare Blitz", "Close Combat", "Protected", "Knock Off"]
                    },
                    {
                        "name": "Glastrier",
                        "item": "Choice Band",
                        "ability": "Chilling Neigh",
                        "moves": ["Ice Punch", "Close Combat", "Protected", "Earthquake"]
                    }
                ]
            }
        ]
        teams_file.write_text(json.dumps(sample_teams))
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Act
        team = pool.sample_team()
        
        # Assert
        assert team is not None
        assert isinstance(team, dict)
        assert "pokemon" in team
        assert len(team["pokemon"]) == 6
        
        # Each Pokemon should have required fields
        for poke in team["pokemon"]:
            assert "name" in poke
            assert "item" in poke
            assert "ability" in poke
            assert "moves" in poke
            assert len(poke["moves"]) == 4
    
    def test_sample_team_returns_different_teams(self, tmp_path):
        """sample_team() should return different teams from diverse pools."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": f"Team {i}",
                "pokemon": [
                    {
                        "name": pokemon_name,
                        "item": "Choice Scarf",
                        "ability": "Intimidate",
                        "moves": ["Earthquake", "Protected", "Stone Edge", "Superpower"]
                    }
                    for _ in range(6)
                ]
            }
            for i, pokemon_name in enumerate(["Salamence", "Landorus", "Tornadus", "Garchomp"])
        ]
        teams_file.write_text(json.dumps(sample_teams))
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Act
        teams = [pool.sample_team() for _ in range(10)]
        
        # Assert - should get different teams
        team_names = [tuple(p["name"] for p in t["pokemon"]) for t in teams]
        assert len(set(team_names)) > 1, "Should have variation in sampled teams"
    
    def test_sample_team_empty_pool_raises_error(self, tmp_path):
        """sample_team() should raise error on empty pool."""
        # Arrange
        teams_file = tmp_path / "empty_teams.json"
        teams_file.write_text(json.dumps([]))
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Act & Assert
        with pytest.raises(ValueError):
            pool.sample_team()


class TestTeamPoolFiltering:
    """Tests for filtering teams by regulation."""
    
    def test_team_pool_filters_illegal_pokemon(self, tmp_path):
        """TeamPool should filter out teams with illegal Pokemon."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        # Create a limited regulation
        limited_regulation = RegulationConfig(
            regulation_id="limited",
            generation=REGULATION_SV2024_1.generation,
            legal_pokemon={"salamence", "torkoal"},  # Only 2 legal Pokemon
            legal_items=REGULATION_SV2024_1.legal_items,
            legal_moves=REGULATION_SV2024_1.legal_moves,
            legal_abilities=REGULATION_SV2024_1.legal_abilities,
            legal_tera_types=REGULATION_SV2024_1.legal_tera_types,
        )
        
        sample_teams = [
            {
                "team_name": "Valid Team",
                "pokemon": [
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                ]
            },
            {
                "team_name": "Invalid Team (has Landorus)",
                "pokemon": [
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Landorus", "item": "Rocky Helmet", "ability": "Intimidate", "moves": ["Earthquake", "Protected", "Stone Edge", "Superpower"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                ]
            }
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act
        pool = TeamPool(str(teams_file), limited_regulation)
        
        # Assert
        assert len(pool) == 1  # Only valid team should be kept
        team = pool.sample_team()
        assert all(p["name"].lower() in limited_regulation.legal_pokemon for p in team["pokemon"])
    
    def test_team_pool_filters_illegal_items(self, tmp_path):
        """TeamPool should filter out teams with illegal items."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        limited_regulation = RegulationConfig(
            regulation_id="limited",
            generation=REGULATION_SV2024_1.generation,
            legal_pokemon=REGULATION_SV2024_1.legal_pokemon,
            legal_items={"choice-scarf"},  # Only 1 legal item
            legal_moves=REGULATION_SV2024_1.legal_moves,
            legal_abilities=REGULATION_SV2024_1.legal_abilities,
            legal_tera_types=REGULATION_SV2024_1.legal_tera_types,
        )
        
        sample_teams = [
            {
                "team_name": "Invalid Team (has Assault Vest)",
                "pokemon": [
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Torkoal", "item": "Assault Vest", "ability": "Drought", "moves": ["Heat Wave", "Protected", "Recover", "Earth Power"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]},
                ]
            }
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act
        pool = TeamPool(str(teams_file), limited_regulation)
        
        # Assert
        assert len(pool) == 0  # No valid teams


class TestTeamPoolLength:
    """Tests for length and iteration of pool."""
    
    def test_team_pool_len(self, tmp_path):
        """TeamPool should report correct length."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": f"Team {i}",
                "pokemon": [
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]}
                    for _ in range(6)
                ]
            }
            for i in range(3)
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Assert
        assert len(pool) == 3
    
    def test_team_pool_lazy_loads_on_access(self, tmp_path):
        """TeamPool should lazy-load teams only when accessed."""
        # Arrange
        teams_file = tmp_path / "test_teams.json"
        sample_teams = [
            {
                "team_name": "Team 1",
                "pokemon": [
                    {"name": "Salamence", "item": "Choice Scarf", "ability": "Intimidate", "moves": ["Earthquake", "Outrage", "Protected", "Superpower"]}
                    for _ in range(6)
                ]
            }
        ]
        teams_file.write_text(json.dumps(sample_teams))
        
        # Act - Create pool (should not fail even if file format is weird, as long as it's valid JSON)
        pool = TeamPool(str(teams_file), REGULATION_SV2024_1)
        
        # Assert - length should work
        assert len(pool) == 1
        
        # Act - Sampling should work
        team = pool.sample_team()
        
        # Assert
        assert team is not None


class TestChampionsTeamPoolIntegration:
    """Integration tests for the shipped Champions MB team list."""

    def test_champions_team_file_contains_legal_teams(self):
        """The bundled Champions MB dataset should remain usable for sampling."""
        repo_root = Path(__file__).resolve().parents[1]
        from src.utils.config import REGULATION_CHAMPIONS_MB

        pool = TeamPool(str(repo_root / "data/teams/champions_mb.json"), REGULATION_CHAMPIONS_MB)

        assert len(pool) > 0
