"""Tests for open-teamsheets integration."""

import pytest
from unittest.mock import patch, MagicMock
from src.belief.open_teamsheets import fetch_team_from_url, create_belief_state_from_teamsheet
from src.belief.belief_state import PokemonBeliefState, OpponentBeliefState


class TestOpenTeamsheets:
    """Tests for open-teamsheets API integration."""

    @patch("requests.get")
    def test_fetch_team_from_url_success(self, mock_get):
        """Test successful fetch from open-teamsheets URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "team": [
                {
                    "name": "Salamence",
                    "level": 50,
                    "item": "Choice Scarf",
                    "ability": "Intimidate",
                    "nature": "Jolly",
                    "moves": ["Earthquake", "Outrage", "Protect", "Superpower"],
                    "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
                    "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
                }
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = fetch_team_from_url("https://example.com/team")
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "Salamence"

    @patch("requests.get")
    def test_fetch_team_from_url_error(self, mock_get):
        """Test error handling when fetch fails."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_team_from_url("https://example.com/notfound")
        assert result is None

    @patch("requests.get")
    def test_fetch_team_from_url_invalid_json(self, mock_get):
        """Test error handling for invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        result = fetch_team_from_url("https://example.com/team")
        assert result is None

    @patch("requests.get")
    def test_fetch_team_from_url_network_error(self, mock_get):
        """Test error handling for network errors."""
        mock_get.side_effect = ConnectionError("Network error")

        result = fetch_team_from_url("https://example.com/team")
        assert result is None

    def test_create_belief_state_from_teamsheet_single_pokemon(self):
        """Test creating belief state from single Pokemon data."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "item": "Choice Scarf",
            "ability": "Intimidate",
            "moves": ["Earthquake", "Outrage", "Protect", "Superpower"],
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.name == "Salamence"
        assert belief.items == {"Choice Scarf"}
        assert belief.abilities == {"Intimidate"}
        assert belief.moves == {"Earthquake", "Outrage", "Protect", "Superpower"}

    def test_create_belief_state_from_teamsheet_minimal(self):
        """Test creating belief state from minimal Pokemon data."""
        pokemon_data = {
            "name": "Kyogre",
            "level": 50,
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.name == "Kyogre"
        assert belief.items == set()  # Unknown item
        assert belief.abilities == set()  # Unknown ability
        assert belief.moves == set()  # Unknown moves

    def test_create_belief_state_from_teamsheet_with_tera(self):
        """Test creating belief state with tera type."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "tera_type": "Water",
            "item": "Life Orb",
            "ability": "Intimidate",
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.tera_types == {"Water"}

    def test_create_belief_state_from_teamsheet_no_tera(self):
        """Test creating belief state without tera type (unknown)."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "item": "Life Orb",
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        # When tera type is not provided, it should be empty (unknown)
        assert belief.tera_types == set()

    def test_create_belief_state_from_teamsheet_multiple_abilities(self):
        """Test creating belief state with potential abilities."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "ability": "Intimidate",  # Known ability
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        # If ability is specified, only that ability should be in the set
        assert belief.abilities == {"Intimidate"}

    @patch("requests.get")
    def test_create_belief_state_from_url(self, mock_get):
        """Test creating complete belief state from URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "team": [
                {
                    "name": "Salamence",
                    "level": 50,
                    "item": "Choice Scarf",
                    "ability": "Intimidate",
                    "tera_type": "Water",
                    "moves": ["Earthquake", "Outrage"],
                },
                {
                    "name": "Groudon",
                    "level": 50,
                    "item": "Rocky Helmet",
                    "ability": "Drought",
                },
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # This would require creating a convenience function
        # For now, test the data flow manually
        team_data = mock_response.json()["team"]
        beliefs = {
            poke["name"]: create_belief_state_from_teamsheet(poke)
            for poke in team_data
        }

        assert len(beliefs) == 2
        assert "Salamence" in beliefs
        assert "Groudon" in beliefs
        assert beliefs["Salamence"].items == {"Choice Scarf"}

    def test_create_belief_state_from_teamsheet_preserves_moves(self):
        """Test that all moves are preserved without modification."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "moves": ["Earthquake", "Outrage", "Protect", "Superpower"],
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.moves == {"Earthquake", "Outrage", "Protect", "Superpower"}
        assert len(belief.moves) == 4

    def test_create_belief_state_handles_empty_moves_list(self):
        """Test handling empty moves list."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
            "moves": [],
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.moves == set()

    def test_create_belief_state_handles_missing_moves_key(self):
        """Test handling missing moves key."""
        pokemon_data = {
            "name": "Salamence",
            "level": 50,
        }

        belief = create_belief_state_from_teamsheet(pokemon_data)

        assert belief.moves == set()


class TestOpenTeamsheetsIntegration:
    """Integration tests for open-teamsheets workflow."""

    def test_full_workflow_from_data(self):
        """Test full workflow: data -> belief states."""
        teamsheet_data = [
            {
                "name": "Salamence",
                "level": 50,
                "item": "Choice Scarf",
                "ability": "Intimidate",
                "tera_type": "Water",
                "moves": ["Earthquake", "Outrage", "Protect"],
            },
            {
                "name": "Kyogre",
                "level": 50,
                "item": "Assault Vest",
                "ability": "Drizzle",
                "tera_type": "Water",
                "moves": ["Hydro Pump", "Protect"],
            },
        ]

        team = {}
        for poke_data in teamsheet_data:
            belief = create_belief_state_from_teamsheet(poke_data)
            team[belief.name] = belief

        opponent_belief = OpponentBeliefState(team=team)

        assert opponent_belief.team_size() == 2
        assert "Salamence" in opponent_belief.team
        assert "Kyogre" in opponent_belief.team

        # Verify Salamence belief
        salamence = opponent_belief.get_pokemon("Salamence")
        assert salamence is not None
        assert salamence.items == {"Choice Scarf"}
        assert salamence.moves == {"Earthquake", "Outrage", "Protect"}

        # Verify Kyogre belief
        kyogre = opponent_belief.get_pokemon("Kyogre")
        assert kyogre is not None
        assert kyogre.items == {"Assault Vest"}
        assert kyogre.abilities == {"Drizzle"}
