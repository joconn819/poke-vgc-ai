"""Shared test fixtures and utilities."""

import pytest
from src.utils.config import REGULATION_SV2024_1, RegulationConfig


@pytest.fixture
def regulation_config() -> RegulationConfig:
    """Provide a test regulation config."""
    return REGULATION_SV2024_1


@pytest.fixture
def test_team_preview():
    """Example team preview data for testing."""
    return {
        "player": [
            {
                "name": "Salamence",
                "level": 50,
                "gender": "M",
                "item": "Choice Scarf",
                "ability": "Intimidate",
                "tera_type": "Water",
                "nature": "Jolly",
                "evs": {"hp": 4, "atk": 252, "spd": 252},
                "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
                "moves": ["Earthquake", "Outrage", "Protect", "Superpower"],
            },
            {
                "name": "Torkoal",
                "level": 50,
                "gender": "F",
                "item": "Assault Vest",
                "ability": "Drought",
                "tera_type": "Fire",
                "nature": "Modest",
                "evs": {"hp": 252, "def": 4, "spa": 252},
                "moves": ["Heat Wave", "Protect", "Recover", "Earth Power"],
            },
            {
                "name": "Rillaboom",
                "level": 50,
                "item": "Life Orb",
                "ability": "Grassy Surge",
                "tera_type": "Grass",
            },
            {
                "name": "Landorus-Therian",
                "level": 50,
                "item": "Rocky Helmet",
                "ability": "Intimidate",
                "tera_type": "Ground",
            },
            {
                "name": "Incineroar",
                "level": 50,
                "item": "Heavy-Duty Boots",
                "ability": "Intimidate",
                "tera_type": "Fire",
            },
            {
                "name": "Glastrier",
                "level": 50,
                "item": "Choice Band",
                "ability": "Chilling Neigh",
                "tera_type": "Ice",
            },
        ],
        "opponent": [
            {"name": "Kyogre", "level": 50},
            {"name": "Groudon", "level": 50},
            {"name": "Venusaur", "level": 50},
            {"name": "Blaziken", "level": 50},
            {"name": "Urshifu", "level": 50},
            {"name": "Calyrex", "level": 50},
        ]
    }
