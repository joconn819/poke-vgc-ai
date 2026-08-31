"""Tests for configuration module."""

import pytest
from src.utils.config import RegulationConfig, Generation, REGULATION_SV2024_1


def test_regulation_config_creation(regulation_config: RegulationConfig):
    """Test that a regulation config can be created."""
    assert regulation_config.regulation_id == "sv2024-1"
    assert regulation_config.generation == Generation.GEN_9
    assert regulation_config.team_size == 6
    assert regulation_config.active_pokemon == 2
    assert regulation_config.is_vgc_doubles is True


def test_pokemon_legality(regulation_config: RegulationConfig):
    """Test Pokemon legality checking."""
    assert regulation_config.is_pokemon_legal("pikachu")
    assert regulation_config.is_pokemon_legal("salamence")
    assert not regulation_config.is_pokemon_legal("mew")
    assert not regulation_config.is_pokemon_legal("invalid-pokemon-name")


def test_item_legality(regulation_config: RegulationConfig):
    """Test item legality checking."""
    assert regulation_config.is_item_legal("choice-band")
    assert regulation_config.is_item_legal("life-orb")
    assert not regulation_config.is_item_legal("master-ball")


def test_move_legality(regulation_config: RegulationConfig):
    """Test move legality checking."""
    assert regulation_config.is_move_legal("earthquake")
    assert regulation_config.is_move_legal("protected")
    assert not regulation_config.is_move_legal("invalid-move")


def test_regulation_is_hashable():
    """Test that regulations can be used in sets/dicts for caching."""
    reg1 = REGULATION_SV2024_1
    reg2 = RegulationConfig(
        regulation_id="sv2024-1",
        generation=Generation.GEN_9,
    )
    # Same regulation ID should hash to same value
    assert hash(reg1) == hash(reg2)
    
    # Can be used in a set
    reg_set = {reg1, reg2}
    assert len(reg_set) == 2  # Different objects, even with same ID


def test_restricted_pokemon(regulation_config: RegulationConfig):
    """Test restricted Pokemon count constraints."""
    assert "groudon" in regulation_config.restricted_pokemon
    assert "kyogre" in regulation_config.restricted_pokemon
    assert "rayquaza" in regulation_config.restricted_pokemon
    assert regulation_config.max_restricted_per_team == 1
