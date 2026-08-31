"""Tests for configuration module."""

import pytest
from src.utils.config import (
    RegulationConfig, Generation, REGULATION_SV2024_1,
    REGULATION_CHAMPIONS_MB
)


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


# Champions MB Tests (Sword/Shield era, Level 50 Doubles)
class TestChampionsMB:
    """Test suite for Champions MB regulation configuration."""
    
    def test_champions_mb_exists(self):
        """Test that Champions MB regulation is available."""
        assert REGULATION_CHAMPIONS_MB is not None
        assert REGULATION_CHAMPIONS_MB.regulation_id == "champions-mb"
        
    def test_champions_mb_generation(self):
        """Test Champions MB is Gen 8 (Sword/Shield)."""
        # Champions MB is based on Galar dex + DLC
        # Gen 8 is Sword/Shield
        assert REGULATION_CHAMPIONS_MB.generation == Generation.GEN_8
        
    def test_champions_mb_level_50_format(self):
        """Test Champions MB uses Level 50 format."""
        assert REGULATION_CHAMPIONS_MB.min_level == 50
        assert REGULATION_CHAMPIONS_MB.max_level == 50
        
    def test_champions_mb_level_50_doubles(self):
        """Test Champions MB is doubles format."""
        assert REGULATION_CHAMPIONS_MB.is_vgc_doubles is True
        assert REGULATION_CHAMPIONS_MB.active_pokemon == 2
        assert REGULATION_CHAMPIONS_MB.team_size == 6
        
    def test_champions_mb_legal_pokemon_count(self):
        """Test that Champions MB has 400+ legal Pokemon."""
        assert len(REGULATION_CHAMPIONS_MB.legal_pokemon) >= 400
        
    def test_champions_mb_specific_legal_pokemon(self):
        """Test specific Pokemon that should be legal in Champions MB."""
        # Galar starter evolutions
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("grookey")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("rillaboom")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("scorbunny")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("cinderace")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("sobble")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("inteleon")
        
        # Galar-exclusive/common competitive Pokemon
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("corviknight")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("duraludon")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("dragapult")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("dracozolt")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("arctozolt")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("flapple")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("appletun")
        
        # DLC additions (Isle of Armor / Crown Tundra)
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("urshifu")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("calyrex")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("glastrier")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("spectrier")
        
        # Classic Legendaries available in Crown Tundra
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("landorus")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("tornadus")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("kyogre")
        assert REGULATION_CHAMPIONS_MB.is_pokemon_legal("groudon")
        
    def test_champions_mb_excluded_pokemon(self):
        """Test Pokemon that should NOT be legal in Champions MB."""
        # Mythical Pokemon that were never available
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("mew")
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("phione")  # Can't breed in Galar
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("manaphy")
        
        # Gen 9+ Pokemon (Scarlet/Violet)
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("sprigatito")
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("fuecoco")
        assert not REGULATION_CHAMPIONS_MB.is_pokemon_legal("quaxly")
        
    def test_champions_mb_restricted_pokemon(self):
        """Test Champions MB restricted Pokemon list."""
        # The restricted Pokemon for Champions MB includes box legends
        # and powerful legendaries
        assert len(REGULATION_CHAMPIONS_MB.restricted_pokemon) > 0
        assert "zacian" in REGULATION_CHAMPIONS_MB.restricted_pokemon or \
               "zamazenta" in REGULATION_CHAMPIONS_MB.restricted_pokemon
        
    def test_champions_mb_max_restricted_per_team(self):
        """Test max restricted Pokemon per team."""
        assert REGULATION_CHAMPIONS_MB.max_restricted_per_team == 1
        
    def test_champions_mb_legal_items(self):
        """Test Champions MB legal items."""
        # Common competitive items
        assert REGULATION_CHAMPIONS_MB.is_item_legal("choice-band")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("choice-scarf")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("choice-specs")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("life-orb")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("assault-vest")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("rocky-helmet")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("air-balloon")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("heavy-duty-boots")
        
        # Gen 8 specific items
        assert REGULATION_CHAMPIONS_MB.is_item_legal("weakness-policy")
        assert REGULATION_CHAMPIONS_MB.is_item_legal("eject-pack")
        
        # Banned items
        assert not REGULATION_CHAMPIONS_MB.is_item_legal("master-ball")
        
    def test_champions_mb_legal_moves(self):
        """Test Champions MB legal moves."""
        # Standard coverage moves
        assert REGULATION_CHAMPIONS_MB.is_move_legal("earthquake")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("protect")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("surf")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("thunderbolt")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("ice-beam")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("shadow-ball")
        assert REGULATION_CHAMPIONS_MB.is_move_legal("focus-blast")
        
    def test_champions_mb_legal_abilities(self):
        """Test Champions MB legal abilities."""
        # Standard abilities
        assert REGULATION_CHAMPIONS_MB.legal_abilities is not None
        assert len(REGULATION_CHAMPIONS_MB.legal_abilities) > 0
        assert "intimidate" in REGULATION_CHAMPIONS_MB.legal_abilities or \
               "drought" in REGULATION_CHAMPIONS_MB.legal_abilities
        
    def test_champions_mb_no_tera_types(self):
        """Test that Champions MB does not include Tera types (Gen 8)."""
        # Terastallization was introduced in Gen 9, so shouldn't be in Gen 8 config
        # Either empty or not present
        assert (len(REGULATION_CHAMPIONS_MB.legal_tera_types) == 0 or
                REGULATION_CHAMPIONS_MB.legal_tera_types is None)
        
    def test_champions_mb_hashable_and_cacheable(self):
        """Test that Champions MB config is hashable for caching."""
        # Should be able to use in dict/set
        config_cache = {REGULATION_CHAMPIONS_MB: "cached_value"}
        assert config_cache[REGULATION_CHAMPIONS_MB] == "cached_value"
        
        # Hash should be consistent
        hash1 = hash(REGULATION_CHAMPIONS_MB)
        hash2 = hash(REGULATION_CHAMPIONS_MB)
        assert hash1 == hash2
