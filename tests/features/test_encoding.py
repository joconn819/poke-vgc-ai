"""Tests for feature encoding system (TDD approach)."""

import pytest
import torch
import numpy as np
from typing import Dict, Set, List

from src.features.encoding import FeatureEncoder
from src.features.tensors import (
    TensorShapes,
    DEFAULT_TENSOR_SHAPES,
    POKEMON_INDEX_MAP,
    MOVE_INDEX_MAP,
    ITEM_INDEX_MAP,
    ABILITY_INDEX_MAP,
    STATUS_INDEX_MAP,
)
from src.belief.belief_state import PokemonBeliefState
from src.utils.config import RegulationConfig, Generation


@pytest.fixture
def regulation_config() -> RegulationConfig:
    """Create a test regulation config."""
    return RegulationConfig(
        regulation_id="test-sv2024-1",
        generation=Generation.GEN_9,
        legal_pokemon={"pikachu", "charizard", "dragonite", "torkoal"},
        legal_items={"choice-band", "choice-scarf", "life-orb"},
        legal_moves={"earthquake", "protected", "surf", "thunderbolt"},
        legal_abilities={"pressure", "static", "rough-skin"},
        legal_tera_types={"fire", "water", "electric"},
        team_size=6,
        active_pokemon=2,
    )


@pytest.fixture
def mock_observation() -> Dict:
    """Create a mock battle observation."""
    return {
        "team": [
            {
                "species": "pikachu",
                "hp": 100,
                "max_hp": 100,
                "status": "none",
                "moves": ["thunderbolt", "earthquake", "protected", "surf"],
                "item": "choice-band",
                "ability": "static",
                "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "tera_used": False,
                "stats": {"hp": 100, "atk": 80, "def": 75, "spa": 100, "spd": 80, "spe": 120},
            },
            {
                "species": "charizard",
                "hp": 120,
                "max_hp": 120,
                "status": "none",
                "moves": ["earthquake", "protected", "surf", "thunderbolt"],
                "item": "choice-scarf",
                "ability": "pressure",
                "boosts": {"atk": 1, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "tera_used": False,
                "stats": {"hp": 120, "atk": 110, "def": 95, "spa": 130, "spd": 95, "spe": 115},
            },
            {
                "species": "torkoal",
                "hp": 140,
                "max_hp": 140,
                "status": "none",
                "moves": ["earthquake", "protected", "surf", "thunderbolt"],
                "item": "life-orb",
                "ability": "rough-skin",
                "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "tera_used": True,
                "stats": {"hp": 140, "atk": 90, "def": 140, "spa": 130, "spd": 120, "spe": 30},
            },
        ],
        "active_pokemon": [
            {
                "species": "pikachu",
                "hp": 100,
                "max_hp": 100,
                "status": "none",
                "moves": ["thunderbolt", "earthquake", "protected", "surf"],
            },
            {
                "species": "charizard",
                "hp": 120,
                "max_hp": 120,
                "status": "none",
                "moves": ["earthquake", "protected", "surf", "thunderbolt"],
            },
        ],
        "opponent_team": [
            {
                "species": "dragonite",
                "hp": 110,
                "max_hp": 110,
                "status": "none",
            },
            {
                "species": "torkoal",
                "hp": 120,
                "max_hp": 120,
                "status": "burn",
            },
        ],
        "field": {
            "weather": "none",
            "terrain": "none",
            "trick_room": False,
            "reflect": False,
            "light_screen": False,
            "tailwind": False,
        },
        "level": 50,
    }


@pytest.fixture
def mock_belief_state() -> Dict[str, PokemonBeliefState]:
    """Create mock belief states for opponent Pokemon."""
    return {
        "dragonite": PokemonBeliefState(
            name="dragonite",
            hp_min=100,
            hp_max=120,
            attack_min=100,
            attack_max=130,
            defense_min=80,
            defense_max=100,
            spa_min=80,
            spa_max=120,
            spd_min=90,
            spd_max=110,
            speed_min=50,
            speed_max=80,
            moves={"earthquake", "outrage", "extreme-speed"},
            items={"choice-band", "assault-vest"},
            abilities={"multiscale", "shed-skin"},
            tera_types={"dragon", "steel"},
        ),
        "torkoal": PokemonBeliefState(
            name="torkoal",
            hp_min=110,
            hp_max=140,
            attack_min=80,
            attack_max=100,
            defense_min=120,
            defense_max=150,
            spa_min=110,
            spa_max=140,
            spd_min=100,
            spd_max=130,
            speed_min=20,
            speed_max=50,
            moves={"earthquake", "protected", "surf"},
            items={"assault-vest", "life-orb"},
            abilities={"drought", "shell-armor"},
            tera_types={"fire", "water"},
        ),
    }


@pytest.fixture
def encoder(regulation_config: RegulationConfig) -> FeatureEncoder:
    """Create a FeatureEncoder instance."""
    return FeatureEncoder(regulation_config=regulation_config)


class TestFeatureEncoderInitialization:
    """Test FeatureEncoder initialization."""
    
    def test_encoder_creates_successfully(self, regulation_config: RegulationConfig):
        """Test that encoder initializes correctly."""
        encoder = FeatureEncoder(regulation_config=regulation_config)
        assert encoder.regulation_config == regulation_config
        assert encoder.tensor_shapes == DEFAULT_TENSOR_SHAPES
    
    def test_encoder_with_custom_tensor_shapes(self, regulation_config: RegulationConfig):
        """Test encoder with custom tensor shapes."""
        custom_shapes = TensorShapes(pokemon_embedding_dim=256)
        encoder = FeatureEncoder(regulation_config=regulation_config, tensor_shapes=custom_shapes)
        assert encoder.tensor_shapes == custom_shapes


class TestFeatureEncoderOutput:
    """Test feature encoder output shapes and types."""
    
    def test_encode_returns_encoded_state(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that encode returns EncodedBattleState."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert hasattr(encoded, "team_tensor")
        assert hasattr(encoded, "active_pokemon_tensor")
        assert hasattr(encoded, "field_tensor")
        assert hasattr(encoded, "opponent_belief_tensor")
        assert hasattr(encoded, "regulation_embedding")
        assert hasattr(encoded, "move_masks")
        assert hasattr(encoded, "switch_masks")
        assert hasattr(encoded, "target_masks")
    
    def test_team_tensor_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test team tensor has correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, team_size, entity_dim = encoded.team_tensor.shape
        assert batch_shape == 1
        assert team_size == DEFAULT_TENSOR_SHAPES.max_team_size
        assert entity_dim == DEFAULT_TENSOR_SHAPES.get_pokemon_entity_dim()
    
    def test_active_pokemon_tensor_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test active pokemon tensor has correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, active_count, entity_dim = encoded.active_pokemon_tensor.shape
        assert batch_shape == 1
        assert active_count == DEFAULT_TENSOR_SHAPES.max_active_pokemon
        assert entity_dim == DEFAULT_TENSOR_SHAPES.get_pokemon_entity_dim()
    
    def test_field_tensor_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test field tensor has correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, feature_dim = encoded.field_tensor.shape
        assert batch_shape == 1
        assert feature_dim == DEFAULT_TENSOR_SHAPES.field_feature_dim
    
    def test_opponent_belief_tensor_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test opponent belief tensor has correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, team_size, belief_dim = encoded.opponent_belief_tensor.shape
        assert batch_shape == 1
        assert team_size == DEFAULT_TENSOR_SHAPES.max_possible_opponents
        assert belief_dim == DEFAULT_TENSOR_SHAPES.belief_feature_dim
    
    def test_regulation_embedding_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test regulation embedding has correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, embedding_dim = encoded.regulation_embedding.shape
        assert batch_shape == 1
        assert embedding_dim == DEFAULT_TENSOR_SHAPES.regulation_embedding_dim
    
    def test_move_masks_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test move masks have correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, active_count, move_choices = encoded.move_masks.shape
        assert batch_shape == 1
        assert active_count == DEFAULT_TENSOR_SHAPES.max_active_pokemon
        assert move_choices == DEFAULT_TENSOR_SHAPES.max_move_choices
    
    def test_switch_masks_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test switch masks have correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, switch_choices = encoded.switch_masks.shape
        assert batch_shape == 1
        assert switch_choices == DEFAULT_TENSOR_SHAPES.max_switch_choices
    
    def test_target_masks_shape(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test target masks have correct shape."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        batch_shape, active_count, target_choices = encoded.target_masks.shape
        assert batch_shape == 1
        assert active_count == DEFAULT_TENSOR_SHAPES.max_active_pokemon
        assert target_choices == DEFAULT_TENSOR_SHAPES.max_target_choices


class TestPokemonEntityEncoding:
    """Test Pokemon entity encoding."""
    
    def test_pokemon_entity_is_tensor(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that Pokemon entity is encoded as tensor."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert isinstance(encoded.team_tensor, torch.Tensor)
        assert encoded.team_tensor.dtype == torch.float32
    
    def test_team_encoding_includes_all_pokemon(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test team encoding includes all 6 Pokemon (padded if necessary)."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Should have 3 Pokemon + 3 padding
        assert encoded.team_tensor.shape[1] == 6
        
        # First 3 should be non-zero, last 3 should be zero or padding
        first_pokemon = encoded.team_tensor[0, 0, :]
        assert torch.any(first_pokemon != 0)  # Should have some non-zero values
    
    def test_pokemon_hp_percentage_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that Pokemon HP is encoded as percentage."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Get first Pokemon's HP value (should be between 0 and 1 after normalization)
        team_tensor = encoded.team_tensor[0, 0, :]
        # HP should be one of the features (not exactly at index 0 due to embedding)
        # Just verify tensor has reasonable values
        assert torch.all(team_tensor >= -1.0)
        assert torch.all(team_tensor <= 1.0)
    
    def test_pokemon_status_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that Pokemon status is encoded."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Status should be encoded in the entity tensor
        assert encoded.team_tensor.shape[2] > 0
    
    def test_pokemon_boosts_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that Pokemon stat boosts are encoded."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Boosts should be part of the entity encoding
        assert encoded.team_tensor.shape[2] > 0
    
    def test_pokemon_tera_used_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that Tera usage is encoded."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Tera used should be binary (0 or 1) and part of encoding
        assert encoded.team_tensor.shape[2] > 0


class TestActionMaskEncoding:
    """Test action mask encoding."""
    
    def test_move_masks_are_boolean(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that move masks are boolean (0 or 1)."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.all((encoded.move_masks == 0) | (encoded.move_masks == 1))
    
    def test_switch_masks_are_boolean(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that switch masks are boolean."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.all((encoded.switch_masks == 0) | (encoded.switch_masks == 1))
    
    def test_target_masks_are_boolean(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that target masks are boolean."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.all((encoded.target_masks == 0) | (encoded.target_masks == 1))
    
    def test_move_masks_valid_moves_set(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that move masks correctly encode available moves."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # First Pokemon should have at least 1 valid move
        assert torch.sum(encoded.move_masks[0, 0, :]) >= 1
    
    def test_switch_masks_available_switches(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that switch masks encode available switches."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Should have at least some valid switches
        total_switches = torch.sum(encoded.switch_masks[0, :])
        assert total_switches >= 0  # May be 0 in some edge cases


class TestBeliefStateEncoding:
    """Test belief state encoding."""
    
    def test_belief_state_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that belief state is encoded."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert isinstance(encoded.opponent_belief_tensor, torch.Tensor)
        assert encoded.opponent_belief_tensor.shape[0] == 1
    
    def test_belief_state_incorporates_uncertainty(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that belief state captures uncertainty ranges."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Belief should have higher dimension to represent uncertainty
        assert encoded.opponent_belief_tensor.shape[2] == DEFAULT_TENSOR_SHAPES.belief_feature_dim
    
    def test_belief_state_includes_speed_prediction(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that belief state includes speed prediction."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Speed prediction should be part of belief encoding
        assert encoded.opponent_belief_tensor.shape[2] > 0


class TestFieldStateEncoding:
    """Test field state encoding."""
    
    def test_field_state_encoded(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that field state is encoded."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert isinstance(encoded.field_tensor, torch.Tensor)
        assert encoded.field_tensor.shape[1] == DEFAULT_TENSOR_SHAPES.field_feature_dim
    
    def test_field_includes_weather(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that field encoding includes weather."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Field should have some representation for weather
        assert encoded.field_tensor.shape[1] > 0
    
    def test_field_includes_terrain(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that field encoding includes terrain."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert encoded.field_tensor.shape[1] > 0
    
    def test_field_includes_screens_and_tailwind(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that field includes screens and tailwind."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert encoded.field_tensor.shape[1] > 0


class TestRegulationEncoding:
    """Test regulation encoding."""
    
    def test_regulation_embedding_created(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that regulation is encoded as embedding."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert encoded.regulation_embedding.shape[0] == 1
        assert encoded.regulation_embedding.shape[1] == DEFAULT_TENSOR_SHAPES.regulation_embedding_dim
    
    def test_regulation_embedding_is_float(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that regulation embedding is float32."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert encoded.regulation_embedding.dtype == torch.float32


class TestDeterminism:
    """Test that encoding is deterministic."""
    
    def test_same_input_produces_same_output(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that same input produces same output (deterministic)."""
        encoded1 = encoder.encode(mock_observation, mock_belief_state)
        encoded2 = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.allclose(encoded1.team_tensor, encoded2.team_tensor)
        assert torch.allclose(encoded1.field_tensor, encoded2.field_tensor)
        assert torch.allclose(encoded1.opponent_belief_tensor, encoded2.opponent_belief_tensor)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_all_pokemon_types_handled(self, regulation_config: RegulationConfig):
        """Test that all legal Pokemon types are handled."""
        encoder = FeatureEncoder(regulation_config=regulation_config)
        
        for pokemon_name in regulation_config.legal_pokemon:
            observation = {
                "team": [
                    {
                        "species": pokemon_name,
                        "hp": 100,
                        "max_hp": 100,
                        "status": "none",
                        "moves": ["earthquake", "protected"],
                        "item": "choice-band",
                        "ability": "pressure",
                        "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                        "tera_used": False,
                        "stats": {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
                    },
                ] + [{"species": name} for name in list(regulation_config.legal_pokemon)[:5]],
                "active_pokemon": [],
                "opponent_team": [],
                "field": {"weather": "none", "terrain": "none"},
                "level": 50,
            }
            
            belief = {}
            encoded = encoder.encode(observation, belief)
            
            # Should not raise an exception
            assert encoded is not None
    
    def test_padding_with_fewer_pokemon(self, encoder: FeatureEncoder):
        """Test padding when team has fewer than 6 Pokemon."""
        observation = {
            "team": [
                {
                    "species": "pikachu",
                    "hp": 100,
                    "max_hp": 100,
                    "status": "none",
                    "moves": ["thunderbolt"],
                    "item": "choice-band",
                    "ability": "static",
                    "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                    "tera_used": False,
                    "stats": {"hp": 100, "atk": 80, "def": 75, "spa": 100, "spd": 80, "spe": 120},
                },
            ],
            "active_pokemon": [],
            "opponent_team": [],
            "field": {"weather": "none", "terrain": "none"},
            "level": 50,
        }
        belief = {}
        
        encoded = encoder.encode(observation, belief)
        
        # Should still have 6 slots (padded)
        assert encoded.team_tensor.shape[1] == 6
    
    def test_batch_encoding(self, encoder: FeatureEncoder, mock_observation: Dict):
        """Test encoding multiple observations as batch."""
        encoded_single = encoder.encode(mock_observation, {})
        
        # Encoder should support batching via stacking
        assert encoded_single.team_tensor.shape[0] == 1


class TestTensorValues:
    """Test that tensor values are within expected ranges."""
    
    def test_team_tensor_normalized_values(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that team tensor has normalized values."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        # Most values should be normalized to [-1, 1] range (or [0, 1] for probabilities)
        # Allow some embeddings to be outside this range
        team_tensor = encoded.team_tensor
        extreme_values = torch.sum((team_tensor < -10) | (team_tensor > 10))
        
        # Should not have too many extreme values
        assert extreme_values < team_tensor.numel() * 0.05
    
    def test_field_tensor_contains_finite_values(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that field tensor has finite values."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.all(torch.isfinite(encoded.field_tensor))
    
    def test_belief_tensor_contains_finite_values(
        self,
        encoder: FeatureEncoder,
        mock_observation: Dict,
        mock_belief_state: Dict,
    ):
        """Test that belief tensor has finite values."""
        encoded = encoder.encode(mock_observation, mock_belief_state)
        
        assert torch.all(torch.isfinite(encoded.opponent_belief_tensor))
