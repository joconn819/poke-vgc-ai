"""Tensor shape definitions and constants for feature encoding."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TensorShapes:
    """Canonical tensor shapes for feature encoding.
    
    This dataclass defines the expected output shapes for the feature encoder.
    All shapes follow (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
    conventions for compatibility with Transformer architectures.
    """
    
    # Pokemon entity embedding dimensions
    pokemon_embedding_dim: int = 128
    move_embedding_dim: int = 64
    item_embedding_dim: int = 32
    ability_embedding_dim: int = 32
    
    # Sequence dimensions
    max_team_size: int = 6  # Max Pokemon per team
    max_active_pokemon: int = 2  # Active Pokemon in VGC doubles
    max_moves_per_pokemon: int = 4  # Moves per Pokemon
    max_possible_opponents: int = 6  # Opponent team size (even if hidden)
    
    # Field state
    field_feature_dim: int = 64
    
    # Belief state
    belief_feature_dim: int = 128
    
    # Regulation token
    regulation_embedding_dim: int = 32
    
    # Action space dimensions
    max_move_choices: int = 8  # Max moves available in a turn
    max_switch_choices: int = 4  # Max switches available in a turn
    max_target_choices: int = 2  # Targets in doubles (opponent active 1 or 2)
    
    # Output dimensions for attention
    # Team: variable length up to max_team_size, each pokemon is pokemon_embedding_dim
    team_sequence_out_dim: int = pokemon_embedding_dim
    
    # Opponent team (belief): variable length, with uncertainty
    opponent_sequence_out_dim: int = pokemon_embedding_dim
    
    # Full state vector
    state_vector_dim: int = 512
    
    def get_pokemon_entity_dim(self) -> int:
        """Get total dimension for a single Pokemon entity."""
        # species_embedding + hp + status_embedding + item_embedding + ability_embedding 
        # + move_embeddings + stat_values (6) + boosts (6) + tera_used (1)
        return (
            self.pokemon_embedding_dim  # species embedding (128)
            + 1  # hp percentage
            + 8  # status embedding (not 1, since status_embedding has dim=8)
            + self.item_embedding_dim  # item (32)
            + self.ability_embedding_dim  # ability (32)
            + self.max_moves_per_pokemon * self.move_embedding_dim  # 4 * 64 = 256
            + 6  # stat values (normalized)
            + 6  # stat boosts
            + 1  # tera used flag
        )
    
    def get_team_tensor_shape(self) -> tuple:
        """Get shape of team tensor: (batch, max_team_size, entity_dim)."""
        return (None, self.max_team_size, self.get_pokemon_entity_dim())
    
    def get_active_pokemon_tensor_shape(self) -> tuple:
        """Get shape of active Pokemon tensor: (batch, active_count, entity_dim)."""
        return (None, self.max_active_pokemon, self.get_pokemon_entity_dim())
    
    def get_field_tensor_shape(self) -> tuple:
        """Get shape of field state tensor: (batch, field_feature_dim)."""
        return (None, self.field_feature_dim)
    
    def get_belief_tensor_shape(self) -> tuple:
        """Get shape of belief state tensor: (batch, opponent_team_size, belief_dim)."""
        return (None, self.max_possible_opponents, self.belief_feature_dim)
    
    def get_action_mask_shape(self) -> tuple:
        """Get shape of action mask tensor."""
        # Masks for moves, switches, targets
        total_actions = (
            self.max_active_pokemon * self.max_move_choices +  # moves for each active pokemon
            self.max_switch_choices +  # switch options
            self.max_active_pokemon * self.max_target_choices  # targets for each active pokemon
        )
        return (None, total_actions)


# Global tensor shape configuration
DEFAULT_TENSOR_SHAPES = TensorShapes()


# Pokemon species index mapping (canonical)
# This should map Pokemon names to indices
POKEMON_INDEX_MAP: Dict[str, int] = {
    # Legendary/Restricted
    "pikachu": 0,
    "charizard": 1,
    "dragonite": 2,
    "salamence": 3,
    "garchomp": 4,
    "landorus": 5,
    "tornadus": 6,
    "thundurus": 7,
    "torkoal": 8,
    "urshifu": 9,
    "calyrex": 10,
    "regieleki": 11,
    "regidrago": 12,
    "glastrier": 13,
    "spectrier": 14,
    "kyurem": 15,
    "zekrom": 16,
    "reshiram": 17,
    "xerneas": 18,
    "yveltal": 19,
    "groudon": 20,
    "kyogre": 21,
    "rayquaza": 22,
    "blaziken": 23,
    "venusaur": 24,
}

REVERSE_POKEMON_INDEX: Dict[int, str] = {v: k for k, v in POKEMON_INDEX_MAP.items()}


# Move index mapping
MOVE_INDEX_MAP: Dict[str, int] = {
    "earthquake": 0,
    "protected": 1,
    "surf": 2,
    "thunderbolt": 3,
    "ice-beam": 4,
    "shadow-ball": 5,
    "focus-blast": 6,
    "power-whip": 7,
    "stone-edge": 8,
    "aqua-jet": 9,
    "sucker-punch": 10,
    "trick-room": 11,
    "follow-me": 12,
    "wide-guard": 13,
}

REVERSE_MOVE_INDEX: Dict[int, str] = {v: k for k, v in MOVE_INDEX_MAP.items()}


# Item index mapping
ITEM_INDEX_MAP: Dict[str, int] = {
    "choice-band": 0,
    "choice-scarf": 1,
    "choice-specs": 2,
    "assault-vest": 3,
    "life-orb": 4,
    "rocky-helmet": 5,
    "air-balloon": 6,
    "heavy-duty-boots": 7,
}

REVERSE_ITEM_INDEX: Dict[int, str] = {v: k for k, v in ITEM_INDEX_MAP.items()}


# Ability index mapping
ABILITY_INDEX_MAP: Dict[str, int] = {
    "pressure": 0,
    "static": 1,
    "rough-skin": 2,
    "levitate": 3,
    "speed-boost": 4,
    "intimidate": 5,
    "drought": 6,
    "drizzle": 7,
}

REVERSE_ABILITY_INDEX: Dict[int, str] = {v: k for k, v in ABILITY_INDEX_MAP.items()}


# Status condition index mapping
STATUS_INDEX_MAP: Dict[str, int] = {
    "none": 0,
    "burn": 1,
    "freeze": 2,
    "paralysis": 3,
    "poison": 4,
    "badly-poisoned": 5,
    "sleep": 6,
}

REVERSE_STATUS_INDEX: Dict[int, str] = {v: k for k, v in STATUS_INDEX_MAP.items()}


# Weather index mapping
WEATHER_INDEX_MAP: Dict[str, int] = {
    "none": 0,
    "hail": 1,
    "raindance": 2,
    "sandstorm": 3,
    "sunnyday": 4,
}

REVERSE_WEATHER_INDEX: Dict[int, str] = {v: k for k, v in WEATHER_INDEX_MAP.items()}


# Terrain index mapping
TERRAIN_INDEX_MAP: Dict[str, int] = {
    "none": 0,
    "electric": 1,
    "grassy": 2,
    "misty": 3,
    "psychic": 4,
}

REVERSE_TERRAIN_INDEX: Dict[int, str] = {v: k for k, v in TERRAIN_INDEX_MAP.items()}


# Tera type index mapping
TERA_TYPE_INDEX_MAP: Dict[str, int] = {
    "normal": 0,
    "fire": 1,
    "water": 2,
    "electric": 3,
    "grass": 4,
    "ice": 5,
    "fighting": 6,
    "poison": 7,
    "ground": 8,
    "flying": 9,
    "psychic": 10,
    "bug": 11,
    "rock": 12,
    "ghost": 13,
    "dragon": 14,
    "dark": 15,
    "steel": 16,
    "fairy": 17,
}

REVERSE_TERA_TYPE_INDEX: Dict[int, str] = {v: k for k, v in TERA_TYPE_INDEX_MAP.items()}
