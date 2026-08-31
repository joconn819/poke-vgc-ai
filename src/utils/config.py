"""Configuration and constants for Pokemon VGC formats."""

from dataclasses import dataclass, field
from typing import Set, Tuple
from enum import Enum


class Generation(Enum):
    """Pokemon generation enum."""
    GEN_9 = 9
    GEN_10 = 10


@dataclass
class RegulationConfig:
    """Configuration for a VGC regulation.
    
    Defines legal Pokemon, items, abilities, moves, and tera types
    for a specific regulation.
    """
    regulation_id: str
    generation: Generation
    
    # Legality constraints
    legal_pokemon: Set[str] = field(default_factory=set)
    legal_items: Set[str] = field(default_factory=set)
    legal_moves: Set[str] = field(default_factory=set)
    legal_abilities: Set[str] = field(default_factory=set)
    legal_tera_types: Set[str] = field(default_factory=set)
    
    # Restrictions
    restricted_pokemon: Set[str] = field(default_factory=set)
    max_restricted_per_team: int = 1
    min_level: int = 1
    max_level: int = 100
    
    # Format specifics
    team_size: int = 6
    active_pokemon: int = 2
    is_vgc_doubles: bool = True
    
    def __hash__(self):
        """Make hashable for caching."""
        return hash(self.regulation_id)
    
    def is_pokemon_legal(self, pokemon_name: str) -> bool:
        """Check if a Pokemon is legal under this regulation."""
        return pokemon_name in self.legal_pokemon
    
    def is_item_legal(self, item_name: str) -> bool:
        """Check if an item is legal under this regulation."""
        return item_name in self.legal_items
    
    def is_move_legal(self, move_name: str) -> bool:
        """Check if a move is legal under this regulation."""
        return move_name in self.legal_moves


# Example regulation for testing (minimal)
REGULATION_SV2024_1 = RegulationConfig(
    regulation_id="sv2024-1",
    generation=Generation.GEN_9,
    legal_pokemon={
        "pikachu", "charizard", "dragonite", "salamence", "garchomp",
        "landorus", "tornadus", "thundurus", "torkoal", "urshifu",
        "calyrex", "regieleki", "regidrago", "glastrier", "spectrier",
        "kyurem", "zekrom", "reshiram", "xerneas", "yveltal",
        "groudon", "kyogre", "rayquaza", "blaziken", "venusaur",
    },
    legal_items={
        "choice-band", "choice-scarf", "choice-specs", "assault-vest",
        "life-orb", "rocky-helmet", "air-balloon", "heavy-duty-boots",
    },
    legal_moves={
        "earthquake", "protected", "surf", "thunderbolt", "ice-beam",
        "shadow-ball", "focus-blast", "power-whip", "stone-edge",
    },
    legal_abilities={
        "pressure", "static", "rough-skin", "levitate", "speed-boost",
        "intimidate", "drought", "drizzle",
    },
    legal_tera_types={
        "normal", "fire", "water", "electric", "grass", "ice", "fighting",
        "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
        "dragon", "dark", "steel", "fairy",
    },
    restricted_pokemon={"groudon", "kyogre", "rayquaza"},
)
