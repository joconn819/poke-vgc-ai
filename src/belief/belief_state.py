"""Belief state modeling for hidden information in Pokemon battles."""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional


@dataclass
class PokemonBeliefState:
    """Belief state for a single Pokemon, tracking uncertainty about its attributes.
    
    Maintains min/max bounds for stats and sets of possible values for
    discrete attributes (moves, items, abilities, tera types).
    """

    name: str
    
    # HP bounds
    hp_min: int = 0
    hp_max: int = 255
    
    # Stat bounds
    attack_min: int = 0
    attack_max: int = 255
    defense_min: int = 0
    defense_max: int = 255
    spa_min: int = 0
    spa_max: int = 255
    spd_min: int = 0
    spd_max: int = 255
    speed_min: int = 0
    speed_max: int = 255
    
    # Discrete attributes (sets of possibilities)
    moves: Set[str] = field(default_factory=set)
    items: Set[str] = field(default_factory=set)
    abilities: Set[str] = field(default_factory=set)
    tera_types: Set[str] = field(default_factory=set)

    def expected_hp(self) -> float:
        """Return expected HP value (midpoint of range)."""
        return (self.hp_min + self.hp_max) / 2.0

    def expected_attack(self) -> float:
        """Return expected attack value (midpoint of range)."""
        return (self.attack_min + self.attack_max) / 2.0

    def expected_defense(self) -> float:
        """Return expected defense value (midpoint of range)."""
        return (self.defense_min + self.defense_max) / 2.0

    def expected_spa(self) -> float:
        """Return expected SpA value (midpoint of range)."""
        return (self.spa_min + self.spa_max) / 2.0

    def expected_spd(self) -> float:
        """Return expected SpD value (midpoint of range)."""
        return (self.spd_min + self.spd_max) / 2.0

    def expected_speed(self) -> float:
        """Return expected speed value (midpoint of range)."""
        return (self.speed_min + self.speed_max) / 2.0

    def variance_hp(self) -> float:
        """Calculate variance of HP (assuming uniform distribution)."""
        if self.hp_min == self.hp_max:
            return 0.0
        # Variance of uniform distribution: (b - a)^2 / 12
        return (self.hp_max - self.hp_min) ** 2 / 12.0

    def variance_attack(self) -> float:
        """Calculate variance of attack stat."""
        if self.attack_min == self.attack_max:
            return 0.0
        return (self.attack_max - self.attack_min) ** 2 / 12.0

    def variance_defense(self) -> float:
        """Calculate variance of defense stat."""
        if self.defense_min == self.defense_max:
            return 0.0
        return (self.defense_max - self.defense_min) ** 2 / 12.0

    def variance_spa(self) -> float:
        """Calculate variance of SpA stat."""
        if self.spa_min == self.spa_max:
            return 0.0
        return (self.spa_max - self.spa_min) ** 2 / 12.0

    def variance_spd(self) -> float:
        """Calculate variance of SpD stat."""
        if self.spd_min == self.spd_max:
            return 0.0
        return (self.spd_max - self.spd_min) ** 2 / 12.0

    def variance_speed(self) -> float:
        """Calculate variance of speed stat."""
        if self.speed_min == self.speed_max:
            return 0.0
        return (self.speed_max - self.speed_min) ** 2 / 12.0

    def get_uncertainty(self) -> float:
        """Calculate total uncertainty across all stats.
        
        Returns sum of variances for all continuous stats.
        """
        return (
            self.variance_hp()
            + self.variance_attack()
            + self.variance_defense()
            + self.variance_spa()
            + self.variance_spd()
            + self.variance_speed()
        )

    def update_moves(self, new_moves: Set[str]) -> None:
        """Update the set of possible moves.
        
        If moves are already known, use intersection (logical AND).
        Otherwise, replace with the new set.
        """
        if self.moves:
            self.moves = self.moves & new_moves
        else:
            self.moves = new_moves.copy()

    def update_items(self, new_items: Set[str]) -> None:
        """Update the set of possible items.
        
        If items are already known, use intersection (logical AND).
        Otherwise, replace with the new set.
        """
        if self.items:
            self.items = self.items & new_items
        else:
            self.items = new_items.copy()

    def update_abilities(self, new_abilities: Set[str]) -> None:
        """Update the set of possible abilities.
        
        If abilities are already known, use intersection (logical AND).
        Otherwise, replace with the new set.
        """
        if self.abilities:
            self.abilities = self.abilities & new_abilities
        else:
            self.abilities = new_abilities.copy()

    def update_tera_types(self, new_tera_types: Set[str]) -> None:
        """Update the set of possible tera types.
        
        If tera types are already known, use intersection (logical AND).
        Otherwise, replace with the new set.
        """
        if self.tera_types:
            self.tera_types = self.tera_types & new_tera_types
        else:
            self.tera_types = new_tera_types.copy()

    def update_hp_range(self, new_min: int, new_max: int) -> None:
        """Update HP range to intersection with new bounds."""
        self.hp_min = max(self.hp_min, new_min)
        self.hp_max = min(self.hp_max, new_max)

    def update_attack_range(self, new_min: int, new_max: int) -> None:
        """Update attack range to intersection with new bounds."""
        self.attack_min = max(self.attack_min, new_min)
        self.attack_max = min(self.attack_max, new_max)

    def update_defense_range(self, new_min: int, new_max: int) -> None:
        """Update defense range to intersection with new bounds."""
        self.defense_min = max(self.defense_min, new_min)
        self.defense_max = min(self.defense_max, new_max)

    def update_spa_range(self, new_min: int, new_max: int) -> None:
        """Update SpA range to intersection with new bounds."""
        self.spa_min = max(self.spa_min, new_min)
        self.spa_max = min(self.spa_max, new_max)

    def update_spd_range(self, new_min: int, new_max: int) -> None:
        """Update SpD range to intersection with new bounds."""
        self.spd_min = max(self.spd_min, new_min)
        self.spd_max = min(self.spd_max, new_max)

    def update_speed_range(self, new_min: int, new_max: int) -> None:
        """Update speed range to intersection with new bounds."""
        self.speed_min = max(self.speed_min, new_min)
        self.speed_max = min(self.speed_max, new_max)


class OpponentBeliefState:
    """Belief state for opponent's entire team.
    
    Maintains a dictionary of individual Pokemon belief states and provides
    methods to query and update the collective belief.
    """

    def __init__(self, team: Optional[Dict[str, PokemonBeliefState]] = None):
        """Initialize opponent belief state.
        
        Args:
            team: Dictionary mapping Pokemon name to PokemonBeliefState.
                  Defaults to empty dict if not provided.
        """
        self.team: Dict[str, PokemonBeliefState] = team or {}

    def add_pokemon(self, name: str, belief: PokemonBeliefState) -> None:
        """Add a Pokemon to the team with its belief state."""
        self.team[name] = belief

    def get_pokemon(self, name: str) -> Optional[PokemonBeliefState]:
        """Get belief state for a specific Pokemon."""
        return self.team.get(name)

    def remove_pokemon(self, name: str) -> None:
        """Remove a Pokemon from the team."""
        self.team.pop(name, None)

    def get_all_pokemon_names(self) -> list[str]:
        """Get list of all Pokemon names in the team."""
        return list(self.team.keys())

    def team_size(self) -> int:
        """Get the number of Pokemon in the team."""
        return len(self.team)

    def update_pokemon_belief(
        self,
        name: str,
        hp_min: Optional[int] = None,
        hp_max: Optional[int] = None,
        attack_min: Optional[int] = None,
        attack_max: Optional[int] = None,
        defense_min: Optional[int] = None,
        defense_max: Optional[int] = None,
        spa_min: Optional[int] = None,
        spa_max: Optional[int] = None,
        spd_min: Optional[int] = None,
        spd_max: Optional[int] = None,
        speed_min: Optional[int] = None,
        speed_max: Optional[int] = None,
        moves: Optional[Set[str]] = None,
        items: Optional[Set[str]] = None,
        abilities: Optional[Set[str]] = None,
        tera_types: Optional[Set[str]] = None,
    ) -> None:
        """Update belief state for a specific Pokemon with new constraints.
        
        Only provided parameters are updated; None values are ignored.
        """
        pokemon = self.get_pokemon(name)
        if pokemon is None:
            return

        if hp_min is not None or hp_max is not None:
            pokemon.update_hp_range(
                hp_min if hp_min is not None else pokemon.hp_min,
                hp_max if hp_max is not None else pokemon.hp_max,
            )

        if attack_min is not None or attack_max is not None:
            pokemon.update_attack_range(
                attack_min if attack_min is not None else pokemon.attack_min,
                attack_max if attack_max is not None else pokemon.attack_max,
            )

        if defense_min is not None or defense_max is not None:
            pokemon.update_defense_range(
                defense_min if defense_min is not None else pokemon.defense_min,
                defense_max if defense_max is not None else pokemon.defense_max,
            )

        if spa_min is not None or spa_max is not None:
            pokemon.update_spa_range(
                spa_min if spa_min is not None else pokemon.spa_min,
                spa_max if spa_max is not None else pokemon.spa_max,
            )

        if spd_min is not None or spd_max is not None:
            pokemon.update_spd_range(
                spd_min if spd_min is not None else pokemon.spd_min,
                spd_max if spd_max is not None else pokemon.spd_max,
            )

        if speed_min is not None or speed_max is not None:
            pokemon.update_speed_range(
                speed_min if speed_min is not None else pokemon.speed_min,
                speed_max if speed_max is not None else pokemon.speed_max,
            )

        if moves is not None:
            pokemon.update_moves(moves)

        if items is not None:
            pokemon.update_items(items)

        if abilities is not None:
            pokemon.update_abilities(abilities)

        if tera_types is not None:
            pokemon.update_tera_types(tera_types)

    def update_pokemon_moves(self, name: str, moves: Set[str]) -> None:
        """Update the possible moves for a specific Pokemon."""
        pokemon = self.get_pokemon(name)
        if pokemon is not None:
            pokemon.update_moves(moves)

    def get_pokemon_expected_hp(self, name: str) -> Optional[float]:
        """Get expected HP for a specific Pokemon."""
        pokemon = self.get_pokemon(name)
        return pokemon.expected_hp() if pokemon else None

    def get_pokemon_uncertainty(self, name: str) -> Optional[float]:
        """Get total uncertainty for a specific Pokemon."""
        pokemon = self.get_pokemon(name)
        return pokemon.get_uncertainty() if pokemon else None

    def average_team_uncertainty(self) -> float:
        """Calculate average uncertainty across the entire team."""
        if not self.team:
            return 0.0
        total_uncertainty = sum(
            pokemon.get_uncertainty() for pokemon in self.team.values()
        )
        return total_uncertainty / len(self.team)
