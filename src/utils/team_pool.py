"""Team pool for sampling Pokemon teams for battles."""

import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.utils.config import RegulationConfig


class TeamPool:
    """Loads and samples Pokemon teams from JSON files.
    
    Supports lazy-loading of teams and filters by regulation legality.
    """
    
    def __init__(self, teams_file: str, regulation: RegulationConfig):
        """Initialize TeamPool by loading teams from a JSON file.
        
        Args:
            teams_file: Path to JSON file containing team data
            regulation: RegulationConfig defining legal Pokemon/items/moves
            
        Raises:
            FileNotFoundError: If teams_file does not exist
        """
        self.teams_file = Path(teams_file)
        self.regulation = regulation
        self._teams: Optional[List[Dict[str, Any]]] = None
        
        # Validate file exists
        if not self.teams_file.exists():
            raise FileNotFoundError(f"Teams file not found: {self.teams_file}")
    
    def _load_teams(self) -> None:
        """Lazy-load teams from JSON file."""
        if self._teams is not None:
            return
        
        with open(self.teams_file, 'r') as f:
            all_teams = json.load(f)
        
        # Filter teams by regulation legality
        self._teams = self._filter_legal_teams(all_teams)
    
    def _filter_legal_teams(self, teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter teams to keep only regulation-legal ones.
        
        A team is legal if:
        - All Pokemon are in legal_pokemon set
        - All items are in legal_items set
        - All moves are in legal_moves set
        - All abilities are in legal_abilities set
        """
        legal_teams = []
        
        for team in teams:
            if self._is_team_legal(team):
                legal_teams.append(team)
        
        return legal_teams
    
    def _is_team_legal(self, team: Dict[str, Any]) -> bool:
        """Check if a team is legal under the regulation.
        
        Args:
            team: Team dict with "pokemon" key containing list of Pokemon
            
        Returns:
            True if all Pokemon, items, moves, and abilities are legal
        """
        if "pokemon" not in team:
            return False
        
        pokemon_list = team["pokemon"]
        if len(pokemon_list) != 6:
            return False
        
        for poke in pokemon_list:
            if not self._is_pokemon_legal(poke):
                return False
        
        return True
    
    def _is_pokemon_legal(self, pokemon: Dict[str, Any]) -> bool:
        """Check if a single Pokemon is legal.
        
        Args:
            pokemon: Pokemon dict with name, item, ability, moves
            
        Returns:
            True if Pokemon, item, ability, and all moves are legal
        """
        # Check name (case-insensitive)
        name = pokemon.get("name", "").lower()
        if name not in self.regulation.legal_pokemon:
            return False
        
        # Check item (case-insensitive, handle various formats)
        item = pokemon.get("item", "").lower().replace(" ", "-")
        if item not in self.regulation.legal_items:
            return False
        
        # Check ability (case-insensitive)
        ability = pokemon.get("ability", "").lower().replace(" ", "-")
        if not self._is_ability_legal(ability):
            return False
        
        # Check moves (case-insensitive)
        moves = pokemon.get("moves", [])
        if len(moves) != 4:
            return False
        
        for move in moves:
            move_lower = move.lower().replace(" ", "-")
            if not self._is_move_legal(move_lower):
                return False
        
        return True

    def _is_move_legal(self, move_name: str) -> bool:
        """Check move legality while tolerating known naming aliases."""
        aliases = {move_name}
        if move_name == "protect":
            aliases.add("protected")
        elif move_name == "protected":
            aliases.add("protect")
        return any(alias in self.regulation.legal_moves for alias in aliases)

    def _is_ability_legal(self, ability_name: str) -> bool:
        """Check ability legality while tolerating known naming aliases."""
        aliases = {ability_name}
        if ability_name == "unseen-hand":
            aliases.add("unseen-fist")
        elif ability_name == "unseen-fist":
            aliases.add("unseen-hand")
        return any(alias in self.regulation.legal_abilities for alias in aliases)
    
    def sample_team(self) -> Dict[str, Any]:
        """Sample a random team from the pool.
        
        Returns:
            A team dict with 6 Pokemon, each having name, item, ability, moves
            
        Raises:
            ValueError: If pool is empty
        """
        self._load_teams()
        
        if not self._teams:
            raise ValueError("No legal teams available in pool")
        
        return random.choice(self._teams)
    
    def __len__(self) -> int:
        """Return the number of legal teams in the pool."""
        self._load_teams()
        return len(self._teams) if self._teams else 0
