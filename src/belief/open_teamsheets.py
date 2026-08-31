"""Integration with open-teamsheets for fetching Pokemon team data."""

from typing import Optional, List, Dict, Any
import requests
from src.belief.belief_state import PokemonBeliefState


def fetch_team_from_url(url: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch team data from an open-teamsheets URL.
    
    Args:
        url: URL to fetch team data from (e.g., open-teamsheets API endpoint)
        
    Returns:
        List of Pokemon data dictionaries, or None if fetch fails
    """
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Try to extract team data from various possible response formats
        if isinstance(data, dict):
            if "team" in data:
                return data["team"]
            elif "pokemon" in data:
                return data["pokemon"]
            else:
                # Assume the entire response is the team
                return [data] if "name" in data else None
        elif isinstance(data, list):
            return data
        else:
            return None
            
    except (requests.exceptions.RequestException, ConnectionError):
        # Network error, timeout, connection error, etc.
        return None
    except (ValueError, KeyError):
        # JSON parsing error or missing expected keys
        return None


def create_belief_state_from_teamsheet(pokemon_data: Dict[str, Any]) -> PokemonBeliefState:
    """Convert open-teamsheets Pokemon data to a deterministic PokemonBeliefState.
    
    When a Pokemon is fetched from an open teamsheet, its attributes are known
    with certainty (not uncertain), so belief bounds are exact.
    
    Args:
        pokemon_data: Dictionary with Pokemon attributes from teamsheet
        
    Returns:
        PokemonBeliefState with known (certain) values
    """
    name = pokemon_data.get("name", "Unknown")
    
    # Extract moves (known with certainty)
    moves = set()
    if "moves" in pokemon_data and isinstance(pokemon_data["moves"], list):
        moves = set(pokemon_data["moves"])
    
    # Extract item (known with certainty)
    items = set()
    if "item" in pokemon_data and pokemon_data["item"]:
        items = {pokemon_data["item"]}
    
    # Extract ability (known with certainty)
    abilities = set()
    if "ability" in pokemon_data and pokemon_data["ability"]:
        abilities = {pokemon_data["ability"]}
    
    # Extract tera type (known with certainty)
    tera_types = set()
    if "tera_type" in pokemon_data and pokemon_data["tera_type"]:
        tera_types = {pokemon_data["tera_type"]}
    
    # Create belief state with unknown stats (use full range)
    # Stats are not typically specified in open-teamsheets, so we leave them unknown
    belief = PokemonBeliefState(
        name=name,
        moves=moves,
        items=items,
        abilities=abilities,
        tera_types=tera_types,
    )
    
    return belief
