"""Feature encoding for Pokemon battle states to PyTorch tensors."""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

from src.features.tensors import (
    TensorShapes,
    DEFAULT_TENSOR_SHAPES,
    POKEMON_INDEX_MAP,
    MOVE_INDEX_MAP,
    ITEM_INDEX_MAP,
    ABILITY_INDEX_MAP,
    STATUS_INDEX_MAP,
    WEATHER_INDEX_MAP,
    TERRAIN_INDEX_MAP,
    TERA_TYPE_INDEX_MAP,
)
from src.belief.belief_state import PokemonBeliefState
from src.utils.config import RegulationConfig


@dataclass
class EncodedBattleState:
    """Encoded battle state with all feature tensors.
    
    Attributes:
        team_tensor: Full team encoding (batch, max_team_size, entity_dim)
        active_pokemon_tensor: Active Pokemon encoding (batch, active_count, entity_dim)
        field_tensor: Field state encoding (batch, field_feature_dim)
        opponent_belief_tensor: Opponent belief state (batch, opponent_team_size, belief_dim)
        regulation_embedding: Regulation token (batch, regulation_embedding_dim)
        move_masks: Legal move masks (batch, active_count, max_move_choices)
        switch_masks: Legal switch masks (batch, max_switch_choices)
        target_masks: Legal target masks (batch, active_count, max_target_choices)
    """
    team_tensor: torch.Tensor
    active_pokemon_tensor: torch.Tensor
    field_tensor: torch.Tensor
    opponent_belief_tensor: torch.Tensor
    regulation_embedding: torch.Tensor
    move_masks: torch.Tensor
    switch_masks: torch.Tensor
    target_masks: torch.Tensor


class FeatureEncoder:
    """Encodes battle observations into feature tensors for Transformer-based policies.
    
    This encoder converts Pokemon battle observations (teams, field state, etc.)
    and belief states (uncertainty about opponent) into PyTorch tensors suitable
    for transformer-based neural networks.
    
    Architecture:
    - Entity-based: Teams are sequences of Pokemon entities
    - Hierarchical: Species embeddings + stats + moves
    - Belief-aware: Incorporates uncertainty ranges from belief state
    - Regulation-aware: Includes regulation token for legal constraints
    """
    
    def __init__(
        self,
        regulation_config: RegulationConfig,
        tensor_shapes: Optional[TensorShapes] = None,
        device: str = "cpu",
    ):
        """Initialize feature encoder.
        
        Args:
            regulation_config: VGC regulation defining legal Pokemon/moves/items.
            tensor_shapes: Custom tensor shape configuration (uses defaults if None).
            device: PyTorch device ("cpu" or "cuda").
        """
        self.regulation_config = regulation_config
        self.tensor_shapes = tensor_shapes or DEFAULT_TENSOR_SHAPES
        self.device = device
        
        # Initialize embedding layers
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding layers for species, moves, items, abilities."""
        self.species_embedding = nn.Embedding(
            len(POKEMON_INDEX_MAP),
            self.tensor_shapes.pokemon_embedding_dim,
        )
        self.move_embedding = nn.Embedding(
            len(MOVE_INDEX_MAP),
            self.tensor_shapes.move_embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            len(ITEM_INDEX_MAP) + 1,  # +1 for padding/none
            self.tensor_shapes.item_embedding_dim,
        )
        self.ability_embedding = nn.Embedding(
            len(ABILITY_INDEX_MAP) + 1,  # +1 for padding/none
            self.tensor_shapes.ability_embedding_dim,
        )
        self.status_embedding = nn.Embedding(
            len(STATUS_INDEX_MAP),
            8,  # Small embedding for status
        )
        self.weather_embedding = nn.Embedding(
            len(WEATHER_INDEX_MAP),
            8,
        )
        self.terrain_embedding = nn.Embedding(
            len(TERRAIN_INDEX_MAP),
            8,
        )
    
    def encode(
        self,
        observation: Dict[str, Any],
        belief_state: Dict[str, PokemonBeliefState],
    ) -> EncodedBattleState:
        """Encode battle observation and belief state to tensors.
        
        Args:
            observation: Battle observation dict with team, active_pokemon, field, etc.
            belief_state: Dict mapping opponent Pokemon names to belief states.
        
        Returns:
            EncodedBattleState with all feature tensors (batch_size=1).
        """
        # Encode team
        team_tensor = self._encode_team(observation.get("team", []))
        
        # Encode active Pokemon
        active_pokemon_tensor = self._encode_active_pokemon(
            observation.get("active_pokemon", []),
        )
        
        # Encode field state
        field_tensor = self._encode_field(observation.get("field", {}))
        
        # Encode opponent belief
        opponent_belief_tensor = self._encode_opponent_belief(
            observation.get("opponent_team", []),
            belief_state,
        )
        
        # Encode regulation
        regulation_embedding = self._encode_regulation()
        
        # Encode action masks
        move_masks, switch_masks, target_masks = self._encode_action_masks(
            observation.get("team", []),
            observation.get("active_pokemon", []),
        )
        
        return EncodedBattleState(
            team_tensor=team_tensor,
            active_pokemon_tensor=active_pokemon_tensor,
            field_tensor=field_tensor,
            opponent_belief_tensor=opponent_belief_tensor,
            regulation_embedding=regulation_embedding,
            move_masks=move_masks,
            switch_masks=switch_masks,
            target_masks=target_masks,
        )
    
    def _encode_team(self, team: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode team as sequence of Pokemon entities.
        
        Args:
            team: List of Pokemon dicts with species, hp, status, moves, etc.
        
        Returns:
            Tensor of shape (1, max_team_size, entity_dim).
        """
        entities = []
        
        for pokemon_dict in team[:self.tensor_shapes.max_team_size]:
            entity = self._encode_pokemon_entity(pokemon_dict)
            entities.append(entity)
        
        # Pad to max_team_size
        entity_dim = self.tensor_shapes.get_pokemon_entity_dim()
        while len(entities) < self.tensor_shapes.max_team_size:
            entities.append(torch.zeros(entity_dim, dtype=torch.float32))
        
        # Stack into tensor (1, max_team_size, entity_dim)
        team_tensor = torch.stack(entities, dim=0).unsqueeze(0)
        return team_tensor.to(self.device)
    
    def _encode_active_pokemon(self, active: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode active Pokemon.
        
        Args:
            active: List of active Pokemon dicts.
        
        Returns:
            Tensor of shape (1, max_active_pokemon, entity_dim).
        """
        entities = []
        
        for pokemon_dict in active[:self.tensor_shapes.max_active_pokemon]:
            entity = self._encode_pokemon_entity(pokemon_dict)
            entities.append(entity)
        
        # Pad to max_active_pokemon
        entity_dim = self.tensor_shapes.get_pokemon_entity_dim()
        while len(entities) < self.tensor_shapes.max_active_pokemon:
            entities.append(torch.zeros(entity_dim, dtype=torch.float32))
        
        active_tensor = torch.stack(entities, dim=0).unsqueeze(0)
        return active_tensor.to(self.device)
    
    def _encode_pokemon_entity(self, pokemon_dict: Dict[str, Any]) -> torch.Tensor:
        """Encode single Pokemon entity.
        
        Combines:
        - Species embedding
        - HP percentage
        - Status code
        - Item embedding
        - Ability embedding
        - Move embeddings
        - Stat normalized values
        - Stat boosts
        - Tera used flag
        
        Args:
            pokemon_dict: Pokemon data dict.
        
        Returns:
            Entity tensor of shape (entity_dim,).
        """
        components = []
        
        # Species embedding
        species_name = pokemon_dict.get("species", "pikachu").lower()
        species_idx = POKEMON_INDEX_MAP.get(species_name, 0)
        species_emb = self.species_embedding(torch.tensor([species_idx], device=self.device))
        components.append(species_emb.squeeze(0))
        
        # HP percentage (normalized to [0, 1])
        hp = pokemon_dict.get("hp", 100)
        max_hp = pokemon_dict.get("max_hp", 100)
        hp_pct = float(hp) / float(max_hp) if max_hp > 0 else 0.0
        hp_pct = torch.clamp(torch.tensor([hp_pct], dtype=torch.float32), 0, 1)
        components.append(hp_pct)
        
        # Status (one-hot or embedded)
        status_name = pokemon_dict.get("status", "none").lower()
        status_idx = STATUS_INDEX_MAP.get(status_name, 0)
        status_emb = self.status_embedding(torch.tensor([status_idx], device=self.device))
        components.append(status_emb.squeeze(0))
        
        # Item embedding
        item_name = pokemon_dict.get("item", "none").lower()
        item_idx = ITEM_INDEX_MAP.get(item_name, len(ITEM_INDEX_MAP))
        item_emb = self.item_embedding(torch.tensor([item_idx], device=self.device))
        components.append(item_emb.squeeze(0))
        
        # Ability embedding
        ability_name = pokemon_dict.get("ability", "none").lower()
        ability_idx = ABILITY_INDEX_MAP.get(ability_name, len(ABILITY_INDEX_MAP))
        ability_emb = self.ability_embedding(torch.tensor([ability_idx], device=self.device))
        components.append(ability_emb.squeeze(0))
        
        # Move embeddings (exactly max_moves_per_pokemon, padded if necessary)
        moves = pokemon_dict.get("moves", [])
        move_embs = []
        for i in range(self.tensor_shapes.max_moves_per_pokemon):
            if i < len(moves):
                move_name = moves[i].lower()
                move_idx = MOVE_INDEX_MAP.get(move_name, 0)
                move_emb = self.move_embedding(torch.tensor([move_idx], device=self.device))
                move_embs.append(move_emb.squeeze(0))
            else:
                # Padding for missing moves
                move_emb = torch.zeros(
                    self.tensor_shapes.move_embedding_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
                move_embs.append(move_emb)
        
        # Concatenate all move embeddings into one component
        if move_embs:
            all_moves = torch.cat(move_embs, dim=0)
            components.append(all_moves)
        
        # Stat values (normalized to [0, 1] assuming stats in range 0-255)
        stats = pokemon_dict.get("stats", {})
        stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        stat_values = []
        for stat_key in stat_keys:
            stat_val = float(stats.get(stat_key, 100))
            normalized = torch.clamp(torch.tensor([stat_val / 255.0], dtype=torch.float32), 0, 1)
            stat_values.append(normalized)
        
        stat_tensor = torch.cat(stat_values, dim=0) if stat_values else torch.zeros(6, dtype=torch.float32, device=self.device)
        components.append(stat_tensor)
        
        # Stat boosts (-6 to +6 normalized to [-1, 1])
        boosts = pokemon_dict.get("boosts", {})
        boost_values = []
        for stat_key in stat_keys:
            boost_val = float(boosts.get(stat_key, 0))
            normalized = torch.clamp(torch.tensor([boost_val / 6.0], dtype=torch.float32), -1, 1)
            boost_values.append(normalized)
        
        boost_tensor = torch.cat(boost_values, dim=0) if boost_values else torch.zeros(6, dtype=torch.float32, device=self.device)
        components.append(boost_tensor)
        
        # Tera used flag (binary)
        tera_used = float(pokemon_dict.get("tera_used", False))
        tera_tensor = torch.tensor([tera_used], dtype=torch.float32, device=self.device)
        components.append(tera_tensor)
        
        # Concatenate all components
        entity = torch.cat(components, dim=0).to(self.device)
        return entity
    
    def _encode_field(self, field_dict: Dict[str, Any]) -> torch.Tensor:
        """Encode field state.
        
        Includes: weather, terrain, trick room, reflect, light screen, tailwind.
        
        Args:
            field_dict: Field state dict.
        
        Returns:
            Tensor of shape (1, field_feature_dim).
        """
        components = []
        
        # Weather embedding
        weather_name = field_dict.get("weather", "none").lower()
        weather_idx = WEATHER_INDEX_MAP.get(weather_name, 0)
        weather_emb = self.weather_embedding(torch.tensor([weather_idx], device=self.device))
        components.append(weather_emb.squeeze(0))
        
        # Terrain embedding
        terrain_name = field_dict.get("terrain", "none").lower()
        terrain_idx = TERRAIN_INDEX_MAP.get(terrain_name, 0)
        terrain_emb = self.terrain_embedding(torch.tensor([terrain_idx], device=self.device))
        components.append(terrain_emb.squeeze(0))
        
        # Binary field conditions
        trick_room = float(field_dict.get("trick_room", False))
        reflect = float(field_dict.get("reflect", False))
        light_screen = float(field_dict.get("light_screen", False))
        tailwind = float(field_dict.get("tailwind", False))
        
        binary_conditions = torch.tensor(
            [trick_room, reflect, light_screen, tailwind],
            dtype=torch.float32,
            device=self.device,
        )
        components.append(binary_conditions)
        
        # Concatenate and pad/truncate to field_feature_dim
        field_features = torch.cat(components, dim=0)
        
        if len(field_features) < self.tensor_shapes.field_feature_dim:
            padding = torch.zeros(
                self.tensor_shapes.field_feature_dim - len(field_features),
                dtype=torch.float32,
                device=self.device,
            )
            field_features = torch.cat([field_features, padding], dim=0)
        else:
            field_features = field_features[:self.tensor_shapes.field_feature_dim]
        
        return field_features.unsqueeze(0).to(self.device)
    
    def _encode_opponent_belief(
        self,
        opponent_team: List[Dict[str, Any]],
        belief_state: Dict[str, PokemonBeliefState],
    ) -> torch.Tensor:
        """Encode opponent team belief state.
        
        Incorporates uncertainty ranges and possible movesets from belief.
        
        Args:
            opponent_team: List of observed opponent Pokemon.
            belief_state: Belief state for opponent Pokemon.
        
        Returns:
            Tensor of shape (1, max_possible_opponents, belief_feature_dim).
        """
        belief_entities = []
        
        for i in range(self.tensor_shapes.max_possible_opponents):
            if i < len(opponent_team):
                opp_pokemon = opponent_team[i]
                species = opp_pokemon.get("species", "").lower()
                belief = belief_state.get(species)
                belief_entity = self._encode_belief_entity(opp_pokemon, belief)
            else:
                # Padding
                belief_entity = torch.zeros(
                    self.tensor_shapes.belief_feature_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
            
            belief_entities.append(belief_entity)
        
        belief_tensor = torch.stack(belief_entities, dim=0).unsqueeze(0)
        return belief_tensor.to(self.device)
    
    def _encode_belief_entity(
        self,
        pokemon_dict: Dict[str, Any],
        belief: Optional[PokemonBeliefState],
    ) -> torch.Tensor:
        """Encode belief for single opponent Pokemon.
        
        Uses belief state to encode uncertainty ranges for stats and
        possible moves/items/abilities.
        
        Args:
            pokemon_dict: Observed Pokemon dict.
            belief: Belief state (None if no uncertainty info).
        
        Returns:
            Belief entity tensor of shape (belief_feature_dim,).
        """
        components = []
        
        if belief is None:
            # Default belief entity with no uncertainty info
            return torch.zeros(
                self.tensor_shapes.belief_feature_dim,
                dtype=torch.float32,
                device=self.device,
            )
        
        # HP range (encode as min, max, expected)
        hp_min = belief.hp_min / 255.0
        hp_max = belief.hp_max / 255.0
        hp_expected = belief.expected_hp() / 255.0
        components.extend([
            torch.tensor([hp_min], dtype=torch.float32, device=self.device),
            torch.tensor([hp_max], dtype=torch.float32, device=self.device),
            torch.tensor([hp_expected], dtype=torch.float32, device=self.device),
        ])
        
        # Speed range (important for turn order)
        speed_min = belief.speed_min / 255.0
        speed_max = belief.speed_max / 255.0
        speed_expected = belief.expected_speed() / 255.0
        components.extend([
            torch.tensor([speed_min], dtype=torch.float32, device=self.device),
            torch.tensor([speed_max], dtype=torch.float32, device=self.device),
            torch.tensor([speed_expected], dtype=torch.float32, device=self.device),
        ])
        
        # Attack and SpA ranges
        atk_expected = belief.expected_attack() / 255.0
        spa_expected = belief.expected_spa() / 255.0
        components.extend([
            torch.tensor([atk_expected], dtype=torch.float32, device=self.device),
            torch.tensor([spa_expected], dtype=torch.float32, device=self.device),
        ])
        
        # Defense and SpD ranges
        def_expected = belief.expected_defense() / 255.0
        spd_expected = belief.expected_spd() / 255.0
        components.extend([
            torch.tensor([def_expected], dtype=torch.float32, device=self.device),
            torch.tensor([spd_expected], dtype=torch.float32, device=self.device),
        ])
        
        # Number of possible moves/items/abilities (as proxy for uncertainty)
        num_moves = len(belief.moves) if belief.moves else 0
        num_items = len(belief.items) if belief.items else 0
        num_abilities = len(belief.abilities) if belief.abilities else 0
        
        components.extend([
            torch.tensor([num_moves / 4.0], dtype=torch.float32, device=self.device),
            torch.tensor([num_items / 2.0], dtype=torch.float32, device=self.device),
            torch.tensor([num_abilities / 2.0], dtype=torch.float32, device=self.device),
        ])
        
        # Concatenate and pad to belief_feature_dim
        belief_entity = torch.cat(components, dim=0)
        
        if len(belief_entity) < self.tensor_shapes.belief_feature_dim:
            padding = torch.zeros(
                self.tensor_shapes.belief_feature_dim - len(belief_entity),
                dtype=torch.float32,
                device=self.device,
            )
            belief_entity = torch.cat([belief_entity, padding], dim=0)
        else:
            belief_entity = belief_entity[:self.tensor_shapes.belief_feature_dim]
        
        return belief_entity.to(self.device)
    
    def _encode_regulation(self) -> torch.Tensor:
        """Encode regulation as embedding.
        
        Creates a unique embedding for the regulation ID.
        
        Returns:
            Tensor of shape (1, regulation_embedding_dim).
        """
        # Create a deterministic but pseudo-random embedding based on regulation_id
        reg_hash = hash(self.regulation_config.regulation_id) % 1000
        torch.manual_seed(reg_hash)
        
        reg_embedding = torch.randn(
            1,
            self.tensor_shapes.regulation_embedding_dim,
            dtype=torch.float32,
            device=self.device,
        ) * 0.1  # Scale to reasonable magnitude
        
        return reg_embedding.to(self.device)
    
    def _encode_action_masks(
        self,
        team: List[Dict[str, Any]],
        active_pokemon: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode legal action masks for moves, switches, and targets.
        
        Args:
            team: Full team list.
            active_pokemon: Currently active Pokemon.
        
        Returns:
            Tuple of (move_masks, switch_masks, target_masks).
        """
        # Move masks: (1, max_active_pokemon, max_move_choices)
        move_masks = torch.zeros(
            1,
            self.tensor_shapes.max_active_pokemon,
            self.tensor_shapes.max_move_choices,
            dtype=torch.float32,
            device=self.device,
        )
        
        for i, pokemon in enumerate(active_pokemon[:self.tensor_shapes.max_active_pokemon]):
            moves = pokemon.get("moves", [])
            for j, move in enumerate(moves[:self.tensor_shapes.max_move_choices]):
                move_masks[0, i, j] = 1.0
        
        # Switch masks: (1, max_switch_choices)
        switch_masks = torch.zeros(
            1,
            self.tensor_shapes.max_switch_choices,
            dtype=torch.float32,
            device=self.device,
        )
        
        # Mark switches for benched Pokemon
        active_species = {p.get("species", "") for p in active_pokemon}
        bench_count = 0
        for pokemon in team:
            species = pokemon.get("species", "")
            if species not in active_species and bench_count < self.tensor_shapes.max_switch_choices:
                switch_masks[0, bench_count] = 1.0
                bench_count += 1
        
        # Target masks: (1, max_active_pokemon, max_target_choices)
        # In doubles, can target either opponent's active Pokemon
        target_masks = torch.ones(
            1,
            self.tensor_shapes.max_active_pokemon,
            self.tensor_shapes.max_target_choices,
            dtype=torch.float32,
            device=self.device,
        ) * 0.5  # Both targets are usually valid
        
        return move_masks.to(self.device), switch_masks.to(self.device), target_masks.to(self.device)
