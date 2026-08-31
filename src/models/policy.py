"""Transformer-based policy network for Pokemon battle decision-making.

This module implements a multi-task learning approach with a shared Transformer
encoder and separate prediction heads for:
- Main task: Action selection (move pair + targets)
- Auxiliary tasks: Win prediction, opponent moveset prediction, opponent item
  prediction, and battle length estimation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyOutput:
    """Output from the policy network."""

    action_logits: torch.Tensor  # (batch_size, 126)
    win_probability: torch.Tensor  # (batch_size, 1)
    opponent_moves: torch.Tensor  # (batch_size, team_size, 4, num_moves)
    opponent_items: torch.Tensor  # (batch_size, team_size, num_items)
    battle_length: torch.Tensor  # (batch_size, num_turns_buckets)


class TransformerPolicy(nn.Module):
    """Transformer-based policy network for Pokemon battle actions.

    Architecture:
    - Embedding layers for Pokemon, moves, items, and abilities
    - Positional encoding for team order
    - Transformer encoder over team entities
    - Multi-head attention over field state
    - Separate prediction heads for main and auxiliary tasks
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        ff_dim: int = 512,
        vocab_size_pokemon: int = 1025,
        vocab_size_moves: int = 656,
        vocab_size_items: int = 376,
        vocab_size_abilities: int = 267,
        max_team_size: int = 6,
        field_feature_dim: int = 32,
        dropout: float = 0.1,
    ):
        """Initialize the policy network.

        Args:
            embedding_dim: Dimension of embeddings and transformer
            num_transformer_layers: Number of transformer encoder layers
            num_attention_heads: Number of attention heads
            ff_dim: Feedforward dimension in transformer
            vocab_size_pokemon: Number of unique Pokemon
            vocab_size_moves: Number of unique moves
            vocab_size_items: Number of unique items
            vocab_size_abilities: Number of unique abilities
            max_team_size: Maximum team size (6 for VGC)
            field_feature_dim: Dimension of field features
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size_moves = vocab_size_moves
        self.vocab_size_items = vocab_size_items
        self.max_team_size = max_team_size

        self.pokemon_embedding = nn.Embedding(vocab_size_pokemon, embedding_dim, padding_idx=0)
        self.move_embedding = nn.Embedding(vocab_size_moves, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(vocab_size_items, embedding_dim, padding_idx=0)
        self.ability_embedding = nn.Embedding(vocab_size_abilities, embedding_dim, padding_idx=0)
        self.status_embedding = nn.Embedding(8, embedding_dim)

        self.team_position_embedding = nn.Embedding(max_team_size, embedding_dim)

        self.team_entity_projection = nn.Sequential(
            nn.Linear(embedding_dim * 5 + 1, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.field_projection = nn.Linear(field_feature_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        self.team_attention = nn.MultiheadAttention(
            embedding_dim, num_attention_heads, dropout=dropout, batch_first=True
        )

        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 126),
        )

        self.win_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid(),
        )

        self.opponent_moves_head = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 4 * vocab_size_moves),
        )

        self.opponent_items_head = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, vocab_size_items),
        )

        self.battle_length_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 50),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pokemon_ids: torch.Tensor,
        move_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ability_ids: torch.Tensor,
        team_mask: torch.Tensor,
        field_features: torch.Tensor,
        hp_fractions: torch.Tensor,
        status: torch.Tensor,
    ) -> PolicyOutput:
        """Forward pass through policy network.

        Args:
            pokemon_ids: (batch_size, team_size) - Pokemon IDs
            move_ids: (batch_size, team_size, 4) - Move IDs for each Pokemon
            item_ids: (batch_size, team_size) - Item IDs
            ability_ids: (batch_size, team_size) - Ability IDs
            team_mask: (batch_size, team_size) - Boolean mask for valid team members
            field_features: (batch_size, field_feature_dim) - Field state features
            hp_fractions: (batch_size, team_size) - HP fractions [0, 1]
            status: (batch_size, team_size) - Status condition indices

        Returns:
            PolicyOutput with action logits and auxiliary predictions
        """
        batch_size, team_size = pokemon_ids.shape

        pokemon_emb = self.pokemon_embedding(pokemon_ids)
        move_emb = self.move_embedding(move_ids).mean(dim=2)
        item_emb = self.item_embedding(item_ids)
        ability_emb = self.ability_embedding(ability_ids)
        status_emb = self.status_embedding(status)

        position_emb = self.team_position_embedding(
            torch.arange(team_size, device=pokemon_ids.device).unsqueeze(0).expand(batch_size, -1)
        )

        team_features = torch.cat(
            [pokemon_emb, move_emb, item_emb, ability_emb, status_emb, hp_fractions.unsqueeze(2)],
            dim=2,
        )

        team_entities = self.team_entity_projection(team_features) + position_emb

        src_key_padding_mask = ~team_mask

        team_encoded = self.transformer_encoder(
            team_entities, src_key_padding_mask=src_key_padding_mask
        )

        field_emb = self.field_projection(field_features)

        field_expanded = field_emb.unsqueeze(1).expand(batch_size, team_size, -1)

        attended, _ = self.team_attention(
            field_expanded, team_encoded, team_encoded, key_padding_mask=src_key_padding_mask
        )

        team_global = team_encoded.masked_fill(src_key_padding_mask.unsqueeze(2), 0).sum(dim=1)
        team_global = team_global / team_mask.sum(dim=1, keepdim=True).clamp(min=1)

        attended_global = attended.masked_fill(src_key_padding_mask.unsqueeze(2), 0).sum(dim=1)
        attended_global = attended_global / team_mask.sum(dim=1, keepdim=True).clamp(min=1)

        combined = torch.cat([team_global, attended_global], dim=1)

        action_logits = self.action_head(combined)

        win_probability = self.win_head(combined)

        opponent_moves_raw = self.opponent_moves_head(team_encoded)
        opponent_moves = opponent_moves_raw.reshape(
            batch_size, team_size, 4, self.vocab_size_moves
        )

        opponent_items = self.opponent_items_head(team_encoded)

        battle_length = self.battle_length_head(combined)

        return PolicyOutput(
            action_logits=action_logits,
            win_probability=win_probability,
            opponent_moves=opponent_moves,
            opponent_items=opponent_items,
            battle_length=battle_length,
        )
