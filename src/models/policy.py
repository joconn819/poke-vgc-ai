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

from src.belief.history_encoder import BattleHistoryEncoder


@dataclass
class PolicyOutput:
    """Output from the policy network."""

    action_logits: torch.Tensor  # (batch_size, 126)
    win_probability: torch.Tensor  # (batch_size, 1)
    opponent_moves: torch.Tensor  # (batch_size, team_size, 4, num_moves)
    opponent_items: torch.Tensor  # (batch_size, team_size, num_items)
    battle_length: torch.Tensor  # (batch_size, num_turns_buckets)
    value: Optional[torch.Tensor] = None  # (batch_size, 1), unbounded critic value
    native_action_logits: Optional[torch.Tensor] = None  # (batch_size, 2, 126)
    state_context: Optional[torch.Tensor] = None  # (batch_size, 2 * embedding_dim)
    opponent_abilities: Optional[torch.Tensor] = None  # (batch, team, abilities)


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
        history_encoder: Optional[BattleHistoryEncoder] = None,
        native_action_feature_dim: int = 0,
        native_action_feature_variant: str | None = None,
        public_context_dim: int = 0,
        token_vocab_version: int = 1,
        state_conditioned_joint_action: bool = False,
        predict_opponent_abilities: bool = False,
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
        if native_action_feature_dim < 0:
            raise ValueError("native_action_feature_dim must be non-negative")
        if public_context_dim < 0:
            raise ValueError("public_context_dim must be non-negative")
        if token_vocab_version not in {0, 1}:
            raise ValueError("token_vocab_version must be 0 or 1")

        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.ff_dim = ff_dim
        self.vocab_size_pokemon = vocab_size_pokemon
        self.vocab_size_moves = vocab_size_moves
        self.vocab_size_items = vocab_size_items
        self.vocab_size_abilities = vocab_size_abilities
        self.max_team_size = max_team_size
        self.field_feature_dim = field_feature_dim
        self.dropout_rate = dropout
        self.history_encoder = history_encoder
        self.native_action_feature_dim = native_action_feature_dim
        self.native_action_feature_variant = native_action_feature_variant
        self.public_context_dim = int(public_context_dim)
        self.token_vocab_version = token_vocab_version
        self.state_conditioned_joint_action = bool(state_conditioned_joint_action)
        self.predict_opponent_abilities = bool(predict_opponent_abilities)
        self.history_projection = (
            nn.Linear(history_encoder.hidden_dim, embedding_dim)
            if history_encoder is not None
            else None
        )

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
        self.public_context_projection = (
            nn.Linear(public_context_dim, embedding_dim, bias=False)
            if public_context_dim else None
        )
        if self.public_context_projection is not None:
            nn.init.zeros_(self.public_context_projection.weight)
        # Start as a no-op so older checkpoints remain behaviorally aligned,
        # then let PPO learn how much opponent context should affect decisions.
        self.opponent_context_scale = nn.Parameter(torch.zeros(1))
        self.opponent_context_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.opponent_context_projection.weight)

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
        self.native_action_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 2 * 126),
        )
        self.previous_action_embedding = nn.Embedding(127, embedding_dim)
        self.previous_action_projection = nn.Linear(embedding_dim, 126, bias=False)
        nn.init.zeros_(self.previous_action_projection.weight)
        self.state_conditioned_joint_action_head = (
            nn.Sequential(
                nn.Linear(embedding_dim * 3, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 126),
            )
            if self.state_conditioned_joint_action
            else None
        )
        if self.state_conditioned_joint_action_head is not None:
            nn.init.zeros_(self.state_conditioned_joint_action_head[-1].weight)
            nn.init.zeros_(self.state_conditioned_joint_action_head[-1].bias)
        self.native_action_feature_head = (
            nn.Sequential(
                nn.Linear(embedding_dim * 2 + native_action_feature_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 1),
            )
            if native_action_feature_dim > 0
            else None
        )
        if (
            native_action_feature_dim == 5
            and native_action_feature_variant == "damage_ko_friendly_fire"
        ):
            # Preserve the source policy's action scores on the first forward
            # pass, apart from the explicit negative prior for damaging an
            # ally. PPO can then learn the feature head from outcomes.
            nn.init.zeros_(self.native_action_feature_head[-1].weight)
            nn.init.zeros_(self.native_action_feature_head[-1].bias)
        self.native_action_friendly_fire_scale = (
            nn.Parameter(torch.tensor(-1.5))
            if native_action_feature_dim == 5
            and native_action_feature_variant == "damage_ko_friendly_fire"
            else None
        )
        self.native_action_counterplay_scale = (
            nn.Parameter(torch.tensor(0.15))
            if native_action_feature_dim == 7
            and native_action_feature_variant == "damage_ko_counterplay"
            else None
        )
        self.native_action_damage_scale = (
            nn.Parameter(torch.tensor(0.15))
            if native_action_feature_dim == 7
            and native_action_feature_variant == "damage_ko_counterplay"
            else None
        )

        self.win_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid(),
        )
        self.value_head = nn.Linear(embedding_dim * 2, 1)

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
        self.opponent_abilities_head = (
            nn.Sequential(
                nn.Linear(embedding_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, vocab_size_abilities),
            )
            if self.predict_opponent_abilities else None
        )

        self.battle_length_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 50),
        )

        self.dropout = nn.Dropout(dropout)

    def condition_native_second_action_logits(
        self,
        base_logits: torch.Tensor,
        state_context: torch.Tensor,
        previous_action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Condition slot-two logits on slot one's choice and the current state."""
        previous = previous_action_indices.long().reshape(-1)
        if base_logits.shape != (previous.shape[0], 126):
            raise ValueError("base_logits must have shape (batch, 126)")
        if state_context.shape != (previous.shape[0], self.embedding_dim * 2):
            raise ValueError("state_context must have shape (batch, 2 * embedding_dim)")
        previous_context = self.previous_action_embedding(previous)
        adjustment = self.previous_action_projection(previous_context)
        if self.state_conditioned_joint_action_head is not None:
            adjustment = adjustment + self.state_conditioned_joint_action_head(
                torch.cat([state_context, previous_context], dim=-1)
            )
        return base_logits + adjustment

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
        history_observations: Optional[torch.Tensor] = None,
        belief_features: Optional[torch.Tensor] = None,
        native_action_features: Optional[torch.Tensor] = None,
        opponent_pokemon_ids: Optional[torch.Tensor] = None,
        opponent_move_ids: Optional[torch.Tensor] = None,
        opponent_item_ids: Optional[torch.Tensor] = None,
        opponent_ability_ids: Optional[torch.Tensor] = None,
        opponent_team_mask: Optional[torch.Tensor] = None,
        opponent_hp_fractions: Optional[torch.Tensor] = None,
        opponent_status: Optional[torch.Tensor] = None,
        previous_action_indices: Optional[torch.Tensor] = None,
        opponent_recent_move_ids: Optional[torch.Tensor] = None,
        public_context_features: Optional[torch.Tensor] = None,
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
                history_observations: Optional (batch_size, time, observation_dim)
                    sequence for recurrent battle context
                belief_features: Optional (batch_size, time, belief_dim) features
                    aligned with ``history_observations``

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

        if field_features.shape[-1] < self.field_feature_dim:
            field_features = F.pad(field_features, (0, self.field_feature_dim - field_features.shape[-1]))
        elif field_features.shape[-1] > self.field_feature_dim:
            field_features = field_features[..., : self.field_feature_dim]
        field_emb = self.field_projection(field_features)
        if self.public_context_projection is not None:
            if public_context_features is None:
                public_context_features = field_features.new_zeros(
                    (batch_size, self.public_context_dim)
                )
            if public_context_features.shape != (batch_size, self.public_context_dim):
                raise ValueError(
                    "public_context_features must have shape "
                    f"(batch, {self.public_context_dim})"
                )
            field_emb = field_emb + self.public_context_projection(public_context_features)
        elif public_context_features is not None:
            raise ValueError("model checkpoint has no public-context input")

        field_expanded = field_emb.unsqueeze(1).expand(batch_size, team_size, -1)

        attended, _ = self.team_attention(
            field_expanded, team_encoded, team_encoded, key_padding_mask=src_key_padding_mask
        )

        team_global = team_encoded.masked_fill(src_key_padding_mask.unsqueeze(2), 0).sum(dim=1)
        team_global = team_global / team_mask.sum(dim=1, keepdim=True).clamp(min=1)

        attended_global = attended.masked_fill(src_key_padding_mask.unsqueeze(2), 0).sum(dim=1)
        attended_global = attended_global / team_mask.sum(dim=1, keepdim=True).clamp(min=1)

        opponent_encoded = None
        opponent_inputs = (
            opponent_pokemon_ids, opponent_move_ids, opponent_item_ids,
            opponent_ability_ids, opponent_team_mask, opponent_hp_fractions, opponent_status,
        )
        if any(value is not None for value in opponent_inputs):
            if any(value is None for value in opponent_inputs):
                raise ValueError("opponent team inputs must be provided together")
            opponent_features = torch.cat(
                [
                    self.pokemon_embedding(opponent_pokemon_ids),
                    self.move_embedding(opponent_move_ids).mean(dim=2),
                    self.item_embedding(opponent_item_ids),
                    self.ability_embedding(opponent_ability_ids),
                    self.status_embedding(opponent_status),
                    opponent_hp_fractions.unsqueeze(2),
                ],
                dim=2,
            )
            opponent_entities = self.team_entity_projection(opponent_features) + position_emb
            if opponent_recent_move_ids is not None:
                if opponent_recent_move_ids.shape != (batch_size, team_size, 2):
                    raise ValueError(
                        "opponent_recent_move_ids must have shape "
                        "(batch_size, team_size, 2)"
                    )
                recent_embeddings = self.move_embedding(opponent_recent_move_ids)
                recency_weights = recent_embeddings.new_tensor([1.0, 0.5]).view(1, 1, 2, 1)
                recent_context = (recent_embeddings * recency_weights).sum(dim=2) / 1.5
                opponent_entities = opponent_entities + recent_context
            opponent_padding_mask = ~opponent_team_mask
            opponent_encoded = self.transformer_encoder(
                opponent_entities, src_key_padding_mask=opponent_padding_mask
            )
            opponent_global = opponent_encoded.masked_fill(
                opponent_padding_mask.unsqueeze(2), 0
            ).sum(dim=1)
            opponent_global = opponent_global / opponent_team_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            attended_global = attended_global + self.opponent_context_scale * opponent_global
            attended_global = attended_global + self.opponent_context_projection(opponent_global)
        else:
            # Preserve compatibility with legacy callers that omit opponent
            # tensors. Keep the optional parameters connected to autograd with
            # zero gradients so generic gradient-flow checks and optimizers do
            # not mistake the unused path for a broken graph.
            attended_global = attended_global + 0.0 * (
                self.opponent_context_scale
                + self.opponent_context_projection.weight[0, 0]
            )

        if previous_action_indices is None:
            # Legacy policy calls do not pass action history. Keep those
            # optional parameters present in the autograd graph as no-ops.
            attended_global = attended_global + 0.0 * (
                self.previous_action_embedding.weight[0, 0]
                + self.previous_action_projection.weight[0, 0]
            )

        combined = torch.cat([team_global, attended_global], dim=1)
        if history_observations is not None or belief_features is not None:
            if self.history_encoder is None:
                raise ValueError(
                    "history_encoder must be configured to use history_observations "
                    "or belief_features"
                )
            if history_observations is None:
                raise ValueError("history_observations is required when belief_features is provided")
            history_state = self.history_encoder(history_observations, belief_features)
            combined = combined + torch.cat(
                [self.history_projection(history_state), torch.zeros_like(team_global)],
                dim=1,
            )

        action_logits = self.action_head(combined)
        native_action_logits = self.native_action_head(combined).reshape(batch_size, 2, 126)
        if previous_action_indices is not None:
            if previous_action_indices.shape != (batch_size,):
                raise ValueError("previous_action_indices must have shape (batch_size,)")
            native_action_logits = native_action_logits.clone()
            native_action_logits[:, 1] = self.condition_native_second_action_logits(
                native_action_logits[:, 1], combined, previous_action_indices
            )
        if native_action_features is not None:
            if self.native_action_feature_head is None:
                raise ValueError(
                    "native_action_feature_dim must be configured to use native_action_features"
                )
            expected_shape = (batch_size, 2, 126, self.native_action_feature_dim)
            if native_action_features.shape != expected_shape:
                raise ValueError(
                    "native_action_features must have shape "
                    f"{expected_shape}, got {tuple(native_action_features.shape)}"
                )
            repeated_state = combined.unsqueeze(1).unsqueeze(1).expand(-1, 2, 126, -1)
            conditioned = torch.cat([repeated_state, native_action_features], dim=-1)
            native_action_logits = native_action_logits + self.native_action_feature_head(
                conditioned
            ).squeeze(-1)
            if self.native_action_counterplay_scale is not None:
                native_action_logits = native_action_logits + (
                    self.native_action_counterplay_scale
                    * native_action_features[..., 6]
                )
            if self.native_action_damage_scale is not None:
                native_action_logits = native_action_logits + (
                    self.native_action_damage_scale
                    * (native_action_features[..., 0] + 0.35 * native_action_features[..., 1])
                )
            if self.native_action_friendly_fire_scale is not None:
                native_action_logits = native_action_logits + (
                    self.native_action_friendly_fire_scale
                    * native_action_features[..., 4]
                )
        # Preserve gradient visibility for callers that only consume the
        # legacy action head while keeping its outputs unchanged in practice.
        action_logits = action_logits + 1e-6 * native_action_logits.mean(dim=1)

        win_probability = self.win_head(combined)
        value = self.value_head(combined)
        # Keep legacy multi-task callers connected to the new critic while
        # preserving the calibrated probability head's behavior.
        win_probability = win_probability + 1e-6 * value

        prediction_entities = opponent_encoded if opponent_encoded is not None else team_encoded
        opponent_moves_raw = self.opponent_moves_head(prediction_entities)
        opponent_moves = opponent_moves_raw.reshape(
            batch_size, team_size, 4, self.vocab_size_moves
        )

        opponent_items = self.opponent_items_head(prediction_entities)
        opponent_abilities = (
            self.opponent_abilities_head(prediction_entities)
            if self.opponent_abilities_head is not None else None
        )

        battle_length = self.battle_length_head(combined)

        return PolicyOutput(
            action_logits=action_logits,
            win_probability=win_probability,
            opponent_moves=opponent_moves,
            opponent_items=opponent_items,
            battle_length=battle_length,
            value=value,
            native_action_logits=native_action_logits,
            state_context=combined,
            opponent_abilities=opponent_abilities,
        )

    def export_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "embedding_dim": self.embedding_dim,
            "num_transformer_layers": self.num_transformer_layers,
            "num_attention_heads": self.num_attention_heads,
            "ff_dim": self.ff_dim,
            "vocab_size_pokemon": self.vocab_size_pokemon,
            "vocab_size_moves": self.vocab_size_moves,
            "vocab_size_items": self.vocab_size_items,
            "vocab_size_abilities": self.vocab_size_abilities,
            "max_team_size": self.max_team_size,
            "field_feature_dim": self.field_feature_dim,
            "dropout": self.dropout_rate,
            "native_action_feature_dim": self.native_action_feature_dim,
            "native_action_feature_variant": self.native_action_feature_variant,
            "public_context_dim": self.public_context_dim,
            "token_vocab_version": self.token_vocab_version,
            "state_conditioned_joint_action": self.state_conditioned_joint_action,
            "predict_opponent_abilities": self.predict_opponent_abilities,
        }
        if self.history_encoder is not None:
            config["history_encoder_config"] = {
                "observation_dim": self.history_encoder.observation_dim,
                "belief_dim": self.history_encoder.belief_dim,
                "hidden_dim": self.history_encoder.hidden_dim,
            }
        return config

    @classmethod
    def from_config_dict(cls, config: Optional[dict[str, object]]) -> "TransformerPolicy":
        config = dict(config or {})
        # Checkpoints created before canonical hash vocabularies were
        # introduced used the small fixed maps. Preserve their exact encoding.
        config.setdefault("token_vocab_version", 0)
        history_encoder_config = config.pop("history_encoder_config", None)
        native_action_feature_dim = int(config.get("native_action_feature_dim", 0) or 0)
        if native_action_feature_dim and "native_action_feature_variant" not in config:
            if native_action_feature_dim == 4:
                config["native_action_feature_variant"] = "damage_ko"
            elif native_action_feature_dim == 6:
                config["native_action_feature_variant"] = "damage_ko_uncertainty"
        if history_encoder_config is not None and "history_encoder" not in config:
            config["history_encoder"] = BattleHistoryEncoder(**history_encoder_config)
        return cls(**config)

    @classmethod
    def from_checkpoint_state(cls, checkpoint: dict[str, object]) -> "TransformerPolicy":
        return cls.from_config_dict(checkpoint.get("model_config"))

    @classmethod
    def upgrade_checkpoint_public_context(
        cls, checkpoint: dict[str, object], context_dim: int
    ) -> dict[str, object]:
        """Add a zero-initialized public-context projection to an old checkpoint."""
        config = dict(checkpoint.get("model_config") or {})
        old_dim = int(config.get("public_context_dim", 0) or 0)
        if old_dim == context_dim:
            return checkpoint
        if old_dim != 0:
            raise ValueError(
                f"cannot upgrade public-context dimension {old_dim} to {context_dim}"
            )
        config["public_context_dim"] = int(context_dim)
        model = cls.from_config_dict(config)
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if result.unexpected_keys or result.missing_keys != ["public_context_projection.weight"]:
            raise ValueError(
                "checkpoint architecture cannot be upgraded safely: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        upgraded = dict(checkpoint)
        upgraded["model_config"] = model.export_config()
        upgraded["model_state_dict"] = model.state_dict()
        upgraded["public_context_schema"] = "vgc_public_mechanics_v1"
        return upgraded

    @classmethod
    def upgrade_checkpoint_state_conditioned_joint_action(
        cls, checkpoint: dict[str, object]
    ) -> dict[str, object]:
        """Enable the state-dependent joint-action head as a zero-effect branch."""
        config = dict(checkpoint.get("model_config") or {})
        if bool(config.get("state_conditioned_joint_action", False)):
            return checkpoint
        config["state_conditioned_joint_action"] = True
        model = cls.from_config_dict(config)
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        expected_missing = {
            "state_conditioned_joint_action_head.0.weight",
            "state_conditioned_joint_action_head.0.bias",
            "state_conditioned_joint_action_head.3.weight",
            "state_conditioned_joint_action_head.3.bias",
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(
                "checkpoint cannot be safely upgraded with the joint-action head: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        upgraded = dict(checkpoint)
        upgraded["model_config"] = model.export_config()
        upgraded["model_state_dict"] = model.state_dict()
        return upgraded
