"""Team-preview action encoding and policy.

The policy treats a VGC team-preview decision as one atomic action: choose two
lead Pokemon, two back Pokemon, and leave two Pokemon out. Order within each
group is irrelevant, so there are ``6! / (2! * 2! * 2!) = 90`` actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence
import hashlib

import torch
from torch import nn
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

TEAM_SIZE = 6
GROUP_SIZE = 2
ACTION_COUNT = 90
TYPE_NAMES = (
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
    "Steel", "Fairy",
)
CATEGORICAL_FEATURE_DIM = 16
TEAM_FEATURE_DIM = CATEGORICAL_FEATURE_DIM + 6 + len(TYPE_NAMES) * 2 + 7
_GEN9_DATA = GenData.from_gen(9)


@dataclass(frozen=True)
class TeamSelectionAction:
    """A legal team selection, grouped as leads, back, and excluded Pokemon."""

    leads: tuple[int, ...]
    back: tuple[int, ...]
    out: tuple[int, ...]


def _all_actions() -> tuple[TeamSelectionAction, ...]:
    actions = []
    for leads in combinations(range(TEAM_SIZE), GROUP_SIZE):
        remaining = [index for index in range(TEAM_SIZE) if index not in leads]
        for back in combinations(remaining, GROUP_SIZE):
            out = tuple(index for index in remaining if index not in back)
            actions.append(TeamSelectionAction(tuple(leads), tuple(back), out))
    return tuple(actions)


ACTIONS = _all_actions()
_ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def team_features(
    team: Sequence[dict[str, Any]],
    feature_dim: int = TEAM_FEATURE_DIM,
    *,
    revealed_only: bool = False,
) -> torch.Tensor:
    """Encode categorical identity plus Gen 9 battle-relevant team features.

    The categorical prefix remains stable for 16-dimensional legacy models.
    New selectors also receive base stats, species typing, move-type coverage,
    and compact move power/category/priority/accuracy summaries.
    """
    if len(team) != TEAM_SIZE or feature_dim <= 0:
        raise ValueError("team must contain six Pokemon and feature_dim must be positive")
    rows = []
    for pokemon in team:
        species_name = pokemon.get("species") or pokemon.get("name", "")
        fields = [species_name]
        if not revealed_only:
            fields += [pokemon.get("item", ""), pokemon.get("ability", "")]
            fields += list(pokemon.get("moves", []))
        categorical = []
        for index in range(CATEGORICAL_FEATURE_DIM):
            digest = hashlib.blake2b(
                f"{fields[index % len(fields)]}|{index}".encode(), digest_size=4
            ).digest()
            categorical.append((int.from_bytes(digest, "little") / 2**32) * 2.0 - 1.0)
        if feature_dim <= CATEGORICAL_FEATURE_DIM:
            rows.append(categorical[:feature_dim])
            continue

        species = _GEN9_DATA.pokedex.get(to_id_str(species_name), {})
        base_stats = species.get("baseStats", {})
        stats = [float(base_stats.get(key, 0)) / 200.0 for key in
                 ("hp", "atk", "def", "spa", "spd", "spe")]
        species_types = set(species.get("types", ()))
        type_features = [float(type_name in species_types) for type_name in TYPE_NAMES]
        moves = [] if revealed_only else [
            _GEN9_DATA.moves.get(to_id_str(name), {}) for name in pokemon.get("moves", ())
        ]
        move_types = {move.get("type") for move in moves if move.get("type")}
        move_type_features = [float(type_name in move_types) for type_name in TYPE_NAMES]
        powers = [max(float(move.get("basePower", 0) or 0), 0.0) for move in moves]
        categories = [move.get("category", "Status") for move in moves]
        numeric_accuracy = [float(move["accuracy"]) for move in moves
                            if isinstance(move.get("accuracy"), (int, float))]
        move_summary = [
            (sum(powers) / max(len(powers), 1)) / 200.0,
            (max(powers, default=0.0)) / 200.0,
            categories.count("Physical") / 4.0,
            categories.count("Special") / 4.0,
            categories.count("Status") / 4.0,
            max((float(move.get("priority", 0) or 0) for move in moves), default=0.0) / 5.0,
            (sum(numeric_accuracy) / len(numeric_accuracy) / 100.0
             if numeric_accuracy else 0.0),
        ]
        values = categorical + stats + type_features + move_type_features + move_summary
        if feature_dim > len(values):
            values.extend([0.0] * (feature_dim - len(values)))
        rows.append(values[:feature_dim])
    return torch.tensor(rows, dtype=torch.float32)


def random_action(rng: Any) -> int:
    """Return a uniformly random atomic preview action."""
    return int(rng.integers(ACTION_COUNT)) if hasattr(rng, "integers") else int(rng.randrange(ACTION_COUNT))


def preview_command(action: TeamSelectionAction | int) -> str:
    """Build Showdown's atomic ``/team`` command for an action."""
    selected = decode_action(action) if isinstance(action, int) else action
    return "/team " + "".join(str(i + 1) for i in selected.leads + selected.back)


def encode_action(
    team: TeamSelectionAction | Iterable[int],
    leads: Iterable[int] | None = None,
) -> int:
    """Encode selected four Pokemon and leads into ``[0, ACTION_COUNT)``."""
    if isinstance(team, TeamSelectionAction):
        if leads is not None:
            raise ValueError("leads must be omitted when team is an action")
        action = team
    else:
        if leads is None:
            raise ValueError("leads are required when team is a sequence")
        selected = tuple(sorted(team))
        lead_tuple = tuple(sorted(leads))
        if len(selected) != 4 or len(set(selected)) != 4:
            raise ValueError("team must contain four distinct Pokemon indices")
        action = TeamSelectionAction(
            lead_tuple,
            tuple(index for index in selected if index not in lead_tuple),
            tuple(index for index in range(TEAM_SIZE) if index not in selected),
        )

    groups = (action.leads, action.back, action.out)
    if any(len(group) != GROUP_SIZE or len(set(group)) != GROUP_SIZE for group in groups):
        raise ValueError("each selection group must contain two distinct Pokemon")
    if any(index < 0 or index >= TEAM_SIZE for group in groups for index in group):
        raise ValueError("team indices must be between 0 and 5")
    if set().union(*map(set, groups)) != set(range(TEAM_SIZE)):
        raise ValueError("selection groups must partition all six Pokemon")
    try:
        return _ACTION_TO_INDEX[action]
    except KeyError as error:
        raise ValueError("invalid team selection") from error


def decode_action(index: int) -> TeamSelectionAction:
    """Decode an integer action into leads, back Pokemon, and excluded Pokemon."""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index must be an integer in [0, {ACTION_COUNT})")
    return ACTIONS[index]


def action_mask(available: torch.Tensor | Sequence[bool]) -> torch.Tensor:
    """Return legal-action masks for available Pokemon.

    ``available`` may be shaped ``(6,)`` or ``(batch, 6)``.  An action is legal
    only when all four selected Pokemon are available.
    """
    available_tensor = torch.as_tensor(available, dtype=torch.bool)
    if available_tensor.shape[-1] != TEAM_SIZE:
        raise ValueError("availability must have a final dimension of six")
    flat = available_tensor.reshape(-1, TEAM_SIZE)
    selected = torch.tensor(
        [action.leads + action.back for action in ACTIONS], device=flat.device
    )
    mask = flat[:, selected].all(dim=2)
    return mask.reshape(*available_tensor.shape[:-1], ACTION_COUNT)


class TeamSelectionPolicy(nn.Module):
    """Small masked policy network for team-preview decisions."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 128, opponent_conditioned: bool = False,
        action_conditioned: bool = False,
    ):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.opponent_conditioned = opponent_conditioned
        self.action_conditioned = action_conditioned
        if action_conditioned:
            # Embed every Pokemon with shared weights, then score each legal
            # lead/back/out assignment from role-group summaries. This makes
            # scores equivariant to a permutation of the six preview slots.
            self.pokemon_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            group_count = 8 + int(opponent_conditioned)
            self.action_scorer = nn.Sequential(
                nn.Linear(group_count * hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.register_buffer(
                "action_groups",
                torch.tensor(
                    [[list(action.leads), list(action.back), list(action.out)]
                     for action in ACTIONS], dtype=torch.long,
                ),
            )
        else:
            # Preserve compatibility with existing flattened-input checkpoints.
            self.network = nn.Sequential(
                nn.Linear(TEAM_SIZE * feature_dim * (2 if opponent_conditioned else 1), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ACTION_COUNT),
            )

    def forward(
        self,
        team_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        opponent_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action scores, with illegal actions set to ``-inf``.

        Team tensors have shape ``(batch, 6, feature_dim)``. New outcome-trained
        models require ``opponent_features`` and predict signed action values.
        Legacy models ignore that keyword, retaining their original logits and
        state-dict layout. ``mask`` is an action mask ``(batch, 90)`` or an
        availability mask ``(batch, 6)``.
        """
        if team_features.ndim != 3 or team_features.shape[1:] != (TEAM_SIZE, self.feature_dim):
            raise ValueError(
                f"team_features must have shape (batch, {TEAM_SIZE}, {self.feature_dim})"
            )
        if self.action_conditioned:
            own = self.pokemon_encoder(team_features)
            groups = self.action_groups
            lead = own[:, groups[:, 0], :]
            back = own[:, groups[:, 1], :]
            out = own[:, groups[:, 2], :]
            parts = [
                lead.sum(dim=2), lead[:, :, 0] * lead[:, :, 1],
                back.sum(dim=2), back[:, :, 0] * back[:, :, 1],
                out.sum(dim=2), own.sum(dim=1)[:, None, :].expand(-1, ACTION_COUNT, -1),
            ]
            # Third pair feature supplies a group-sensitive term so the scorer
            # can distinguish complementary pairs with the same pooled sum.
            parts.insert(2, (lead[:, :, 0] - lead[:, :, 1]).abs())
            parts.insert(5, (back[:, :, 0] - back[:, :, 1]).abs())
            if self.opponent_conditioned:
                if opponent_features is None or opponent_features.shape != team_features.shape:
                    raise ValueError("opponent_features must match team_features shape")
                opponent = self.pokemon_encoder(opponent_features).mean(dim=1)
                parts.append(opponent[:, None, :].expand(-1, ACTION_COUNT, -1))
            logits = self.action_scorer(torch.cat(parts, dim=-1)).squeeze(-1)
        else:
            features = team_features.reshape(team_features.shape[0], -1)
            if self.opponent_conditioned:
                if opponent_features is None or opponent_features.shape != team_features.shape:
                    raise ValueError("opponent_features must match team_features shape")
                features = torch.cat(
                    (features, opponent_features.reshape(opponent_features.shape[0], -1)), dim=1
                )
            logits = self.network(features)
        if mask is None:
            return logits
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if mask_tensor.shape[-1] == TEAM_SIZE:
            mask_tensor = action_mask(mask_tensor)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] == 1 and logits.shape[0] != 1:
            mask_tensor = mask_tensor.expand(logits.shape[0], -1)
        if mask_tensor.shape != logits.shape:
            raise ValueError(f"mask must have shape {tuple(logits.shape)} or (batch, 6)")
        return logits.masked_fill(~mask_tensor, -torch.inf)

    def sample(
        self,
        team_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        opponent_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return ``(indices, log probabilities)``."""
        logits = self(team_features, mask, opponent_features=opponent_features)
        distribution = torch.distributions.Categorical(logits=logits)
        indices = distribution.sample()
        return indices, distribution.log_prob(indices)
