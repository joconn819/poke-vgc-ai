# Feature Encoding System for Pokemon Battles

This module provides a canonical feature encoding system that converts Pokemon battle observations into PyTorch tensors optimized for Transformer-based neural networks.

## Architecture Overview

### Entity-Based Representation
The encoder uses an **entity-based architecture** where teams are represented as variable-length sequences of Pokemon entities:

```
Team Tensor: (batch_size=1, max_team_size=6, entity_dim=470)
├─ Pokemon[0] entity
├─ Pokemon[1] entity
├─ Pokemon[2] entity
├─ Pokemon[3] (padding)
├─ Pokemon[4] (padding)
└─ Pokemon[5] (padding)
```

### Pokemon Entity Encoding (470 dimensions)
Each Pokemon entity combines multiple feature groups:

1. **Species Embedding** (128 dims)
   - Learned embedding for Pokemon species

2. **State Features** (1 + 8 dims)
   - HP percentage (1 dim): [0, 1] normalized
   - Status embedding (8 dims): burn, freeze, paralysis, etc.

3. **Item & Ability Embeddings** (32 + 32 dims)
   - Item embedding (32 dims)
   - Ability embedding (32 dims)

4. **Move Embeddings** (256 dims)
   - 4 moves × 64-dim embeddings
   - Padded with zero vectors for missing moves

5. **Stat Features** (12 dims)
   - Normalized stat values (6 dims): [0, 1]
   - Stat boosts (6 dims): [-1, 1] for range [-6, +6]

6. **Tera Status** (1 dim)
   - Binary flag: 1 if Tera used, 0 otherwise

### Complete Feature Tensors

#### Observable Mechanics
- **Team Tensor**: (1, 6, 470)
  - Full team representation with padding
  
- **Active Pokemon Tensor**: (1, 2, 470)
  - Currently active Pokemon in doubles format

#### Field State
- **Field Tensor**: (1, 64)
  - Weather (8-dim embedding)
  - Terrain (8-dim embedding)
  - Binary conditions: trick room, reflect, light screen, tailwind

#### Belief State (Hidden Information)
- **Opponent Belief Tensor**: (1, 6, 128)
  - Uncertainty ranges for each opponent Pokemon
  - HP bounds (min, max, expected)
  - Speed prediction (crucial for turn order)
  - Stat expectations
  - Move/item/ability set sizes

#### Regulation & Masks
- **Regulation Embedding**: (1, 32)
  - Regulation-specific embedding token

- **Move Masks**: (1, 2, 8)
  - Legal move indices per active Pokemon

- **Switch Masks**: (1, 4)
  - Available switch targets

- **Target Masks**: (1, 2, 2)
  - Valid targets per active Pokemon in doubles

## Key Features

### Deterministic Encoding
- Same input always produces same output
- Suitable for distributed training and reproducibility

### Padding & Masking
- Variable-length teams automatically padded to max_team_size
- Proper attention masks for Transformers
- Efficient batching of different team sizes

### Belief Incorporation
- Uncertainty ranges from belief state encoded directly
- Speed prediction enables turn-order reasoning
- Possible moveset cardinality captured

### Regulation-Aware
- Unique embedding per regulation
- Enables policy sharing across regulations
- Level information (50 for Champions MB) can be included

## Usage Example

```python
from src.features import FeatureEncoder
from src.utils.config import RegulationConfig, Generation

# Create encoder
regulation = RegulationConfig(
    regulation_id="sv2024-1",
    generation=Generation.GEN_9,
    legal_pokemon={"pikachu", "charizard", ...},
    legal_items={"choice-band", ...},
    legal_moves={"earthquake", ...},
    legal_abilities={"static", ...},
)
encoder = FeatureEncoder(regulation)

# Encode observation
observation = {
    "team": [...],
    "active_pokemon": [...],
    "opponent_team": [...],
    "field": {...},
    "level": 50,
}
belief = {"opponent_species": PokemonBeliefState(...), ...}

encoded = encoder.encode(observation, belief)

# Access tensors
team = encoded.team_tensor                    # (1, 6, 470)
field = encoded.field_tensor                  # (1, 64)
opponent_belief = encoded.opponent_belief_tensor  # (1, 6, 128)
move_masks = encoded.move_masks              # (1, 2, 8)
switch_masks = encoded.switch_masks          # (1, 4)
target_masks = encoded.target_masks          # (1, 2, 2)
```

## Index Mappings

Canonical index mappings are provided for:
- **Pokemon Species**: 25+ legal Pokemon in Gen 9 VGC
- **Moves**: 14+ common competitive moves
- **Items**: 8 common competitive items
- **Abilities**: 8 competitive abilities
- **Status Conditions**: burn, freeze, paralysis, poison, sleep
- **Weather**: hail, rain, sand, sun
- **Terrain**: electric, grassy, misty, psychic
- **Tera Types**: all 18 types

## Testing

Comprehensive test suite with 38 tests covering:
- ✓ Tensor shape validation
- ✓ All Pokemon types handled
- ✓ Padding with variable team sizes
- ✓ Action mask encoding
- ✓ Belief state incorporation
- ✓ Field state encoding
- ✓ Regulation encoding
- ✓ Deterministic encoding
- ✓ Value normalization
- ✓ Finite value checking

Run tests:
```bash
pytest tests/features/test_encoding.py -v
```

## Design Decisions

1. **Entity-Based Over Flattened**: Enables Transformer attention over Pokemon entities
2. **Separate Belief Tensor**: Keeps uncertainty features isolated for interpretability
3. **8-Dim Status Embedding**: Allows model to learn status importance
4. **Normalized Values**: HP, boosts in [0,1]/[-1,1] for stable training
5. **Padding with Zeros**: Zero-padding allows attention masking
6. **Fixed Shapes**: Enables batching and static memory allocation

## Future Enhancements

- Variable-length move sequences (1-4 moves)
- Conditional probability encoding for belief state
- Weather/terrain effect modifiers
- Held item effect encoding
- Move power/accuracy normalization
