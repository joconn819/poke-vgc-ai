"""
# Transformer-based Policy Network for Pokemon VGC

Complete multi-task learning implementation for Pokemon battle decision-making.

## Architecture Overview

The policy network uses a Transformer encoder to learn battle state representations and produces:

### Main Task
- **Action Selection** (126 possible action pairs): move1, move2, target1, target2
  - Outputs: logits for all valid action combinations
  - Loss: cross-entropy with action masking

### Auxiliary Tasks (Multi-Task Learning)
- **Win Probability**: Binary classification of current game state win probability
- **Opponent Movesets**: Predict 4 moves for each visible opponent Pokemon
- **Opponent Items**: Predict held item for each visible opponent Pokemon  
- **Battle Length**: Predict number of turns remaining (50 turn buckets)

## Model Sizing

Default configuration (meets 1-5M parameter target):
- Embedding dimension: 128
- Transformer layers: 4
- Attention heads: 8
- Feedforward dimension: 512
- Total parameters: ~2.5M (scales from 800K to 5M+ with config)

## Module Structure

### src/models/policy.py
- `TransformerPolicy`: Main model class
  - Embeddings for Pokemon, moves, items, abilities, status
  - Positional embeddings for team order
  - Transformer encoder over team entities
  - Cross-attention between field state and team
  - Separate heads for main and auxiliary tasks
  
- `PolicyOutput`: Dataclass containing all model outputs

### src/models/losses.py
- `PolicyLoss`: Combined loss function
  - Weighted multi-task loss (main task vs auxiliary)
  - Action masking for illegal moves
  - Configurable task weights (default: 1.0 main, 0.1x auxiliary)

### src/models/inference.py
- `PolicyInference`: Inference-time utilities
  - Action masking and distribution manipulation
  - Sampling (stochastic) vs argmax (deterministic) action selection
  - Action pair decoding (flattened index → move pairs + targets)
  - Auxiliary prediction extraction

## Usage Example

```python
from src.models.policy import TransformerPolicy
from src.models.losses import PolicyLoss
from src.models.inference import PolicyInference

# Initialize model
model = TransformerPolicy(
    embedding_dim=128,
    num_transformer_layers=4,
    num_attention_heads=8,
    ff_dim=512,
)

# Forward pass
batch_data = {
    "pokemon_ids": torch.randint(0, 1025, (batch_size, 6)),
    "move_ids": torch.randint(0, 656, (batch_size, 6, 4)),
    "item_ids": torch.randint(0, 376, (batch_size, 6)),
    "ability_ids": torch.randint(0, 267, (batch_size, 6)),
    "team_mask": torch.ones(batch_size, 6, dtype=torch.bool),
    "field_features": torch.randn(batch_size, 32),
    "hp_fractions": torch.rand(batch_size, 6),
    "status": torch.randint(0, 8, (batch_size, 6)),
}

output = model(**batch_data)
# output.action_logits: (batch_size, 126)
# output.win_probability: (batch_size, 1)
# output.opponent_moves: (batch_size, team_size, 4, 656)
# output.opponent_items: (batch_size, team_size, 376)
# output.battle_length: (batch_size, 50)

# Compute loss with multi-task learning
loss_fn = PolicyLoss(
    action_weight=1.0,
    win_weight=0.1,
    opponent_moves_weight=0.1,
    opponent_items_weight=0.1,
    battle_length_weight=0.1,
)

predictions = {
    "action_logits": output.action_logits,
    "win_probability": output.win_probability,
    "opponent_moves": output.opponent_moves,
    "opponent_items": output.opponent_items,
    "battle_length": output.battle_length,
}

targets = {
    "action_indices": torch.randint(0, 126, (batch_size,)),
    "win_target": torch.randint(0, 2, (batch_size, 1)).float(),
    "opponent_move_indices": torch.randint(0, 656, (batch_size, 6, 4)),
    "opponent_item_indices": torch.randint(0, 376, (batch_size, 6)),
    "battle_length_targets": torch.randint(0, 50, (batch_size,)),
}

loss = loss_fn(predictions, targets, action_mask=action_mask)

# Inference with action masking
inference = PolicyInference()
action_mask = torch.ones(batch_size, 126, dtype=torch.bool)
action_mask[:, 100:126] = False  # Mask invalid actions

# Deterministic action selection
actions, probs = inference.batch_inference(
    output.action_logits, action_mask, sample=False
)

# Stochastic action selection
actions, probs = inference.batch_inference(
    output.action_logits, action_mask, sample=True
)

# Decode action pairs
for action_idx in actions:
    move1, move2, target1, target2 = inference.decode_action_pair(action_idx.item())

# Extract auxiliary predictions
opp_moves = inference.predict_opponent_moves(output.opponent_moves)
opp_items = inference.predict_opponent_items(output.opponent_items)
battle_len = inference.predict_battle_length(output.battle_length)
win_prob = inference.predict_win_probability(output.win_probability)
```

## Input Specification

All inputs should be properly padded to max_team_size=6:

- **pokemon_ids**: Pokemon Pokedex IDs (1-1025)
- **move_ids**: Move IDs for each Pokemon's 4 moves (0-656, 0 for empty)
- **item_ids**: Item IDs held by each Pokemon (0-376, 0 for none)
- **ability_ids**: Ability IDs (0-267)
- **team_mask**: Boolean mask indicating valid team members
- **field_features**: Game state features (weather, terrain, etc.) - 32 dims
- **hp_fractions**: Current HP / Max HP for each Pokemon (0-1)
- **status**: Status condition (0=none, 1-7 for various conditions)

## Key Features

### Action Masking
Invalid actions (illegal moves, fainted Pokemon, etc.) are masked via:
1. Setting logits to -∞ before softmax
2. Using safe log_softmax that handles masked values
3. Ensures sampled/argmax actions always respect mask

### Multi-Task Learning
- Auxiliary tasks improve generalization
- Enables transfer learning (e.g., moveset prediction helps move selection)
- Configurable weights allow tuning task importance
- Each auxiliary task uses appropriate loss (CE for classification, BCE for win)

### Efficiency
- Batch processing: independent batch items use no cross-batch attention
- Padding-aware: team_mask prevents attention to padding tokens
- Optional action masking: only applies if needed
- Temperature scaling: control stochasticity in sampling

## Testing

Comprehensive test suite (48 tests) covers:
- Model architecture and forward passes
- Output shape correctness
- Gradient flow through all heads
- Determinism with fixed seeds
- Variable team sizes with padding
- Batch independence
- Loss computation and masking
- Inference sampling and argmax
- Edge cases (empty batches, all masked actions, etc.)

Run tests:
```bash
pytest tests/models/ -v
```

## Architecture Diagram

```
Input Features
    ↓
Embedding Layer (Pokemon, Moves, Items, Abilities, Status)
    ↓
Position Encoding (Team order)
    ↓
Team Entity Projection (combine embeddings)
    ↓
Transformer Encoder (4 layers, 8 heads, 512 FFN)
    ↓
Dual Attention Module (team × field cross-attention)
    ↓
Global Pooling
    ↓
┌─────────────────────────────────────────────┐
│                                             │
↓                ↓              ↓          ↓  ↓
Action Head  Win Head  Opponent  Opponent Battle
             Moves Head Items Head Length Head
│                                             │
└─────────────────────────────────────────────┘
      ↓              ↓              ↓          ↓        ↓
Action Logits   Win Prob    Opponent    Opponent   Battle
(batch, 126)  (batch, 1)    Moves      Items      Length
                         (b,6,4,656) (b,6,376)  (b,50)
```

## Future Improvements

- Sequence modeling: handle move/ability history across turns
- Recurrent components: LSTM/GRU for temporal dependencies
- Hierarchical action space: decompose into move selection, then targeting
- Curriculum learning: start with high-level actions, then fine-grained targets
- Data augmentation: synthetic battle state generation
- Uncertainty estimation: ensemble or MC-Dropout for confidence scores
"""
