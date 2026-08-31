# Synthetic Battle Generator for Training Data

This module provides tools for generating synthetic Pokemon VGC battles
to create training data for supervised learning of battle policies.

## Key Components

1. **BattleGenerator**
   - Generates realistic Pokemon VGC battles using heuristic agents
   - Supports different agent strategies (random, max damage)
   - Records complete game trajectories with states and actions
   - Tracks battle statistics (win rates, average length)

2. **BattleRecord**
   - Dataclass containing a complete battle record
   - Includes player/opponent teams, winner, and trajectory
   - Trajectory is a list of TrajectoryStep objects

3. **TrajectoryStep**
   - Single step in a battle
   - Contains state, legal actions mask, action taken, reward, done flag

4. **BattleReplayer**
   - Iterator interface for traversing battle trajectories
   - Extracts training data efficiently
   - Supports bulk data extraction (get_states, get_actions, etc)

## Architecture

BattleGenerator uses the VGCDoublesEnv with two heuristic agents:
- Player Agent: Configured strategy (default: max_damage)
- Opponent Agent: Configured strategy (default: random)

The battle runs until:
- One team faints completely (terminal condition)
- Max steps reached (500 steps default)

Each step records:
- Encoded observation/state (from environment)
- Legal actions mask (which actions are valid)
- Action taken by player agent
- Immediate reward
- Terminal flag

## Usage

Basic example:

```python
from src.training import BattleGenerator, BattleReplayer
from src.utils.config import REGULATION_SV2024_1

# Create generator
generator = BattleGenerator(
    regulation_config=REGULATION_SV2024_1,
    seed=42,
    player_agent_type="max_damage",
    opponent_agent_type="random",
)

# Create teams from legal Pokemon
pokemon_list = list(REGULATION_SV2024_1.legal_pokemon)[:6]
team = [
    {
        "name": pokemon,
        "level": 50,
        "item": "choice-band",
        "ability": "static",
        "moves": ["earthquake", "protected"],
        "tera_type": "normal",
    }
    for pokemon in pokemon_list
]

# Generate battle
record = generator.generate_battle(player_team=team, opponent_team=team)

# Replay and extract training data
replayer = BattleReplayer(record)
for state, action, done, reward in replayer:
    # Use (state, action) for supervised learning
    # Use reward and done for value targets
    pass

# Get statistics
stats = generator.get_battle_statistics()
print(f"Player win rate: {stats['player_wins'] / stats['total_battles']}")
```

## Determinism and Seeds

The BattleGenerator supports seed-based reproducibility:

```python
gen1 = BattleGenerator(regulation_config=config, seed=42)
gen2 = BattleGenerator(regulation_config=config, seed=42)

record1 = gen1.generate_battle(team1, team2)
record2 = gen2.generate_battle(team1, team2)

# record1 and record2 will be identical
```

Different seeds produce different battles:

```python
gen1 = BattleGenerator(regulation_config=config, seed=42)
gen2 = BattleGenerator(regulation_config=config, seed=999)

record1 = gen1.generate_battle(team1, team2)  # Different outcome
record2 = gen2.generate_battle(team1, team2)
```

## Agent Types

Available agents:
- `"random"`: Uniformly randomly samples from legal actions
- `"max_damage"`: Heuristically selects high-damage actions
- `"switch_preserving"`: Prefers moves, switches only when needed
- `"team_preview"`: Makes informed switches based on team composition

## Constraints

1. All Pokemon in teams must be legal according to RegulationConfig
2. All items, moves, abilities, and tera types must be legal
3. Battles have max 500 steps to prevent infinite loops
4. Reward is normalized (> 0 for player win, <= 0 for opponent win)

## Testing

The implementation includes comprehensive tests:

```bash
pytest tests/training/test_synthetic_battles.py
```

Key test categories:
- Generator initialization and basic generation
- Trajectory format and completeness
- Determinism with seeds
- Team and action legality
- Battle replayer functionality
- Statistics tracking

## Future Enhancements

1. Multi-action support (simultaneous player/opponent actions)
2. Belief state tracking for learning incomplete information
3. Monte Carlo tree search for better battle simulation
4. Policy distillation from expert battles
5. Self-play training loop integration
