# Pokemon VGC AI

State-of-the-art AI agent for Pokemon Video Game Championships (VGC) using reinforcement learning, belief modeling, and Pokemon Showdown.

## Vision

- **Regulation-agnostic** battle understanding
- **Meta-aware** adaptation
- **Belief modeling** for hidden information (opponent movesets, items, spreads)
- **open-teamsheets integration** for reducing uncertainty
- Transfer learning across regulation changes

## Quick start

```bash
poetry install
pytest tests/
```

## Repository structure

```
src/
  env/           # poke-env wrappers and Gym interface
  showdown/      # Local Showdown process management
  belief/        # Hidden-information modeling and inference
  features/      # State representation and encoding
  models/        # Neural network architectures
  training/      # RL and supervised learning loops
  evaluation/    # Benchmarking and win-rate tracking
  baselines/     # Heuristic baseline agents
  replays/       # Replay parsing and data loading
  utils/         # open-teamsheets integration, config, etc.
configs/
  regulations/   # Format rules, dex restrictions, item pools
  experiments/   # Experiment configurations
scripts/         # Data processing and orchestration
tests/           # Test suite (pytest)
data/            # (gitignored) Large datasets and checkpoints
```

## Development

All components follow test-driven development (TDD). Run tests with:

```bash
pytest tests/ -v
pytest tests/belief --cov=src/belief  # Coverage by module
```

## Research phases

See [plan.md](https://github.com/your-org/poke-vgc-rl/blob/main/RESEARCH.md) for the full multi-phase development plan.
