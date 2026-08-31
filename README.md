# Pokemon VGC AI

This repository is an experiment in using AI to carry out a complex software
engineering and research project end to end. The system is being developed
with AI assistance as a learning experience for me: using an AI-driven
workflow to design, implement, test, run, and document a substantial
reinforcement learning project.

The project is also intentionally outside my usual area of expertise. My
typical interests are in **Numerics**, especially **Optimization** and
**Finite Elements**, rather than machine learning. Building a regulation-
agnostic Pokemon VGC agent is therefore both a technical project and an
experiment in learning a new domain through AI-assisted development.

The target is a state-of-the-art AI agent for Pokemon Video Game Championships
(VGC) using reinforcement learning, belief modeling, and Pokemon Showdown.

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
PYTHONPATH=. venv/bin/python scripts/generate_data.py --num-battles 100
PYTHONPATH=. venv/bin/python scripts/train.py --dataset-path data/generated/synthetic_dataset.pt
PYTHONPATH=. venv/bin/python scripts/eval.py --dataset-path data/generated/synthetic_dataset.pt --checkpoint-path data/checkpoints/supervised/best.pt
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

See [plan.md](/home/justin/.copilot/session-state/c842c69c-ed6f-439b-b952-3974dfb6ccf4/plan.md) for the full multi-phase development plan.

## Supervised training workflow

Phase 4 now includes:

- [scripts/generate_data.py](/home/justin/p/poke_learning_2/scripts/generate_data.py) for deterministic synthetic dataset generation
- [src/training/data.py](/home/justin/p/poke_learning_2/src/training/data.py) for step-level supervised dataset creation
- [src/training/train.py](/home/justin/p/poke_learning_2/src/training/train.py) for offline policy training and checkpointing
- [scripts/eval.py](/home/justin/p/poke_learning_2/scripts/eval.py) for checkpoint evaluation

The current baseline trains on flattened battle trajectories produced by heuristic self-play, then optimizes the multi-head Transformer policy against:

- action prediction,
- win prediction,
- opponent move prediction,
- opponent item prediction,
- battle length prediction.

## Self-play RL workflow (Phase 5)

- [src/training/ppo.py](/home/justin/p/poke_learning_2/src/training/ppo.py) implements GAE advantage estimation and clipped PPO losses.
- [src/training/self_play.py](/home/justin/p/poke_learning_2/src/training/self_play.py) provides `PolicyAgent` (wraps `TransformerPolicy` for rollout action selection with log-probs/value) and `OpponentPool` (mixes heuristic baselines with past checkpoints).
- [src/training/self_play_train.py](/home/justin/p/poke_learning_2/src/training/self_play_train.py) implements rollout generation and the `SelfPlayTrainer` PPO update loop.
- [scripts/train_rl.py](/home/justin/p/poke_learning_2/scripts/train_rl.py) is the CLI entry point. Warm-start from a supervised checkpoint and iterate:

```bash
PYTHONPATH=. venv/bin/python scripts/train_rl.py \
  --init-from-checkpoint data/checkpoints/champions-mb-supervised-run1/best.pt \
  --checkpoint-dir data/checkpoints/champions-mb-self-play-run1 \
  --iterations 200 --episodes-per-update 32 --max-steps-per-episode 30 \
  --device cuda --resume
```

Rerun the same command with `--resume` and a higher `--iterations` to continue an in-progress run; checkpoints (`latest.pt`, `iter_NNNNNN.pt`) and `history.jsonl` are written to `--checkpoint-dir`.

**Known limitation**: `VGCDoublesEnv` still uses placeholder battle dynamics, so the self-play terminal reward is not yet grounded in real battle outcomes. Self-play currently validates the training mechanics (rollouts, PPO update, checkpoint resumability, opponent pool mixing) end-to-end; meaningful policy improvement requires richer environment dynamics first.

## Real Showdown battles

Use [ShowdownDoublesEnv](/home/justin/p/poke_learning_2/src/env/showdown_env.py) for the native `poke-env` two-agent environment:

```python
from src.env import ShowdownDoublesEnv

env = ShowdownDoublesEnv(
    battle_format="gen8doublesou",
    team="<packed Showdown team>",
    fake=False,
)
observations, info = env.reset()
# Submit one native poke-env action array per env.agents entry.
observations, rewards, terminated, truncated, info = env.step(actions)
```

This connects to the local Showdown websocket by default. Start a local Showdown server or pass a `poke_env` `ServerConfiguration` for a remote server. `VGCDoublesEnv` remains the legacy flattened Gym adapter used by the current offline/self-play scaffolding until its policy/action encoder is migrated to native poke-env actions.

## Benchmarking

[src/evaluation/benchmark.py](/home/justin/p/poke_learning_2/src/evaluation/benchmark.py) provides reproducible pairwise matchups, ordered benchmark matrices, Wilson 95% confidence intervals, and JSON serialization. Use `benchmark_matchup` or `benchmark_matrix` with fresh agent factories so each matchup is isolated and repeatable.
