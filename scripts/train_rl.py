"""Run self-play PPO training for the VGC battle policy.

Initializes from a supervised checkpoint (recommended) and iterates
self-play PPO updates, saving resumable checkpoints along the way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.policy import TransformerPolicy
from src.training.self_play_train import SelfPlayConfig, SelfPlayTrainer
from src.utils.config import REGULATION_CHAMPIONS_MB
from src.utils.team_pool import TeamPool


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play PPO training for VGC battle policy")
    parser.add_argument("--init-from-checkpoint", type=Path, help="Supervised checkpoint to warm-start from")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/self_play"))
    parser.add_argument("--team-pool-path", type=str, default="data/teams/champions_mb.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--max-steps-per-episode", type=int, default=30)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-sample-probability", type=float, default=0.3)
    parser.add_argument("--add-self-to-pool-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt in --checkpoint-dir")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="poke-vgc-rl")
    args = parser.parse_args()

    team_pool = TeamPool(teams_file=args.team_pool_path, regulation=REGULATION_CHAMPIONS_MB)

    model = TransformerPolicy()
    config = SelfPlayConfig(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        episodes_per_update=args.episodes_per_update,
        max_steps_per_episode=args.max_steps_per_episode,
        epochs_per_update=args.epochs_per_update,
        learning_rate=args.learning_rate,
        checkpoint_sample_probability=args.checkpoint_sample_probability,
        add_self_to_pool_every=args.add_self_to_pool_every,
        init_from_checkpoint=str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
        team_pool_path=args.team_pool_path,
        seed=args.seed,
    )
    trainer = SelfPlayTrainer(model=model, config=config, team_pool=team_pool)

    start_iteration = 0
    latest_path = args.checkpoint_dir / "latest.pt"
    if args.resume and latest_path.exists():
        start_iteration = trainer.resume_from_checkpoint(latest_path) + 1
        print(f"Resumed self-play from iteration {start_iteration}")

    wandb_run = _maybe_init_wandb(args)

    for iteration in range(start_iteration, args.iterations):
        metrics = trainer.train_iteration(iteration=iteration)
        print(f"[iter {iteration}] {metrics}")
        if wandb_run is not None:
            wandb_run.log(metrics, step=iteration)

    if wandb_run is not None:
        wandb_run.finish()


def _maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        config={
            "iterations": args.iterations,
            "episodes_per_update": args.episodes_per_update,
            "max_steps_per_episode": args.max_steps_per_episode,
            "learning_rate": args.learning_rate,
        },
    )


if __name__ == "__main__":
    main()
