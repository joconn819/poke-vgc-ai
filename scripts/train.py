"""Train the offline supervised battle policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split

from src.models.policy import TransformerPolicy
from src.training.data import SyntheticBattleDataset, build_dataloader
from src.training.synthetic_battles import BattleGenerator
from src.training.train import SupervisedTrainer, SupervisedTrainingConfig
from src.utils.config import REGULATION_CHAMPIONS_MB
from src.utils.team_pool import TeamPool


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised VGC battle policy")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-output-path", type=Path, default=Path("data/generated/synthetic_dataset.pt"))
    parser.add_argument("--num-battles", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/supervised"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="poke-vgc-rl")
    args = parser.parse_args()

    dataset_path = _ensure_dataset(args)
    dataset = SyntheticBattleDataset.load(dataset_path)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = max(len(dataset) - val_size, 1)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformerPolicy()
    trainer = SupervisedTrainer(
        model=model,
        config=SupervisedTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        ),
    )
    wandb_run = _maybe_init_wandb(args, len(dataset))
    resume_path = _resolve_resume_path(args)
    metrics = trainer.train(train_loader, val_loader, resume_from=resume_path)
    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()
    print(metrics)


def _ensure_dataset(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return args.dataset_path

    if args.num_battles is None:
        raise ValueError("Pass --dataset-path or provide --num-battles to generate a dataset.")

    team_pool = TeamPool(
        teams_file="data/teams/champions_mb.json",
        regulation=REGULATION_CHAMPIONS_MB,
    )
    generator = BattleGenerator(
        regulation_config=REGULATION_CHAMPIONS_MB,
        seed=args.seed,
        player_agent_type="max_damage",
        opponent_agent_type="random",
    )
    dataset = SyntheticBattleDataset.from_generator(
        generator=generator,
        team_pool=team_pool,
        num_battles=args.num_battles,
        regulation_config=REGULATION_CHAMPIONS_MB,
    )
    dataset.save(args.dataset_output_path)
    return args.dataset_output_path


def _maybe_init_wandb(args: argparse.Namespace, dataset_size: int):
    if not args.wandb:
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dataset_size": dataset_size,
            "resume": args.resume or args.resume_from is not None,
        },
    )


def _resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume_from is not None:
        return args.resume_from
    if args.resume:
        candidate = args.checkpoint_dir / "latest.pt"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
