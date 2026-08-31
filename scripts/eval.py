"""Evaluate a trained supervised battle policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.policy import TransformerPolicy
from src.training.data import SyntheticBattleDataset, build_dataloader
from src.training.train import SupervisedTrainer, SupervisedTrainingConfig, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate supervised VGC battle policy")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = SyntheticBattleDataset.load(args.dataset_path)
    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformerPolicy()
    load_checkpoint(args.checkpoint_path, model)

    trainer = SupervisedTrainer(
        model=model,
        config=SupervisedTrainingConfig(
            epochs=1,
            checkpoint_dir=args.checkpoint_path.parent,
            device=args.device,
        ),
    )
    metrics = trainer._run_epoch(loader, training=False)
    print(metrics)


if __name__ == "__main__":
    main()
