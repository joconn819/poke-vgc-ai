"""Train the policy on native Showdown trajectory records."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.policy import TransformerPolicy
from src.training.native_ppo_train import NativePPOTrainer, load_native_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/native-rl"))
    parser.add_argument("--init-from-checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = TransformerPolicy()
    trainer = NativePPOTrainer(model, args.checkpoint_dir)
    if args.init_from_checkpoint:
        state = torch.load(args.init_from_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
    records = load_native_records(args.trajectory_path)
    for update in range(args.updates):
        metrics = trainer.update(records, device=args.device)
        trainer.save(args.checkpoint_dir / f"update_{update + 1:06d}.pt")
        print({"update": update + 1, **metrics})


if __name__ == "__main__":
    main()
