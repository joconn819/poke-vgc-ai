"""Train the policy on native Showdown trajectory records."""

from __future__ import annotations

import argparse
import gc
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
    gc_group = parser.add_mutually_exclusive_group()
    gc_group.add_argument(
        "--disable-gc-during-ppo-update",
        dest="disable_gc_during_ppo_update",
        action="store_true",
        help="Disable cyclic GC during each PPO update (default).",
    )
    gc_group.add_argument(
        "--enable-gc-during-ppo-update",
        dest="disable_gc_during_ppo_update",
        action="store_false",
        help="Keep cyclic GC enabled during PPO updates.",
    )
    parser.set_defaults(disable_gc_during_ppo_update=True)
    args = parser.parse_args()

    model = TransformerPolicy()
    trainer = NativePPOTrainer(model, args.checkpoint_dir)
    if args.init_from_checkpoint:
        state = torch.load(args.init_from_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
    records = load_native_records(args.trajectory_path)
    for update in range(args.updates):
        gc_was_enabled = gc.isenabled()
        if args.disable_gc_during_ppo_update and gc_was_enabled:
            gc.disable()
        try:
            metrics = trainer.update(records, device=args.device)
        finally:
            if args.disable_gc_during_ppo_update and gc_was_enabled:
                gc.enable()
        trainer.save(args.checkpoint_dir / f"update_{update + 1:06d}.pt")
        print({"update": update + 1, **metrics})


if __name__ == "__main__":
    main()
