"""Supervised training loop for offline policy learning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.losses import PolicyLoss
from src.models.policy import PolicyOutput, TransformerPolicy
from src.training.metrics import summarize_metrics


@dataclass
class SupervisedTrainingConfig:
    """Configuration for offline supervised training."""

    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0
    checkpoint_dir: str | Path = "data/checkpoints/supervised"
    device: str = "cpu"


class SupervisedTrainer:
    """Trains a policy network on synthetic supervised examples."""

    def __init__(
        self,
        model: TransformerPolicy,
        config: SupervisedTrainingConfig,
        criterion: Optional[PolicyLoss] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.criterion = criterion or PolicyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str | Path] = None,
    ) -> Dict[str, float]:
        best_metrics: Dict[str, float] = {}
        best_val_loss = float("inf")
        start_epoch = 1

        if resume_from is not None:
            resume_state = self.resume_from_checkpoint(resume_from)
            start_epoch = resume_state["epoch"] + 1
            best_val_loss = float(resume_state["best_val_loss"])
            best_metrics = dict(resume_state["best_metrics"])

        if start_epoch > self.config.epochs:
            return best_metrics

        for epoch in range(start_epoch, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            metrics = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_action_accuracy": train_metrics["action_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_action_accuracy": val_metrics["action_accuracy"],
                "val_win_brier_score": val_metrics["win_brier_score"],
                "val_opponent_moves_accuracy": val_metrics["opponent_moves_accuracy"],
                "val_opponent_items_accuracy": val_metrics["opponent_items_accuracy"],
                "val_battle_length_mae": val_metrics["battle_length_mae"],
            }

            self._append_history(metrics)

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_metrics = metrics
                self._save_checkpoint(
                    path=self.checkpoint_dir / "best.pt",
                    epoch=epoch,
                    metrics=metrics,
                    best_metrics=best_metrics,
                    best_val_loss=best_val_loss,
                )

            self._save_checkpoint(
                path=self.checkpoint_dir / "latest.pt",
                epoch=epoch,
                metrics=metrics,
                best_metrics=best_metrics,
                best_val_loss=best_val_loss,
            )

        return best_metrics

    def resume_from_checkpoint(self, path: str | Path) -> Dict[str, object]:
        """Restore model and optimizer state and return resume metadata."""
        checkpoint = load_checkpoint(path, self.model, self.optimizer)
        best_metrics = checkpoint.get("best_metrics") or checkpoint["metrics"]
        best_val_loss = checkpoint.get("best_val_loss", checkpoint["metrics"]["val_loss"])
        return {
            "epoch": int(checkpoint["epoch"]),
            "best_metrics": best_metrics,
            "best_val_loss": float(best_val_loss),
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_action_accuracy = 0.0
        total_win_brier = 0.0
        total_opponent_moves_accuracy = 0.0
        total_opponent_items_accuracy = 0.0
        total_battle_length_mae = 0.0
        num_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                output = self.model(
                    pokemon_ids=batch["pokemon_ids"],
                    move_ids=batch["move_ids"],
                    item_ids=batch["item_ids"],
                    ability_ids=batch["ability_ids"],
                    team_mask=batch["team_mask"],
                    field_features=batch["field_features"],
                    hp_fractions=batch["hp_fractions"],
                    status=batch["status"],
                )
                loss = self.criterion(
                    predictions=_policy_output_to_dict(output),
                    targets={
                        "action_indices": batch["action_index"],
                        "win_target": batch["win_target"],
                        "opponent_move_indices": batch["opponent_move_indices"],
                        "opponent_item_indices": batch["opponent_item_indices"],
                        "battle_length_targets": batch["battle_length_target"],
                    },
                    action_mask=batch["action_mask"],
                )

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                self.optimizer.step()

            metrics = summarize_metrics(
                action_logits=output.action_logits,
                win_probability=output.win_probability,
                opponent_moves=output.opponent_moves,
                opponent_items=output.opponent_items,
                battle_length=output.battle_length,
                batch=batch,
            )

            total_loss += loss.item()
            total_action_accuracy += metrics["action_accuracy"]
            total_win_brier += metrics["win_brier_score"]
            total_opponent_moves_accuracy += metrics["opponent_moves_accuracy"]
            total_opponent_items_accuracy += metrics["opponent_items_accuracy"]
            total_battle_length_mae += metrics["battle_length_mae"]
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot run training or evaluation on an empty dataloader")

        return {
            "loss": total_loss / num_batches,
            "action_accuracy": total_action_accuracy / num_batches,
            "win_brier_score": total_win_brier / num_batches,
            "opponent_moves_accuracy": total_opponent_moves_accuracy / num_batches,
            "opponent_items_accuracy": total_opponent_items_accuracy / num_batches,
            "battle_length_mae": total_battle_length_mae / num_batches,
        }

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
        best_metrics: Dict[str, float],
        best_val_loss: float,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "metrics": metrics,
                "best_metrics": best_metrics,
                "best_val_loss": best_val_loss,
                "config": _serialize_config(self.config),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def _append_history(self, metrics: Dict[str, float]) -> None:
        history_path = self.checkpoint_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(metrics, sort_keys=True) + "\n")

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}


def load_checkpoint(
    path: str | Path,
    model: TransformerPolicy,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _policy_output_to_dict(output: PolicyOutput) -> Dict[str, torch.Tensor]:
    return {
        "action_logits": output.action_logits,
        "win_probability": output.win_probability,
        "opponent_moves": output.opponent_moves,
        "opponent_items": output.opponent_items,
        "battle_length": output.battle_length,
    }


def _serialize_config(config: SupervisedTrainingConfig) -> Dict[str, object]:
    payload = asdict(config)
    payload["checkpoint_dir"] = str(payload["checkpoint_dir"])
    return payload
