"""PPO updates over native Showdown trajectory JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import AdamW

from src.models.policy import TransformerPolicy
from src.training.data import build_model_inputs
from src.training.native_ppo import compute_native_gae, extract_player_transitions, valid_native_records


def load_native_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and filter complete native trajectory records."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return valid_native_records(records)


class NativePPOTrainer:
    """Small, resumable PPO trainer for real Showdown trajectories."""

    def __init__(self, model: TransformerPolicy, checkpoint_dir: str | Path, lr: float = 3e-4):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def update(self, records: Iterable[dict[str, Any]], device: str = "cpu") -> dict[str, float]:
        self.model.to(device).train()
        total_loss = torch.zeros((), device=device)
        transitions = 0
        for record in records:
            teams = record["teams"]
            for player, opponent in (("player1", "player2"), ("player2", "player1")):
                action_keys = list(record["trajectory"][0]["actions"])
                player_key = action_keys[0 if player == "player1" else 1]
                opponent_key = action_keys[1 if player == "player1" else 0]
                extracted = extract_player_transitions(record, player_key)
                advantages, returns = compute_native_gae(extracted)
                for index, transition in enumerate(extracted):
                    inputs = build_model_inputs(
                        teams[player], teams[opponent],
                        transition.observation, transition.action_mask,
                    )
                    batch = {
                        key: value.unsqueeze(0).to(device)
                        for key, value in inputs.items()
                        if key not in {"action_mask", "opponent_move_indices", "opponent_item_indices"}
                    }
                    output = self.model(**batch)
                    mask = transition.action_mask
                    half = mask.size // 2
                    current_log_prob = torch.zeros((), device=device)
                    for slot, action in enumerate(transition.action):
                        slot_mask = torch.zeros(126, dtype=torch.bool, device=device)
                        slot_mask[:half] = torch.as_tensor(
                            mask[slot * half:(slot + 1) * half], dtype=torch.bool, device=device
                        )
                        logits = output.action_logits.masked_fill(~slot_mask.unsqueeze(0), -torch.inf)
                        current_log_prob = current_log_prob + torch.log_softmax(logits, dim=1)[0, int(action)]
                    advantage = advantages[0, index].to(device)
                    total_loss = total_loss + (
                        -(current_log_prob - transition.old_log_prob).exp() * advantage
                        + 0.5 * (output.win_probability[0, 0] - returns[0, index].to(device)).pow(2)
                    )
                    transitions += 1
        if transitions == 0:
            raise ValueError("no valid native transitions available")
        loss = total_loss / transitions
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu()), "transitions": float(transitions)}

    def save(self, path: str | Path) -> None:
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()}, path)
