import json

import torch

from src.models.policy import TransformerPolicy
from src.training.native_ppo_train import NativePPOTrainer, load_native_records


def test_native_trainer_updates_and_saves(tmp_path):
    record = {
        "teams_differ": True,
        "teams": {"player1": [], "player2": []},
        "trajectory": [{
            "observations": {"player1": [0.0] * 512, "player2": [0.0] * 512},
            "action_masks": {"player1": [1] * 214, "player2": [1] * 214},
            "actions": {"player1": [0, 1], "player2": [0, 1]},
            "rewards": {"player1": 1.0, "player2": -1.0},
            "log_probs": {"player1": -1.0, "player2": -1.0},
            "values": {"player1": 0.5, "player2": 0.5},
        }],
    }
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert len(load_native_records(path)) == 1
    model = TransformerPolicy(embedding_dim=16, num_transformer_layers=1, ff_dim=32)
    trainer = NativePPOTrainer(model, tmp_path / "checkpoints")
    metrics = trainer.update(load_native_records(path))
    assert metrics["transitions"] == 2
    trainer.save(tmp_path / "checkpoints" / "latest.pt")
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
