import numpy as np
import torch

from src.training.native_ppo import (
    NativeTransition,
    compute_native_gae,
    factorized_ppo_loss,
    valid_native_records,
)


def _record(timeout=False):
    return {
        "timed_out": timeout,
        "teams_differ": True,
        "trajectory": [{
            "observations": {"p": [0.0]},
            "action_masks": {"p": [1, 0, 1, 0]},
            "actions": {"p": [0, 2]},
            "rewards": {"p": 1.0},
            "log_probs": {"p": -0.5},
            "values": {"p": 0.2},
        }],
    }


def test_filters_invalid_native_records():
    assert len(valid_native_records([_record(), _record(timeout=True)])) == 1


def test_native_gae_and_factorized_loss():
    transition = NativeTransition(np.zeros(2), np.ones(4), np.array([0, 2]), 1.0, -0.5, 0.2, True)
    advantages, returns = compute_native_gae([transition])
    loss = factorized_ppo_loss(
        torch.tensor([[-0.4]]), torch.tensor([[-0.5]]), advantages, torch.tensor([[0.2]]), returns
    )
    assert torch.isfinite(loss)
