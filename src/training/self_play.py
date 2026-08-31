"""Self-play infrastructure: policy-as-agent wrapper and opponent pool.

`PolicyAgent` adapts a `TransformerPolicy` to the `Agent` interface so it can
be dropped into `BattleGenerator`/rollout loops alongside heuristic agents.
`OpponentPool` mixes heuristic baselines with past policy checkpoints so
self-play doesn't collapse against a single frozen opponent.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.baselines.base_agent import Agent
from src.baselines.max_damage_agent import MaxDamageAgent
from src.baselines.random_agent import RandomAgent
from src.models.inference import PolicyInference
from src.models.policy import TransformerPolicy
from src.training.data import build_model_inputs


class PolicyAgent(Agent):
    """Wraps a `TransformerPolicy` so it can act as a battle agent.

    Also exposes `predict_with_value`, returning the log-probability and
    win-probability (used as a value baseline) of the selected action so
    self-play rollouts can be used directly for PPO training.
    """

    def __init__(
        self,
        model: TransformerPolicy,
        player_team: Sequence[Dict[str, Any]],
        opponent_team: Sequence[Dict[str, Any]],
        deterministic: bool = False,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.player_team = player_team
        self.opponent_team = opponent_team
        self.deterministic = deterministic
        self.temperature = temperature
        self.device = torch.device(device)

    def _forward(self, observation: np.ndarray, legal_actions_mask: np.ndarray):
        inputs = build_model_inputs(
            player_team=self.player_team,
            opponent_team=self.opponent_team,
            state=observation,
            legal_actions_mask=legal_actions_mask,
        )
        batched = {key: value.unsqueeze(0).to(self.device) for key, value in inputs.items()}
        action_mask = batched.pop("action_mask")
        batched.pop("opponent_move_indices", None)
        batched.pop("opponent_item_indices", None)

        with torch.no_grad():
            output = self.model(
                pokemon_ids=batched["pokemon_ids"],
                move_ids=batched["move_ids"],
                item_ids=batched["item_ids"],
                ability_ids=batched["ability_ids"],
                team_mask=batched["team_mask"],
                field_features=batched["field_features"],
                hp_fractions=batched["hp_fractions"],
                status=batched["status"],
            )
        return output, action_mask

    def predict(self, observation: np.ndarray, legal_actions_mask: np.ndarray) -> int:
        action, _, _ = self.predict_with_value(observation, legal_actions_mask)
        return action

    def predict_with_value(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Select an action and return (action, log_prob, value_estimate).

        `value_estimate` uses the model's win-probability head as a value
        baseline, since a dedicated value head is not (yet) part of
        `TransformerPolicy`.
        """
        output, action_mask = self._forward(observation, legal_actions_mask)

        actions, probs = PolicyInference.batch_inference(
            output.action_logits,
            action_mask,
            sample=not self.deterministic,
            temperature=self.temperature,
        )
        action = int(actions.item())
        log_prob = float(torch.log(probs.clamp(min=1e-8)).item())
        value = float(output.win_probability.squeeze(1).item())
        return action, log_prob, value

    def predict_native_with_value(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Select two masked native doubles actions.

        The native poke-env action mask concatenates one 126-action mask per
        active Pokémon, while the policy's action head represents one action.
        Each slot is therefore sampled independently from the shared battle
        representation.
        """
        mask = np.asarray(legal_actions_mask, dtype=np.int8)
        if mask.size % 2 or mask.size // 2 > 126:
            raise ValueError(f"Expected two masks with at most 126 actions, got {mask.size}")
        actions: List[int] = []
        log_prob = 0.0
        values: List[float] = []
        for slot_mask in np.split(mask, 2):
            padded_mask = np.zeros(126, dtype=np.int8)
            padded_mask[: slot_mask.size] = slot_mask
            action, slot_log_prob, value = self.predict_with_value(observation, padded_mask)
            actions.append(action)
            log_prob += slot_log_prob
            values.append(value)
        return np.asarray(actions, dtype=np.int64), log_prob, float(np.mean(values))


class OpponentPool:
    """Samples opponents for self-play: heuristics + past policy checkpoints."""

    def __init__(
        self,
        heuristic_agent_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
        checkpoint_sample_probability: float = 0.5,
    ):
        self.heuristic_agent_types = heuristic_agent_types or ["random", "max_damage"]
        self.checkpoint_sample_probability = checkpoint_sample_probability
        self._checkpoint_paths: List[str] = []
        self._rng = random.Random(seed)

    def add_checkpoint(self, path: str | Path) -> None:
        self._checkpoint_paths.append(str(path))

    def num_checkpoints(self) -> int:
        return len(self._checkpoint_paths)

    def _sample_heuristic(self) -> Agent:
        agent_type = self._rng.choice(self.heuristic_agent_types)
        if agent_type == "random":
            return RandomAgent(seed=self._rng.randint(0, 2**31 - 1))
        if agent_type == "max_damage":
            return MaxDamageAgent()
        raise ValueError(f"Unknown heuristic agent type: {agent_type}")

    def sample(
        self,
        player_team: Optional[Sequence[Dict[str, Any]]] = None,
        opponent_team: Optional[Sequence[Dict[str, Any]]] = None,
        model_factory: Optional[Callable[[], TransformerPolicy]] = None,
        device: str = "cpu",
    ) -> Agent:
        """Sample an opponent agent.

        Falls back to a heuristic agent whenever there are no checkpoints
        available yet, or `model_factory`/team info isn't supplied (needed to
        instantiate a `PolicyAgent`).
        """
        use_checkpoint = (
            self._checkpoint_paths
            and model_factory is not None
            and player_team is not None
            and opponent_team is not None
            and self._rng.random() < self.checkpoint_sample_probability
        )
        if use_checkpoint:
            checkpoint_path = self._rng.choice(self._checkpoint_paths)
            model = model_factory()
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return PolicyAgent(
                model=model,
                player_team=opponent_team,
                opponent_team=player_team,
                deterministic=False,
                device=device,
            )
        return self._sample_heuristic()
