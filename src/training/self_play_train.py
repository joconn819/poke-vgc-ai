"""Self-play PPO training loop for the VGC battle policy.

Generates rollouts by having the current policy play against an opponent
pool (heuristics + past checkpoints), computes GAE advantages, and updates
the policy with a clipped PPO objective. The model's win-probability head
doubles as the value function baseline.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW

from src.env.vgc_env import VGCDoublesEnv
from src.models.policy import TransformerPolicy
from src.training.data import _team_members, build_model_inputs  # noqa: F401 (shared helpers)
from src.training.ppo import compute_gae, ppo_policy_loss, ppo_value_loss
from src.training.self_play import OpponentPool, PolicyAgent
from src.utils.config import RegulationConfig
from src.utils.team_pool import TeamPool

_MODEL_INPUT_KEYS = (
    "pokemon_ids",
    "move_ids",
    "item_ids",
    "ability_ids",
    "team_mask",
    "field_features",
    "hp_fractions",
    "status",
)


@dataclass
class RolloutBatch:
    """Padded batch of self-play episodes for PPO updates."""

    inputs: Dict[str, torch.Tensor]  # each (num_episodes, max_steps, ...)
    action_mask: torch.Tensor  # (num_episodes, max_steps, 126)
    actions: torch.Tensor  # (num_episodes, max_steps)
    old_log_probs: torch.Tensor  # (num_episodes, max_steps)
    values: torch.Tensor  # (num_episodes, max_steps)
    rewards: torch.Tensor  # (num_episodes, max_steps)
    step_mask: torch.Tensor  # (num_episodes, max_steps), 1.0 for valid steps
    dones: torch.Tensor  # (num_episodes, max_steps)


def generate_rollouts(
    model: TransformerPolicy,
    team_pool: TeamPool,
    opponent_pool: OpponentPool,
    num_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> RolloutBatch:
    """Play `num_episodes` self-play battles and collect padded rollout tensors."""
    rng = np.random.RandomState(seed)
    model.eval()

    per_step_inputs: List[List[Dict[str, torch.Tensor]]] = []
    per_episode_actions: List[List[int]] = []
    per_episode_log_probs: List[List[float]] = []
    per_episode_values: List[List[float]] = []
    per_episode_rewards: List[List[float]] = []
    per_episode_masks: List[List[np.ndarray]] = []
    per_episode_dones: List[List[float]] = []

    regulation_config: RegulationConfig = team_pool.regulation

    for episode_index in range(num_episodes):
        player_team = _team_members(team_pool.sample_team())
        opponent_team = _team_members(team_pool.sample_team())

        env = VGCDoublesEnv(
            regulation_config=regulation_config,
            player_team=player_team,
            opponent_team=opponent_team,
        )

        player_agent = PolicyAgent(
            model=model,
            player_team=player_team,
            opponent_team=opponent_team,
            deterministic=False,
            device=device,
        )
        opponent_agent = opponent_pool.sample(
            player_team=opponent_team,
            opponent_team=player_team,
            model_factory=lambda: copy.deepcopy(model),
            device=device,
        )

        observation, _ = env.reset()

        step_inputs: List[Dict[str, torch.Tensor]] = []
        actions: List[int] = []
        log_probs: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        masks: List[np.ndarray] = []
        dones: List[float] = []

        winner: Optional[str] = None
        for step_index in range(max_steps):
            legal_mask, _ = env._get_legal_moves()

            action, log_prob, value = player_agent.predict_with_value(observation, legal_mask)
            opponent_agent.predict(observation, legal_mask)

            model_inputs = build_model_inputs(
                player_team=player_team,
                opponent_team=opponent_team,
                state=observation,
                legal_actions_mask=legal_mask,
            )

            next_observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or step_index == max_steps - 1

            if done:
                # Deterministic outcome resolution for the placeholder env:
                # alternate winner by RNG so terminal signal is nontrivial
                # while the richer env dynamics are still being built out.
                winner = "player" if rng.random() < 0.5 else "opponent"
                reward = 1.0 if winner == "player" else -1.0
            else:
                reward = 0.0

            step_inputs.append(model_inputs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(legal_mask.astype(np.uint8))
            dones.append(1.0 if done else 0.0)

            observation = next_observation
            if done:
                break

        per_step_inputs.append(step_inputs)
        per_episode_actions.append(actions)
        per_episode_log_probs.append(log_probs)
        per_episode_values.append(values)
        per_episode_rewards.append(rewards)
        per_episode_masks.append(masks)
        per_episode_dones.append(dones)

    return _pad_rollouts(
        per_step_inputs,
        per_episode_actions,
        per_episode_log_probs,
        per_episode_values,
        per_episode_rewards,
        per_episode_masks,
        per_episode_dones,
        max_steps=max_steps,
    )


def _pad_rollouts(
    per_step_inputs: List[List[Dict[str, torch.Tensor]]],
    per_episode_actions: List[List[int]],
    per_episode_log_probs: List[List[float]],
    per_episode_values: List[List[float]],
    per_episode_rewards: List[List[float]],
    per_episode_masks: List[List[np.ndarray]],
    per_episode_dones: List[List[float]],
    max_steps: int,
) -> RolloutBatch:
    num_episodes = len(per_step_inputs)

    sample_input = per_step_inputs[0][0]
    inputs: Dict[str, torch.Tensor] = {
        key: torch.zeros((num_episodes, max_steps, *sample_input[key].shape), dtype=sample_input[key].dtype)
        for key in _MODEL_INPUT_KEYS
    }
    action_mask = torch.zeros((num_episodes, max_steps, 126), dtype=torch.bool)
    actions = torch.zeros((num_episodes, max_steps), dtype=torch.long)
    old_log_probs = torch.zeros((num_episodes, max_steps), dtype=torch.float32)
    values = torch.zeros((num_episodes, max_steps), dtype=torch.float32)
    rewards = torch.zeros((num_episodes, max_steps), dtype=torch.float32)
    step_mask = torch.zeros((num_episodes, max_steps), dtype=torch.float32)
    dones = torch.zeros((num_episodes, max_steps), dtype=torch.float32)

    for episode_index in range(num_episodes):
        episode_len = len(per_step_inputs[episode_index])
        for step_index in range(episode_len):
            step_input = per_step_inputs[episode_index][step_index]
            for key in _MODEL_INPUT_KEYS:
                inputs[key][episode_index, step_index] = step_input[key]
            action_mask[episode_index, step_index] = torch.as_tensor(
                per_episode_masks[episode_index][step_index], dtype=torch.bool
            )
        actions[episode_index, :episode_len] = torch.tensor(
            per_episode_actions[episode_index], dtype=torch.long
        )
        old_log_probs[episode_index, :episode_len] = torch.tensor(
            per_episode_log_probs[episode_index], dtype=torch.float32
        )
        values[episode_index, :episode_len] = torch.tensor(
            per_episode_values[episode_index], dtype=torch.float32
        )
        rewards[episode_index, :episode_len] = torch.tensor(
            per_episode_rewards[episode_index], dtype=torch.float32
        )
        dones[episode_index, :episode_len] = torch.tensor(
            per_episode_dones[episode_index], dtype=torch.float32
        )
        step_mask[episode_index, :episode_len] = 1.0

    return RolloutBatch(
        inputs=inputs,
        action_mask=action_mask,
        actions=actions,
        old_log_probs=old_log_probs,
        values=values,
        rewards=rewards,
        step_mask=step_mask,
        dones=dones,
    )


@dataclass
class SelfPlayConfig:
    """Configuration for the self-play PPO trainer."""

    checkpoint_dir: str | Path = "data/checkpoints/self_play"
    device: str = "cpu"
    episodes_per_update: int = 16
    max_steps_per_episode: int = 30
    epochs_per_update: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    heuristic_agent_types: Optional[List[str]] = None
    checkpoint_sample_probability: float = 0.3
    add_self_to_pool_every: int = 5
    init_from_checkpoint: Optional[str] = None
    team_pool_path: str = "data/teams/champions_mb.json"
    seed: Optional[int] = None


class SelfPlayTrainer:
    """Runs iterative self-play PPO updates."""

    def __init__(
        self,
        model: TransformerPolicy,
        config: SelfPlayConfig,
        team_pool: TeamPool,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.team_pool = team_pool
        self.opponent_pool = OpponentPool(
            heuristic_agent_types=config.heuristic_agent_types,
            seed=config.seed,
            checkpoint_sample_probability=config.checkpoint_sample_probability,
        )
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.init_from_checkpoint:
            checkpoint = torch.load(config.init_from_checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def resume_from_checkpoint(self, path: str | Path) -> int:
        """Load model/optimizer state from a self-play checkpoint.

        Returns the iteration index the checkpoint was saved at, so callers
        can resume the iteration loop from `iteration + 1`.
        """
        checkpoint = torch.load(Path(path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint["iteration"])

    def train_iteration(self, iteration: int) -> Dict[str, float]:
        batch = generate_rollouts(
            model=self.model,
            team_pool=self.team_pool,
            opponent_pool=self.opponent_pool,
            num_episodes=self.config.episodes_per_update,
            max_steps=self.config.max_steps_per_episode,
            seed=None if self.config.seed is None else self.config.seed + iteration,
            device=self.config.device,
        )

        advantages, returns = compute_gae(
            batch.rewards, batch.values, batch.dones, gamma=self.config.gamma, lam=self.config.gae_lambda
        )
        # Normalize advantages over valid steps only for stabler updates.
        valid = batch.step_mask.bool()
        if valid.any():
            valid_adv = advantages[valid]
            advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        self.model.train()
        last_metrics: Dict[str, float] = {}
        for _ in range(self.config.epochs_per_update):
            last_metrics = self._update_step(batch, advantages, returns)

        self._save_checkpoint(iteration, last_metrics)

        if (iteration + 1) % self.config.add_self_to_pool_every == 0:
            self.opponent_pool.add_checkpoint(str(self.checkpoint_dir / "latest.pt"))

        last_metrics["mean_reward"] = float(
            (batch.rewards * batch.step_mask).sum() / batch.step_mask.sum().clamp(min=1)
        )
        last_metrics["iteration"] = iteration
        self._append_history(last_metrics)
        return last_metrics

    def _update_step(
        self,
        batch: RolloutBatch,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        num_episodes, max_steps = batch.actions.shape
        flat_mask = batch.step_mask.reshape(-1).bool()

        flat_inputs = {
            key: value.reshape(num_episodes * max_steps, *value.shape[2:])[flat_mask].to(self.device)
            for key, value in batch.inputs.items()
        }
        flat_action_mask = batch.action_mask.reshape(num_episodes * max_steps, 126)[flat_mask].to(
            self.device
        )
        flat_actions = batch.actions.reshape(-1)[flat_mask].to(self.device)
        flat_old_log_probs = batch.old_log_probs.reshape(-1)[flat_mask].to(self.device)
        flat_advantages = advantages.reshape(-1)[flat_mask].to(self.device)
        flat_returns = returns.reshape(-1)[flat_mask].to(self.device)

        output = self.model(
            pokemon_ids=flat_inputs["pokemon_ids"],
            move_ids=flat_inputs["move_ids"],
            item_ids=flat_inputs["item_ids"],
            ability_ids=flat_inputs["ability_ids"],
            team_mask=flat_inputs["team_mask"],
            field_features=flat_inputs["field_features"],
            hp_fractions=flat_inputs["hp_fractions"],
            status=flat_inputs["status"],
        )

        masked_logits = output.action_logits.clone()
        masked_logits[~flat_action_mask] = float("-inf")
        log_probs_all = torch.log_softmax(masked_logits, dim=1)
        log_probs = log_probs_all.gather(1, flat_actions.unsqueeze(1)).squeeze(1)

        values = output.win_probability.squeeze(1)

        policy_loss = ppo_policy_loss(
            log_probs, flat_old_log_probs, flat_advantages, clip_epsilon=self.config.clip_epsilon
        )
        value_loss = ppo_value_loss(values, flat_returns.clamp(0.0, 1.0))
        loss = policy_loss + self.config.value_loss_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(loss.item()),
        }

    def _save_checkpoint(self, iteration: int, metrics: Dict[str, float]) -> None:
        payload = {
            "iteration": iteration,
            "metrics": metrics,
            "config": _serialize_config(self.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, self.checkpoint_dir / "latest.pt")
        torch.save(payload, self.checkpoint_dir / f"iter_{iteration:06d}.pt")

    def _append_history(self, metrics: Dict[str, float]) -> None:
        import json

        history_path = self.checkpoint_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(metrics, sort_keys=True) + "\n")


def _serialize_config(config: SelfPlayConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["checkpoint_dir"] = str(payload["checkpoint_dir"])
    return payload
