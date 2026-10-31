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
from src.training.native_ppo import (
    estimate_native_action_features,
    normalize_counterplay_score_row,
    resolve_native_action_feature_variant,
)


class PolicyAgent(Agent):
    """Wraps a `TransformerPolicy` so it can act as a battle agent.

    Also exposes `predict_with_value`, returning the log-probability and
    win-probability (used as a value baseline) of the selected action so
    self-play rollouts can be used directly for PPO training.
    """

    @property
    def opponent_team(self) -> list[dict[str, str]]:
        return self._opponent_preview

    @opponent_team.setter
    def opponent_team(self, team: Sequence[Dict[str, Any]]) -> None:
        # Team preview exposes species only. Strip hidden sets whenever the
        # policy receives opponent roster data, including later refreshes.
        self._opponent_preview = [
            {"species": str(member.get("species") or member.get("name"))}
            for member in team
            if isinstance(member, dict) and (member.get("species") or member.get("name"))
        ]

    def __init__(
        self,
        model: TransformerPolicy,
        player_team: Sequence[Dict[str, Any]],
        opponent_team: Sequence[Dict[str, Any]],
        deterministic: bool = False,
        temperature: float = 1.0,
        device: str = "cpu",
        history_observations: Optional[np.ndarray] = None,
        belief_features: Optional[np.ndarray] = None,
        public_context_features: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.player_team = player_team
        # The property strips hidden opponent sets before any feature code sees
        # this data, even when the coordinator supplies omniscient teams.
        self.opponent_team = opponent_team
        self.deterministic = deterministic
        self.temperature = temperature
        self.device = torch.device(device)
        self.history_observations = history_observations
        self.belief_features = belief_features
        self.public_context_features = public_context_features
        self.last_native_action_features: Optional[np.ndarray] = None
        # Team roster and sets stay fixed for the duration of a match. Cache
        # their token tensors so every decision only encodes turn-dependent
        # observation fields. The cache key includes the visible team data.
        self._static_team_cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        self.model.to(self.device)

    def _forward(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        history_observations: Optional[np.ndarray] = None,
        belief_features: Optional[np.ndarray] = None,
        native_action_features: Optional[np.ndarray] = None,
        previous_action_indices: Optional[torch.Tensor] = None,
        public_context_features: Optional[np.ndarray] = None,
    ):
        inputs = build_model_inputs(
            player_team=self.player_team,
            opponent_team=self.opponent_team,
            state=observation,
            legal_actions_mask=legal_actions_mask,
            token_vocab_version=self.model.token_vocab_version,
            _static_team_cache=self._static_team_cache,
        )
        batched = {key: value.unsqueeze(0).to(self.device) for key, value in inputs.items()}
        action_mask = batched.pop("action_mask")
        history = self.history_observations if history_observations is None else history_observations
        beliefs = self.belief_features if belief_features is None else belief_features
        if history is not None:
            history_tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device)
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)
            batched["history_observations"] = history_tensor
        if beliefs is not None:
            belief_tensor = torch.as_tensor(beliefs, dtype=torch.float32, device=self.device)
            if belief_tensor.ndim == 2:
                belief_tensor = belief_tensor.unsqueeze(0)
            batched["belief_features"] = belief_tensor
        if native_action_features is not None:
            native_tensor = torch.as_tensor(
                native_action_features, dtype=torch.float32, device=self.device
            )
            if native_tensor.ndim == 3:
                native_tensor = native_tensor.unsqueeze(0)
            batched["native_action_features"] = native_tensor
        if previous_action_indices is not None:
            batched["previous_action_indices"] = previous_action_indices
        public_context = (
            self.public_context_features
            if public_context_features is None else public_context_features
        )
        if getattr(self.model, "public_context_dim", 0):
            if public_context is None:
                public_context = np.zeros(self.model.public_context_dim, dtype=np.float32)
            context_tensor = torch.as_tensor(
                public_context, dtype=torch.float32, device=self.device
            ).reshape(1, -1)
            if context_tensor.shape[1] != self.model.public_context_dim:
                raise ValueError("public mechanics context has the wrong feature dimension")
            batched["public_context_features"] = context_tensor

        model_inputs = {
            "pokemon_ids": batched["pokemon_ids"],
            "move_ids": batched["move_ids"],
            "item_ids": batched["item_ids"],
            "ability_ids": batched["ability_ids"],
            "team_mask": batched["team_mask"],
            "field_features": batched["field_features"],
            "hp_fractions": batched["hp_fractions"],
            "status": batched["status"],
            "opponent_pokemon_ids": batched["opponent_pokemon_ids"],
            "opponent_move_ids": batched["opponent_move_ids"],
            "opponent_recent_move_ids": batched["opponent_recent_move_ids"],
            "opponent_item_ids": batched["opponent_item_ids"],
            "opponent_ability_ids": batched["opponent_ability_ids"],
            "opponent_team_mask": batched["opponent_team_mask"],
            "opponent_hp_fractions": batched["opponent_hp_fractions"],
            "opponent_status": batched["opponent_status"],
        }
        if "history_observations" in batched:
            model_inputs["history_observations"] = batched["history_observations"]
        if "belief_features" in batched:
            model_inputs["belief_features"] = batched["belief_features"]
        if "native_action_features" in batched:
            model_inputs["native_action_features"] = batched["native_action_features"]
        if "previous_action_indices" in batched:
            model_inputs["previous_action_indices"] = batched["previous_action_indices"]
        if "public_context_features" in batched:
            model_inputs["public_context_features"] = batched["public_context_features"]
        with torch.inference_mode():
            output = self.model(**model_inputs)
        return output, action_mask

    def build_native_action_features(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        predicted_opponent_action: Optional[Sequence[int]] = None,
    ) -> Optional[np.ndarray]:
        feature_variant, feature_dim = resolve_native_action_feature_variant(
            getattr(self.model, "native_action_feature_variant", None),
            getattr(self.model, "native_action_feature_dim", 0),
        )
        if feature_dim <= 0 or feature_variant is None:
            return None
        return estimate_native_action_features(
            self.player_team,
            self.opponent_team,
            observation,
            legal_actions_mask,
            feature_variant=feature_variant,
            predicted_opponent_action=predicted_opponent_action,
        )

    def predict(self, observation: np.ndarray, legal_actions_mask: np.ndarray) -> int:
        action, _, _ = self.predict_with_value(observation, legal_actions_mask)
        return action

    def predict_with_value(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        history_observations: Optional[np.ndarray] = None,
        belief_features: Optional[np.ndarray] = None,
        *,
        use_critic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select an action and return (action, log_prob, value_estimate).

        By default the third result is the bounded win-probability estimate.
        PPO rollout collection sets ``use_critic=True`` to retain the separate
        unbounded critic baseline used by its value loss.
        """
        output, action_mask = self._forward(
            observation, legal_actions_mask, history_observations, belief_features
        )

        actions, probs = PolicyInference.batch_inference(
            output.action_logits,
            action_mask,
            sample=not self.deterministic,
            temperature=self.temperature,
        )
        action = int(actions.item())
        log_prob = float(torch.log(probs.clamp(min=1e-8)).item())
        value_head = output.value if use_critic and output.value is not None else output.win_probability
        value = float(value_head.squeeze(1).item())
        if not use_critic:
            value = float(np.clip(value, 0.0, 1.0))
        return action, log_prob, value

    def predict_native_with_value(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        history_observations: Optional[np.ndarray] = None,
        belief_features: Optional[np.ndarray] = None,
        predicted_opponent_action: Optional[Sequence[int]] = None,
        counterplay_action_scores: Optional[np.ndarray] = None,
        counterplay_pair_scores: Optional[np.ndarray] = None,
        counterplay_prior_weight: float = 0.0,
        counterplay_prior_min_margin: float = 0.0,
        counterplay_teacher_first_action: Optional[int] = None,
        *,
        use_critic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Select two masked native doubles actions.

        Native poke-env currently exposes 107 legal indices per active
        Pokémon; we pad each slot to the policy's 126-wide action head so
        native checkpoints remain compatible with the broader PPO model.
        """
        mask = np.asarray(legal_actions_mask, dtype=np.int8)
        if mask.size % 2 or mask.size // 2 > 126:
            raise ValueError(f"Expected two masks with at most 126 actions, got {mask.size}")
        native_action_features = self.build_native_action_features(
            observation, legal_actions_mask, predicted_opponent_action
        )
        if not np.isfinite(counterplay_prior_weight) or counterplay_prior_weight < 0:
            raise ValueError("counterplay_prior_weight must be finite and nonnegative")
        if not np.isfinite(counterplay_prior_min_margin) or counterplay_prior_min_margin < 0:
            raise ValueError("counterplay_prior_min_margin must be finite and nonnegative")
        score_prior = None
        if counterplay_action_scores is not None:
            score_prior = np.asarray(counterplay_action_scores, dtype=np.float32)
            if score_prior.ndim != 2 or score_prior.shape[0] != 2 or score_prior.shape[1] > 126:
                raise ValueError("counterplay_action_scores must have shape (2, <=126)")
            if not np.isfinite(score_prior).all():
                raise ValueError("counterplay_action_scores must be finite")
            if score_prior.shape[1] < 126:
                score_prior = np.pad(score_prior, ((0, 0), (0, 126 - score_prior.shape[1])))
        pair_score_prior = None
        if counterplay_pair_scores is not None:
            pair_score_prior = np.asarray(counterplay_pair_scores, dtype=np.float32)
            if (
                pair_score_prior.ndim != 2
                or pair_score_prior.shape[0] > 126
                or pair_score_prior.shape[1] > 126
                or not np.isfinite(pair_score_prior).all()
            ):
                raise ValueError("counterplay_pair_scores must have shape (<=126, <=126)")
            if pair_score_prior.shape != (126, 126):
                pair_score_prior = np.pad(
                    pair_score_prior,
                    (
                        (0, 126 - pair_score_prior.shape[0]),
                        (0, 126 - pair_score_prior.shape[1]),
                    ),
                    constant_values=-1e6,
                )
        self.last_native_action_features = native_action_features
        actions: List[int] = []
        log_prob = 0.0
        values: List[float] = []
        padded_masks = []
        for slot_mask in np.split(mask, 2):
            padded_mask = np.zeros(126, dtype=np.int8)
            padded_mask[: slot_mask.size] = slot_mask
            padded_masks.append(padded_mask)

        # Both native action heads share the same encoded state. The second
        # head's conditioning on the first choice is only an additive learned
        # projection in TransformerPolicy.forward, so run the expensive state
        # encoder once and apply that projection after sampling slot one.
        output, first_action_mask = self._forward(
            observation,
            padded_masks[0],
            history_observations,
            belief_features,
            native_action_features,
        )
        for slot, padded_mask in enumerate(padded_masks):
            action_mask = (
                first_action_mask
                if slot == 0
                else torch.as_tensor(
                    padded_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
            )
            if slot == 1 and actions:
                previous = actions[0]
                # Showdown rejects two simultaneous orders for the same
                # switch target, and rejects pass/pass when another order is
                # legal. Constrain the second draw before sampling so the
                # recorded log-probability matches the submitted action.
                if previous == 0:
                    action_mask[0, 0] = False
                elif 1 <= previous <= 6:
                    action_mask[0, previous] = False
                if not torch.any(action_mask):
                    action_mask[0] = torch.as_tensor(padded_mask, dtype=torch.bool, device=self.device)
            logits = (
                output.native_action_logits[:, slot]
                if output.native_action_logits is not None
                else output.action_logits
            )
            if slot == 1 and actions and output.native_action_logits is not None:
                previous_action = torch.tensor(
                    [actions[0]], dtype=torch.long, device=self.device
                )
                if output.state_context is None:
                    raise RuntimeError("native policy output is missing its state context")
                logits = self.model.condition_native_second_action_logits(
                    logits, output.state_context, previous_action
                )
            if counterplay_prior_weight and (
                score_prior is not None or pair_score_prior is not None
            ):
                prior_applies = (
                    slot == 0
                    or pair_score_prior is not None
                    or (
                        counterplay_teacher_first_action is not None
                        and actions[0] == int(counterplay_teacher_first_action)
                    )
                )
                if prior_applies:
                    legal = action_mask[0].detach().cpu().numpy().astype(bool)
                    if slot == 1 and pair_score_prior is not None:
                        prior_row = pair_score_prior[actions[0]]
                    elif score_prior is not None:
                        prior_row = score_prior[slot]
                    else:
                        prior_row = np.max(pair_score_prior, axis=1)
                    normalized = normalize_counterplay_score_row(prior_row, legal)
                    supported_scores = normalized[legal]
                    prior_margin = (
                        float(np.partition(supported_scores, -1)[-1] - np.partition(supported_scores, -2)[-2])
                        if supported_scores.size > 1 else 0.0
                    )
                    if prior_margin >= counterplay_prior_min_margin:
                        logits = logits + counterplay_prior_weight * torch.as_tensor(
                            normalized, dtype=logits.dtype, device=logits.device
                        ).unsqueeze(0)
            sampled, probs = PolicyInference.batch_inference(
                logits,
                action_mask,
                sample=not self.deterministic,
                temperature=self.temperature,
            )
            action = int(sampled.item())
            slot_log_prob = float(torch.log(probs.clamp(min=1e-8)).item())
            value_head = output.value if use_critic and output.value is not None else output.win_probability
            value = float(value_head.squeeze(1).item())
            if not use_critic:
                value = float(np.clip(value, 0.0, 1.0))
            actions.append(action)
            log_prob += slot_log_prob
            values.append(value)
        return np.asarray(actions, dtype=np.int64), log_prob, float(np.mean(values))

    def log_prob_native_action(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        action: Sequence[int],
    ) -> float:
        """Return the policy probability of a specified legal joint action."""
        mask = np.asarray(legal_actions_mask, dtype=np.int8)
        chosen = np.asarray(action, dtype=np.int64)
        if mask.size % 2 or mask.size // 2 > 126 or chosen.shape != (2,):
            raise ValueError("expected two native action slots with at most 126 choices")
        slot_masks = []
        for raw_slot_mask in np.split(mask, 2):
            padded = np.zeros(126, dtype=np.int8)
            padded[: len(raw_slot_mask)] = raw_slot_mask
            slot_masks.append(padded)
        for slot, selected in enumerate(chosen):
            if selected < 0 or selected >= 126 or not slot_masks[slot][selected]:
                raise ValueError("specified native joint action is illegal")

        native_features = self.build_native_action_features(observation, mask)
        output, first_mask = self._forward(
            observation, slot_masks[0], native_action_features=native_features
        )
        first_logits = (
            output.native_action_logits[:, 0]
            if output.native_action_logits is not None else output.action_logits
        )
        first_log_probs = torch.log_softmax(
            first_logits.masked_fill(~first_mask, -torch.inf) / self.temperature,
            dim=-1,
        )

        second_mask = torch.as_tensor(
            slot_masks[1], dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        if chosen[0] == 0:
            second_mask[0, 0] = False
        elif 1 <= chosen[0] <= 6:
            second_mask[0, int(chosen[0])] = False
        if not second_mask.any():
            second_mask[0] = torch.as_tensor(
                slot_masks[1], dtype=torch.bool, device=self.device
            )
        second_logits = (
            output.native_action_logits[:, 1]
            if output.native_action_logits is not None else output.action_logits
        )
        if output.native_action_logits is not None:
            previous_action = torch.as_tensor(
                [int(chosen[0])], dtype=torch.long, device=self.device
            )
            if output.state_context is None:
                raise RuntimeError("native policy output is missing its state context")
            second_logits = self.model.condition_native_second_action_logits(
                second_logits, output.state_context, previous_action
            )
        second_log_probs = torch.log_softmax(
            second_logits.masked_fill(~second_mask, -torch.inf) / self.temperature,
            dim=-1,
        )
        return float(
            first_log_probs[0, int(chosen[0])].item()
            + second_log_probs[0, int(chosen[1])].item()
        )

    def log_prob_native_actions(
        self,
        observation: np.ndarray,
        legal_actions_mask: np.ndarray,
        actions: Sequence[Sequence[int]],
    ) -> list[float]:
        """Score many legal joint native actions with one shared first pass."""
        mask = np.asarray(legal_actions_mask, dtype=np.int8)
        chosen = np.asarray(actions, dtype=np.int64)
        if mask.size % 2 or mask.size // 2 > 126 or chosen.ndim != 2 or chosen.shape[1] != 2:
            raise ValueError("expected joint actions with two native slots and at most 126 choices")
        legal = [
            set(np.flatnonzero(slot_mask).tolist())
            for slot_mask in np.split(mask, 2)
        ]
        if any(
            int(pair[0]) not in legal[0] or int(pair[1]) not in legal[1]
            for pair in chosen
        ):
            raise ValueError("specified native joint action is illegal")
        padded_masks = []
        for slot_mask in np.split(mask, 2):
            padded = np.zeros(126, dtype=np.int8)
            padded[:len(slot_mask)] = slot_mask
            padded_masks.append(padded)
        native_features = self.build_native_action_features(observation, mask)
        output, first_mask = self._forward(
            observation, padded_masks[0], native_action_features=native_features
        )
        first_logits = (
            output.native_action_logits[:, 0]
            if output.native_action_logits is not None else output.action_logits
        )
        first_log_probs = torch.log_softmax(
            first_logits.masked_fill(~first_mask, -torch.inf) / self.temperature,
            dim=-1,
        )[0]
        second_logits_base = (
            output.native_action_logits[:, 1]
            if output.native_action_logits is not None else output.action_logits
        )
        scores: dict[int, torch.Tensor] = {}
        results = []
        for first_action, second_action in chosen:
            first_action, second_action = int(first_action), int(second_action)
            if first_action not in scores:
                second_mask = torch.as_tensor(
                    padded_masks[1], dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                if first_action == 0:
                    second_mask[0, 0] = False
                elif 1 <= first_action <= 6:
                    second_mask[0, first_action] = False
                if not second_mask.any():
                    second_mask[0] = torch.as_tensor(
                        padded_masks[1], dtype=torch.bool, device=self.device
                    )
                second_logits = second_logits_base
                if output.native_action_logits is not None:
                    previous_action = torch.as_tensor(
                        [first_action], dtype=torch.long, device=self.device
                    )
                    if output.state_context is None:
                        raise RuntimeError("native policy output is missing its state context")
                    second_logits = self.model.condition_native_second_action_logits(
                        second_logits, output.state_context, previous_action
                    )
                scores[first_action] = torch.log_softmax(
                    second_logits.masked_fill(~second_mask, -torch.inf)
                    / self.temperature,
                    dim=-1,
                )[0]
            results.append(float((first_log_probs[first_action] + scores[first_action][second_action]).item()))
        return results

    @staticmethod
    def predict_native_pair_with_value(
        first: "PolicyAgent",
        second: "PolicyAgent",
        first_observation: np.ndarray,
        first_mask: np.ndarray,
        second_observation: np.ndarray,
        second_mask: np.ndarray,
        *,
        use_critic: bool = False,
    ) -> Tuple[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float, float]]:
        """Infer two simultaneous decisions with one shared-model batch.

        This is intended for self-play where both sides use the same policy
        weights. Each side still receives only its own player-visible inputs.
        The two native action slots remain sequential because slot two is
        conditioned on the sampled slot-one action.
        """
        policies = (first, second)
        if first.model is not second.model or first.device != second.device:
            raise ValueError("batched native inference requires one shared model and device")
        if first.temperature != second.temperature or first.deterministic != second.deterministic:
            raise ValueError("batched native inference requires matching sampling settings")
        observations = (first_observation, second_observation)
        native_masks = tuple(np.asarray(mask, dtype=np.int8) for mask in (first_mask, second_mask))
        for mask in native_masks:
            if mask.size % 2 or mask.size // 2 > 126:
                raise ValueError(f"Expected two masks with at most 126 actions, got {mask.size}")

        base_inputs: list[dict[str, torch.Tensor]] = []
        padded_masks: list[torch.Tensor] = []
        for policy, observation, mask in zip(policies, observations, native_masks):
            native_features = policy.build_native_action_features(observation, mask)
            policy.last_native_action_features = native_features
            inputs = build_model_inputs(
                player_team=policy.player_team,
                opponent_team=policy.opponent_team,
                state=observation,
                legal_actions_mask=mask,
                token_vocab_version=policy.model.token_vocab_version,
                _static_team_cache=policy._static_team_cache,
            )
            inputs = {
                key: value.unsqueeze(0).to(policy.device)
                for key, value in inputs.items()
            }
            inputs.pop("action_mask")
            history = policy.history_observations
            if history is not None:
                history_tensor = torch.as_tensor(history, dtype=torch.float32, device=policy.device)
                if history_tensor.ndim == 2:
                    history_tensor = history_tensor.unsqueeze(0)
                inputs["history_observations"] = history_tensor
            if getattr(policy.model, "public_context_dim", 0):
                context = policy.public_context_features
                if context is None:
                    context = np.zeros(policy.model.public_context_dim, dtype=np.float32)
                inputs["public_context_features"] = torch.as_tensor(
                    context, dtype=torch.float32, device=policy.device
                ).reshape(1, -1)
            beliefs = policy.belief_features
            if beliefs is not None:
                belief_tensor = torch.as_tensor(beliefs, dtype=torch.float32, device=policy.device)
                if belief_tensor.ndim == 2:
                    belief_tensor = belief_tensor.unsqueeze(0)
                inputs["belief_features"] = belief_tensor
            if native_features is not None:
                native_tensor = torch.as_tensor(
                    native_features, dtype=torch.float32, device=policy.device
                )
                if native_tensor.ndim == 3:
                    native_tensor = native_tensor.unsqueeze(0)
                inputs["native_action_features"] = native_tensor
            base_inputs.append(inputs)
            padded = torch.zeros((2, 126), dtype=torch.bool, device=policy.device)
            halves = np.split(mask, 2)
            for slot, half in enumerate(halves):
                padded[slot, : half.size] = torch.as_tensor(
                    half, dtype=torch.bool, device=policy.device
                )
            padded_masks.append(padded)

        model_keys = (
            "pokemon_ids", "move_ids", "item_ids", "ability_ids", "team_mask",
            "field_features", "hp_fractions", "status", "opponent_pokemon_ids",
            "opponent_move_ids", "opponent_recent_move_ids", "opponent_item_ids", "opponent_ability_ids",
            "opponent_team_mask", "opponent_hp_fractions", "opponent_status",
        )

        def forward_slot(slot: int, previous_actions: Optional[Sequence[int]] = None):
            model_inputs = {
                key: torch.cat([inputs[key] for inputs in base_inputs], dim=0)
                for key in model_keys
            }
            for optional_key in (
                "history_observations", "belief_features", "native_action_features",
                "public_context_features",
            ):
                values = [inputs.get(optional_key) for inputs in base_inputs]
                if any(value is not None for value in values):
                    if any(value is None for value in values):
                        raise ValueError(f"both batched policies must provide {optional_key}")
                    model_inputs[optional_key] = torch.cat(values, dim=0)
            if previous_actions is not None:
                model_inputs["previous_action_indices"] = torch.as_tensor(
                    previous_actions, dtype=torch.long, device=first.device
                )
            with torch.inference_mode():
                output = first.model(**model_inputs)
            return output, torch.stack([mask[slot] for mask in padded_masks])

        actions_by_slot: list[list[int]] = [[], []]
        log_probs = [0.0, 0.0]
        values = [[], []]
        for slot in range(2):
            previous = [actions[0] for actions in actions_by_slot] if slot == 1 else None
            output, action_mask = forward_slot(slot, previous)
            if slot == 1:
                for row, previous_action in enumerate(previous or ()):
                    if previous_action == 0:
                        action_mask[row, 0] = False
                    elif 1 <= previous_action <= 6:
                        action_mask[row, previous_action] = False
                    if not torch.any(action_mask[row]):
                        action_mask[row] = padded_masks[row][slot]
            logits = (
                output.native_action_logits[:, slot]
                if output.native_action_logits is not None
                else output.action_logits
            )
            sampled, probabilities = PolicyInference.batch_inference(
                logits,
                action_mask,
                sample=not first.deterministic,
                temperature=first.temperature,
            )
            sampled_values = sampled.detach().cpu().tolist()
            probability_values = probabilities.detach().cpu().tolist()
            value_head = output.value if use_critic and output.value is not None else output.win_probability
            value_values = value_head.squeeze(1).detach().cpu().tolist()
            for row in range(2):
                actions_by_slot[row].append(int(sampled_values[row]))
                log_probs[row] += float(np.log(max(float(probability_values[row]), 1e-8)))
                value = float(value_values[row])
                values[row].append(value if use_critic else float(np.clip(value, 0.0, 1.0)))

        results = tuple(
            (
                np.asarray(actions_by_slot[row], dtype=np.int64),
                log_probs[row],
                float(np.mean(values[row])),
            )
            for row in range(2)
        )
        return results  # type: ignore[return-value]


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
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval()
            return PolicyAgent(
                model=model,
                player_team=opponent_team,
                opponent_team=player_team,
                deterministic=False,
                device=device,
            )
        return self._sample_heuristic()
