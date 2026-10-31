#!/usr/bin/env python3
"""Fine-tune a battle policy from paired robust action advantages.

For each player-view state, estimate every candidate's value against the
matched response policies, relative to the policy's own joint action. A
conservative soft target distills only positive robust advantages into the
actor. Team clusters and chance repeats remain disjoint for validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.models.policy import TransformerPolicy
from src.training.data import build_model_inputs


def _load_rows(path: Path) -> tuple[list[dict], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("rows", [])
    if not raw:
        raise ValueError("counterfactual dataset has no action rows")
    responses = sorted(raw[0].get("outcomes_by_opponent", {}))
    if len(responses) < 2:
        raise ValueError("need outcomes against at least two response classes")
    dedup: dict[tuple[int, int, tuple[int, int]], dict] = {}
    for row in raw:
        if not row.get("player_team") or not row.get("opponent_public_team"):
            raise ValueError("rows need player_team and opponent_public_team; recollect with current collector")
        outcomes = row.get("outcomes_by_opponent") or {}
        if set(outcomes) != set(responses):
            raise ValueError("every action needs paired outcomes for all response classes")
        if row.get("policy_log_probability") is None:
            raise ValueError("every candidate must have a behavior-policy log probability")
        key = (int(row["team_cluster"]), int(row["state_repeat"]),
               tuple(int(v) for v in row["action"]))
        prior = dedup.get(key)
        if prior is None:
            dedup[key] = {**row, "candidate_names": [str(row["candidate"])]}
        else:
            for field in ("observation", "legal_action_mask", "player_team",
                          "opponent_public_team", "native_action_features",
                          "public_context_features", "policy_log_probability"):
                if prior.get(field) != row.get(field):
                    raise ValueError(f"duplicate action has inconsistent {field}: {key}")
            if prior["outcomes_by_opponent"] != outcomes:
                raise ValueError(f"duplicate action has inconsistent outcomes: {key}")
            prior["candidate_names"].append(str(row["candidate"]))

    states: dict[tuple[int, int], dict] = {}
    for row in dedup.values():
        key = (int(row["team_cluster"]), int(row["state_repeat"]))
        state = states.setdefault(key, {
            "team_cluster": key[0], "state_repeat": key[1],
            "observation": row["observation"],
            "legal_action_mask": row["legal_action_mask"],
            "player_team": row["player_team"],
            "opponent_public_team": row["opponent_public_team"],
            "native_action_features": row["native_action_features"],
            "public_context_features": row["public_context_features"],
            "actions": [],
        })
        for field in ("observation", "legal_action_mask", "player_team",
                      "opponent_public_team", "native_action_features",
                      "public_context_features"):
            if state[field] != row.get(field):
                raise ValueError(f"candidate actions in state {key} disagree on {field}")
        state["actions"].append({
            "action": [int(v) for v in row["action"]],
            "names": row["candidate_names"],
            "old_logp": float(row["policy_log_probability"]),
            "outcomes": {
                name: [int(v) for v in row["outcomes_by_opponent"][name]]
                for name in responses
            },
        })
    result = list(states.values())
    for state in result:
        state["actions"].sort(key=lambda x: tuple(x["action"]))
        base = [a for a in state["actions"] if "policy" in a["names"]]
        if len(base) != 1:
            raise ValueError(f"state {state['team_cluster']}:{state['state_repeat']} needs one policy action")
        state["base_action"] = tuple(base[0]["action"])
        counts = {len(values) for action in state["actions"] for values in action["outcomes"].values()}
        if len(counts) != 1 or next(iter(counts)) < 2:
            raise ValueError("paired outcomes need an equal count of at least two chance samples")
    return result, responses


def _target_distribution(state: dict, responses: list[str], chance_slice: slice,
                         advantage_margin: float, temperature: float) -> np.ndarray:
    base_action = state["base_action"]
    base = next(row for row in state["actions"] if tuple(row["action"]) == base_action)
    baseline_outcomes = base["outcomes"]
    robust_advantages = []
    for row in state["actions"]:
        per_response = []
        for name in responses:
            candidate = np.asarray(row["outcomes"][name][chance_slice], dtype=np.float64)
            baseline = np.asarray(baseline_outcomes[name][chance_slice], dtype=np.float64)
            per_response.append(float(np.mean(candidate - baseline)))
        advantage = min(per_response)
        robust_advantages.append(max(0.0, advantage - advantage_margin))
    scores = np.asarray(robust_advantages, dtype=np.float64) / max(temperature, 1e-6)
    scores -= scores.max(initial=0.0)
    weights = np.exp(np.clip(scores, -60.0, 0.0))
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("invalid robust action target distribution")
    target = weights / weights.sum()
    base_index = next(i for i, row in enumerate(state["actions"])
                      if tuple(row["action"]) == base_action)
    if not any(robust_advantages):
        target.fill(0.0)
        target[base_index] = 1.0
    return target.astype(np.float32)


def _regret_matching_distribution(state: dict, responses: list[str],
                                  chance_slice: slice,
                                  iterations: int) -> np.ndarray:
    """Return the row strategy for a candidate-action vs response-class game.

    Columns are the observed opponent response policies, not all legal
    opponent actions; this is an empirical restricted game, not full VGC Nash.
    """
    payoffs = np.asarray([
        [float(np.mean(action["outcomes"][name][chance_slice]))
         for name in responses]
        for action in state["actions"]
    ], dtype=np.float64)
    row_count, col_count = payoffs.shape
    row_regret = np.zeros(row_count, dtype=np.float64)
    col_regret = np.zeros(col_count, dtype=np.float64)
    row_sum = np.zeros(row_count, dtype=np.float64)
    for _ in range(iterations):
        positive = np.maximum(row_regret, 0.0)
        row = (positive / positive.sum() if positive.sum()
               else np.full(row_count, 1.0 / row_count))
        positive = np.maximum(col_regret, 0.0)
        col = (positive / positive.sum() if positive.sum()
               else np.full(col_count, 1.0 / col_count))
        value = float(row @ payoffs @ col)
        row_regret += payoffs @ col - value
        col_regret += value - row @ payoffs
        row_sum += row
    return (row_sum / iterations).astype(np.float32)


def _policy_inputs(state: dict, token_vocab_version: int,
                   native_feature_dim: int, public_context_dim: int) -> dict[str, torch.Tensor]:
    inputs = build_model_inputs(
        player_team=state["player_team"],
        opponent_team=state["opponent_public_team"],
        state=state["observation"],
        legal_actions_mask=state["legal_action_mask"],
        token_vocab_version=token_vocab_version,
    )
    accepted = {
        "pokemon_ids", "move_ids", "item_ids", "ability_ids", "team_mask",
        "field_features", "hp_fractions", "status", "history_observations",
        "belief_features", "native_action_features", "opponent_pokemon_ids",
        "opponent_move_ids", "opponent_item_ids", "opponent_ability_ids",
        "opponent_team_mask", "opponent_hp_fractions", "opponent_status",
        "opponent_recent_move_ids", "previous_action_indices",
    }
    inputs = {key: value for key, value in inputs.items() if key in accepted}
    if native_feature_dim:
        features = state["native_action_features"]
        if features is None:
            raise ValueError("model requires native action features but a state has none")
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.shape != (2, 126, native_feature_dim):
            raise ValueError(f"invalid native action feature shape: {tuple(tensor.shape)}")
        inputs["native_action_features"] = tensor
    if public_context_dim:
        context = state["public_context_features"]
        if context is None or len(context) != public_context_dim:
            raise ValueError("state has missing or incompatible public context features")
        inputs["public_context_features"] = torch.as_tensor(context, dtype=torch.float32)
    return inputs


def _candidate_log_probs(model: TransformerPolicy, output, state: dict,
                         device: torch.device) -> torch.Tensor:
    actions = [row["action"] for row in state["actions"]]
    width = len(state["legal_action_mask"]) // 2
    legal = np.asarray(state["legal_action_mask"], dtype=np.int8)
    masks = []
    for slot in range(2):
        padded = np.zeros(126, dtype=bool)
        padded[:width] = legal[slot * width:(slot + 1) * width].astype(bool)
        masks.append(torch.as_tensor(padded, dtype=torch.bool, device=device))
    first_logits = output.native_action_logits[0, 0]
    first_log_probs = torch.log_softmax(
        first_logits.masked_fill(~masks[0], -torch.inf), dim=-1
    )
    second_base = output.native_action_logits[:, 1]
    result = []
    conditioned: dict[int, torch.Tensor] = {}
    for first_action, second_action in actions:
        first_action = int(first_action)
        if first_action not in conditioned:
            second_mask = masks[1].clone()
            if first_action == 0:
                second_mask[0] = False
            elif 1 <= first_action <= 6:
                second_mask[first_action] = False
            if not second_mask.any():
                second_mask = masks[1]
            previous = torch.tensor([first_action], dtype=torch.long, device=device)
            logits = model.condition_native_second_action_logits(
                second_base, output.state_context, previous
            )[0]
            conditioned[first_action] = torch.log_softmax(
                logits.masked_fill(~second_mask, -torch.inf), dim=-1
            )
        result.append(first_log_probs[first_action] + conditioned[first_action][int(second_action)])
    return torch.stack(result)


def _state_loss(model: TransformerPolicy, batch_states: list[dict],
                targets: list[np.ndarray], inputs: list[dict[str, torch.Tensor]],
                device: torch.device, kl_coefficient: float) -> torch.Tensor:
    batched = {
        key: torch.stack([item[key] for item in inputs]).to(device)
        for key in inputs[0]
    }
    outputs = model(**batched)
    losses = []
    for index, (state, target, item) in enumerate(zip(batch_states, targets, inputs)):
        output = type(outputs)(**{
            field: (getattr(outputs, field)[index:index + 1]
                    if isinstance(getattr(outputs, field), torch.Tensor) else None)
            for field in outputs.__dataclass_fields__
        })
        new_logp = _candidate_log_probs(model, output, state, device)
        target_tensor = torch.as_tensor(target, dtype=new_logp.dtype, device=device)
        old_logp = torch.as_tensor(
            [action["old_logp"] for action in state["actions"]],
            dtype=new_logp.dtype, device=device,
        )
        old_prob = torch.softmax(old_logp, dim=0)
        new_logp_normalized = torch.log_softmax(new_logp, dim=0)
        old_logp_normalized = torch.log_softmax(old_logp, dim=0)
        cross_entropy = -(target_tensor * new_logp_normalized).sum()
        kl_anchor = (old_prob * (old_logp_normalized - new_logp_normalized)).sum()
        losses.append(cross_entropy + kl_coefficient * kl_anchor)
    return torch.stack(losses).mean()


def _candidate_test_delta(states: list[dict], targets_by_state: dict[tuple[int, int], np.ndarray],
                          action_scores: dict[tuple[int, int], np.ndarray],
                          responses: list[str], chance_slices: dict[tuple[int, int], slice]) -> dict:
    by_team: dict[int, list[float]] = {}
    mixed_by_team: dict[int, list[float]] = {}
    mixed_by_response: dict[str, list[float]] = {name: [] for name in responses}
    picked_counts: dict[str, int] = {}
    for state in states:
        key = (state["team_cluster"], state["state_repeat"])
        chance_slice = chance_slices[key]
        scores = action_scores[key]
        picked_index = int(np.argmax(scores))
        picked = state["actions"][picked_index]
        base = next(row for row in state["actions"] if tuple(row["action"]) == state["base_action"])
        deltas = [float(np.mean(np.asarray(picked["outcomes"][name][chance_slice], dtype=float)
                                - np.asarray(base["outcomes"][name][chance_slice], dtype=float)))
                  for name in responses]
        by_team.setdefault(state["team_cluster"], []).append(min(deltas))
        # The training target is a mixed strategy. Also report its actor
        # distribution over the sampled candidate set; argmax-only screening
        # would otherwise erase the very randomization the target teaches.
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        candidate_deltas = np.asarray([
            [float(np.mean(np.asarray(action["outcomes"][name][chance_slice], dtype=float)
                           - np.asarray(base["outcomes"][name][chance_slice], dtype=float)))
             for name in responses]
            for action in state["actions"]
        ], dtype=np.float64)
        mixed_deltas = probabilities @ candidate_deltas
        for index, name in enumerate(responses):
            mixed_by_response[name].append(float(mixed_deltas[index]))
        mixed_by_team.setdefault(state["team_cluster"], []).append(
            float(np.min(mixed_deltas))
        )
        name = picked["names"][0]
        picked_counts[name] = picked_counts.get(name, 0) + 1
    values = np.asarray([np.mean(v) for v in by_team.values()], dtype=np.float64)
    rng = np.random.default_rng(411209)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    mixed_values = np.asarray([np.mean(v) for v in mixed_by_team.values()], dtype=np.float64)
    mixed_samples = mixed_values[
        rng.integers(0, len(mixed_values), (10000, len(mixed_values)))
    ].mean(axis=1)
    return {
        "team_clusters": len(values),
        "selected_minus_policy_worst_response": float(values.mean()),
        "cluster_bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "picked_candidate_counts": picked_counts,
        "candidate_restricted_mixed_distribution": {
            "scope": "actor probabilities renormalized over only the collected candidate actions",
            "mean_delta_by_response": {
                name: float(np.mean(mixed_by_response[name])) for name in responses
            },
            "mean_statewise_worst_response_delta": float(mixed_values.mean()),
            "cluster_bootstrap_95_ci": np.quantile(mixed_samples, [0.025, 0.975]).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20291101)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--advantage-temperature", type=float, default=0.15)
    parser.add_argument(
        "--target-objective",
        choices=("positive_worst_advantage", "regret_matching"),
        default="positive_worst_advantage",
        help="distill sparse robust advantages or a mixed strategy from the empirical payoff matrix",
    )
    parser.add_argument("--regret-matching-iterations", type=int, default=5000)
    parser.add_argument("--kl-coefficient", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.regret_matching_iterations < 1):
        parser.error("epochs, batch size, and learning rate must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    states, responses = _load_rows(args.data)
    chance_slices: dict[tuple[int, int], tuple[slice, slice]] = {}
    for state in states:
        key = (state["team_cluster"], state["state_repeat"])
        count = len(state["actions"][0]["outcomes"][responses[0]])
        midpoint = count // 2
        if midpoint < 1 or count - midpoint < 1:
            raise ValueError(f"state {key} needs disjoint training and test chance repeats")
        chance_slices[key] = (slice(0, midpoint), slice(midpoint, count))
    chance_split_counts: dict[str, dict[str, int]] = {}
    for train_slice, test_slice in chance_slices.values():
        label = str(train_slice.stop + test_slice.stop - test_slice.start)
        group = chance_split_counts.setdefault(label, {"states": 0, "train": train_slice.stop,
                                                       "test": test_slice.stop - test_slice.start})
        group["states"] += 1
    clusters = np.asarray(sorted({state["team_cluster"] for state in states}))
    shuffled = rng.permutation(clusters)
    test_count = max(1, round(len(shuffled) * args.test_fraction))
    test_teams = set(map(int, shuffled[:test_count]))
    remaining = rng.permutation(shuffled[test_count:])
    valid_count = max(1, round(len(remaining) * 0.20))
    valid_teams = set(map(int, remaining[:valid_count]))
    train_teams = set(map(int, remaining[valid_count:]))
    split = {
        "train": [s for s in states if s["team_cluster"] in train_teams],
        "validation": [s for s in states if s["team_cluster"] in valid_teams],
        "test": [s for s in states if s["team_cluster"] in test_teams],
    }

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TransformerPolicy.from_checkpoint_state(source)
    model.load_state_dict(source["model_state_dict"])
    model.to(device)
    model.eval()
    native_dim = int(model.native_action_feature_dim)
    public_dim = int(model.public_context_dim)
    input_cache = {
        (s["team_cluster"], s["state_repeat"]): _policy_inputs(
            s, model.token_vocab_version, native_dim, public_dim
        )
        for s in states
    }
    if args.target_objective == "regret_matching":
        targets = {
            (s["team_cluster"], s["state_repeat"]): _regret_matching_distribution(
                s, responses, chance_slices[(s["team_cluster"], s["state_repeat"])][0],
                args.regret_matching_iterations,
            )
            for s in states
        }
    else:
        targets = {
            (s["team_cluster"], s["state_repeat"]): _target_distribution(
                s, responses, chance_slices[(s["team_cluster"], s["state_repeat"])][0],
                args.advantage_margin,
                args.advantage_temperature,
            )
            for s in states
        }

    def evaluate_objective(split_name: str) -> float:
        selected = split[split_name]
        if not selected:
            return math.inf
        losses = []
        with torch.no_grad():
            for start in range(0, len(selected), args.batch_size):
                batch = selected[start:start + args.batch_size]
                loss = _state_loss(
                    model, batch,
                    [targets[(s["team_cluster"], s["state_repeat"])] for s in batch],
                    [input_cache[(s["team_cluster"], s["state_repeat"])] for s in batch],
                    device, args.kl_coefficient,
                )
                losses.append(float(loss.cpu()))
        return float(np.mean(losses))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_score = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    epochs = []
    train = split["train"]
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        model.eval()  # deterministic encoding with gradients enabled
        train_losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [train[int(i)] for i in order[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = _state_loss(
                model, batch,
                [targets[(s["team_cluster"], s["state_repeat"])] for s in batch],
                [input_cache[(s["team_cluster"], s["state_repeat"])] for s in batch],
                device, args.kl_coefficient,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = evaluate_objective("validation")
        epochs.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                       "validation_loss": validation_loss})
        print(json.dumps(epochs[-1]), flush=True)
        if validation_loss < best_score - 1e-5:
            best_score = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)

    # Evaluate candidate-set actor choices on untouched chance samples. This
    # is a screening metric; the full-match paired evaluation remains the gate.
    test_states = split["test"]
    action_scores: dict[tuple[int, int], np.ndarray] = {}
    with torch.no_grad():
        for state in test_states:
            key = (state["team_cluster"], state["state_repeat"])
            item = {name: tensor.unsqueeze(0).to(device)
                    for name, tensor in input_cache[key].items()}
            output = model(**item)
            action_scores[key] = _candidate_log_probs(model, output, state, device).cpu().numpy()
    test_selection = _candidate_test_delta(
        test_states, targets, action_scores, responses,
        {key: chance_slices[key][1] for key in action_scores}
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "counterfactual_policy.pt"
    source["model_state_dict"] = {
        key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
    }
    source["counterfactual_policy_update"] = {
        "data": str(args.data), "seed": args.seed, "best_epoch": best_epoch,
        "objective": args.target_objective + " with policy KL anchor",
        "regret_matching_iterations": args.regret_matching_iterations,
        "responses": responses, "team_split": {k: len({s['team_cluster'] for s in v})
                                                for k, v in split.items()},
        "chance_split_by_repeat_count": chance_split_counts,
        "validation_loss": best_score,
        "test_candidate_set_selection": test_selection,
    }
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(source, temporary)
        temporary.replace(target_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "checkpoint": str(target_path), "data": str(args.data),
        "source_checkpoint": str(args.checkpoint), "responses": responses,
        "team_split": {k: len({s["team_cluster"] for s in v}) for k, v in split.items()},
        "state_split": {k: len(v) for k, v in split.items()},
        "chance_split_by_repeat_count": chance_split_counts,
        "target": {"objective": args.target_objective,
                   "regret_matching_iterations": args.regret_matching_iterations,
                   "advantage_margin": args.advantage_margin,
                   "temperature": args.advantage_temperature,
                   "kl_coefficient": args.kl_coefficient},
        "best_epoch": best_epoch, "best_validation_loss": best_score,
        "epochs": epochs, "test_candidate_set_selection": test_selection,
        "information_boundary": "own full team and opponent preview species only",
        "checkpoint_role": "experimental candidate; promote only after fresh paired full-match gain",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
