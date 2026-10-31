#!/usr/bin/env python3
"""Fit/evaluate an experimental action-conditioned win-probability model.

Inputs are only the acting player's observed state and its candidate joint
action. Opponent identity is withheld from the model and used only to stratify
the held-out report.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn


class ActionWinModel(nn.Module):
    def __init__(self, observation_dim: int, action_width: int,
                 extra_feature_dim: int = 0, response_count: int = 1) -> None:
        super().__init__()
        if response_count < 1:
            raise ValueError("response_count must be positive")
        self.observation_dim = observation_dim
        self.action_width = action_width
        self.extra_feature_dim = extra_feature_dim
        self.response_count = response_count
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 2 * action_width + extra_feature_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, response_count),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return output.squeeze(-1) if self.response_count == 1 else output


def _inputs(observation: list[float], action: list[int], action_width: int,
            native_action_features=None, public_context_features=None,
            policy_log_probability=None, include_policy_log_probability: bool = False) -> np.ndarray:
    features = np.zeros(2 * action_width, dtype=np.float32)
    for slot, raw_action in enumerate(action):
        index = int(raw_action)
        if not 0 <= index < action_width:
            raise ValueError(f"action index {index} is outside width {action_width}")
        features[slot * action_width + index] = 1.0
    extras = []
    if native_action_features is not None:
        table = np.asarray(native_action_features, dtype=np.float32)
        if table.ndim != 3 or table.shape[0] != 2:
            raise ValueError("native action features must have shape (2, action_width, feature_dim)")
        for slot, raw_action in enumerate(action):
            extras.extend(table[slot, int(raw_action)].tolist())
    if public_context_features is not None:
        extras.extend(np.asarray(public_context_features, dtype=np.float32).reshape(-1).tolist())
    if policy_log_probability is not None:
        extras.extend([float(np.clip(policy_log_probability, -30.0, 0.0)) / 30.0, 1.0])
    elif include_policy_log_probability:
        extras.extend([-1.0, 0.0])
    return np.concatenate((np.asarray(observation, dtype=np.float32), features,
                           np.asarray(extras, dtype=np.float32)))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p))))


def _calibration(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(probabilities)
    result = []
    for chunk in np.array_split(order, min(bins, len(order))):
        if len(chunk):
            result.append({
                "count": int(len(chunk)),
                "mean_probability": float(probabilities[chunk].mean()),
                "observed_win_rate": float(labels[chunk].mean()),
            })
    return result


def _cluster_bootstrap(values: dict[int, float], seed: int, rounds: int = 20000) -> list[float]:
    rows = np.asarray(list(values.values()), dtype=np.float64)
    if rows.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = rows[rng.integers(0, rows.size, (rounds, rows.size))].mean(axis=1)
    return np.quantile(sample, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102805)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--split-chance-repeats", action="store_true",
        help="train on the first half of each action's common chance seeds and evaluate selection on the second half",
    )
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.4:
        parser.error("test-fraction must be between 0.05 and 0.4")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    payload = json.loads(args.data.read_text(encoding="utf-8"))
    raw_rows = payload["rows"]
    deduplicated: dict[tuple[int, int, tuple[int, int]], dict] = {}
    for row in raw_rows:
        key = (int(row["team_cluster"]), int(row["state_repeat"]),
               (int(row["action"][0]), int(row["action"][1])))
        prior = deduplicated.get(key)
        outcomes = row.get("outcomes", [row.get("outcome")])
        outcomes = [int(value) for value in outcomes]
        if prior is None:
            deduplicated[key] = {
                **row,
                "outcomes": outcomes,
                "candidate_names": {str(row["candidate"])},
            }
        else:
            if (prior["outcomes"] != outcomes
                    or prior["observation"] != row["observation"]):
                raise ValueError(f"duplicate action has inconsistent labels: {key}")
            prior["candidate_names"].add(str(row["candidate"]))
    rows = list(deduplicated.values())
    training_outcomes_by_row: list[list[int]] = []
    evaluation_outcomes_by_row: list[list[int]] = []
    for row in rows:
        outcomes = [int(value) for value in row["outcomes"]]
        if args.split_chance_repeats:
            midpoint = len(outcomes) // 2
            if midpoint < 1 or len(outcomes) - midpoint < 1:
                parser.error("split-chance-repeats needs at least two outcomes per action")
            training_outcomes_by_row.append(outcomes[:midpoint])
            evaluation_outcomes_by_row.append(outcomes[midpoint:])
        else:
            training_outcomes_by_row.append(outcomes)
            evaluation_outcomes_by_row.append(outcomes)
    action_width = len(rows[0]["legal_action_mask"]) // 2
    observation_dim = len(rows[0]["observation"])
    feature_rows = [row for row in rows if row.get("native_action_features") is not None
                    and row.get("public_context_features") is not None]
    if feature_rows and len(feature_rows) != len(rows):
        raise ValueError("training data mixes rows with and without action/context features")
    action_feature_dim = (len(feature_rows[0]["native_action_features"][0][0]) if feature_rows else 0)
    context_dim = (len(feature_rows[0]["public_context_features"]) if feature_rows else 0)
    use_policy_logp = bool(feature_rows) and all(
        row.get("policy_log_probability") is not None for row in feature_rows
    )
    extra_feature_dim = 2 * action_feature_dim + context_dim + (2 if use_policy_logp else 0)
    for row in rows:
        if len(row["legal_action_mask"]) != 2 * action_width:
            raise ValueError("all native action masks must have the same width")
        for slot, action in enumerate(row["action"]):
            if row["legal_action_mask"][slot * action_width + int(action)] != 1:
                raise ValueError(f"illegal recorded action in cluster {row['team_cluster']}")

    cluster_ids = np.asarray(sorted({int(row["team_cluster"]) for row in rows}))
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(cluster_ids)
    test_count = max(1, round(len(shuffled) * args.test_fraction))
    test_clusters = set(int(x) for x in shuffled[:test_count])
    train_pool_clusters = set(int(x) for x in shuffled[test_count:])
    train_pool_shuffled = rng.permutation(sorted(train_pool_clusters))
    validation_count = max(1, round(len(train_pool_shuffled) * 0.20))
    validation_clusters = set(int(x) for x in train_pool_shuffled[:validation_count])
    train_clusters = train_pool_clusters - validation_clusters

    sample_to_row = np.asarray([
        row_index
        for row_index, outcomes in enumerate(training_outcomes_by_row)
        for _ in outcomes
    ], dtype=np.int64)
    x = np.stack([
        _inputs(rows[row_index]["observation"], rows[row_index]["action"], action_width,
                rows[row_index].get("native_action_features"),
                rows[row_index].get("public_context_features"),
                rows[row_index].get("policy_log_probability") if use_policy_logp else None,
                use_policy_logp)
        for row_index in sample_to_row
    ])
    y = np.asarray([
        float(outcome)
        for outcomes in training_outcomes_by_row
        for outcome in outcomes
    ], dtype=np.float32)
    cluster_array = np.asarray([
        int(rows[row_index]["team_cluster"]) for row_index in sample_to_row
    ])
    train_index = np.flatnonzero(np.isin(cluster_array, list(train_clusters)))
    validation_index = np.flatnonzero(np.isin(cluster_array, list(validation_clusters)))
    test_index = np.flatnonzero(np.isin(cluster_array, list(test_clusters)))

    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    model = ActionWinModel(observation_dim, action_width, extra_feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    best_validation_brier = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(train_index)
        for start in range(0, len(permutation), args.batch_size):
            batch = torch.as_tensor(permutation[start:start + args.batch_size], device=device)
            logits = model(x_tensor[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(x_tensor[validation_index])).cpu().numpy()
        val_brier = _brier(val_p, y[validation_index])
        epoch_log.append({"epoch": epoch, "validation_brier": val_brier})
        if val_brier < best_validation_brier - 1e-5:
            best_validation_brier = val_brier
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no best action-value model checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_p = torch.sigmoid(model(x_tensor[test_index])).cpu().numpy()
    train_mean = float(y[train_index].mean())
    test_probability_by_row: dict[int, list[float]] = {}
    for sample_index, probability in zip(test_index.tolist(), test_p.tolist()):
        row_index = int(sample_to_row[sample_index])
        test_probability_by_row.setdefault(row_index, []).append(float(probability))

    # Apply the frozen model to each held-out state's candidate set. Outcomes
    # are from separate complete-match continuations and were not used to fit.
    state_candidates: dict[tuple[int, int], list[tuple[int, dict, float]]] = {}
    test_unique_rows = sorted(test_probability_by_row)
    for row_index in test_unique_rows:
        row = rows[row_index]
        key = (int(row["team_cluster"]), int(row["state_repeat"]))
        state_candidates.setdefault(key, []).append((
            row_index, row, float(np.mean(test_probability_by_row[row_index]))
        ))
    test_probabilities = np.asarray([
        probability
        for row_index in test_unique_rows
        for probability in [float(np.mean(test_probability_by_row[row_index]))]
        for _ in evaluation_outcomes_by_row[row_index]
    ], dtype=np.float32)
    test_y = np.asarray([
        float(outcome)
        for row_index in test_unique_rows
        for outcome in evaluation_outcomes_by_row[row_index]
    ], dtype=np.float32)
    test_baseline_p = np.full_like(test_y, train_mean, dtype=np.float32)
    policy_by_cluster: dict[int, list[float]] = {}
    selected_by_cluster: dict[int, list[float]] = {}
    selected_vs_policy_by_cluster: dict[int, list[float]] = {}
    per_opponent: dict[str, dict[int, list[tuple[float, float]]]] = {}
    chosen_actions = []
    for (cluster, state_repeat), candidates in state_candidates.items():
        policy_rows = [item for item in candidates if "policy" in item[1]["candidate_names"]]
        if not policy_rows:
            continue
        policy_index, policy_row, _ = policy_rows[0]
        selected_index, selected_row, selected_probability = max(candidates, key=lambda item: item[2])
        p_outcome = float(np.mean(evaluation_outcomes_by_row[policy_index]))
        selected_outcome = float(np.mean(evaluation_outcomes_by_row[selected_index]))
        policy_by_cluster.setdefault(cluster, []).append(p_outcome)
        selected_by_cluster.setdefault(cluster, []).append(selected_outcome)
        selected_vs_policy_by_cluster.setdefault(cluster, []).append(selected_outcome - p_outcome)
        opp = str(policy_row["opponent_strategy"])
        per_opponent.setdefault(opp, {}).setdefault(cluster, []).append(
            (selected_outcome, p_outcome)
        )
        chosen_actions.append({
            "team_cluster": cluster,
            "state_repeat": state_repeat,
            "opponent_strategy": opp,
            "action": selected_row["action"],
            "candidate_names": sorted(selected_row["candidate_names"]),
            "predicted_win_probability": selected_probability,
            "observed_outcome": selected_outcome,
            "policy_action": policy_row["action"],
            "policy_outcome": float(np.mean(policy_row["outcomes"])),
        })

    def reduce_clusters(values: dict[int, list[float]]) -> dict[int, float]:
        return {cluster: float(np.mean(items)) for cluster, items in values.items()}

    selected_cluster_rate = reduce_clusters(selected_by_cluster)
    policy_cluster_rate = reduce_clusters(policy_by_cluster)
    delta_cluster_rate = reduce_clusters(selected_vs_policy_by_cluster)
    selection = {
        "heldout_states": len(chosen_actions),
        "heldout_team_clusters": len(selected_cluster_rate),
        "policy_win_rate": float(np.mean(list(policy_cluster_rate.values()))) if policy_cluster_rate else None,
        "selected_win_rate": float(np.mean(list(selected_cluster_rate.values()))) if selected_cluster_rate else None,
        "selected_minus_policy": float(np.mean(list(delta_cluster_rate.values()))) if delta_cluster_rate else None,
        "selected_minus_policy_cluster_bootstrap_95_ci": _cluster_bootstrap(
            delta_cluster_rate, args.seed + 11
        ) if delta_cluster_rate else None,
    }
    selection_by_opponent = {}
    for opponent, clusters in per_opponent.items():
        delta = {cluster: float(np.mean([pair[0] - pair[1] for pair in pairs]))
                 for cluster, pairs in clusters.items()}
        selection_by_opponent[opponent] = {
            "team_clusters": len(delta),
            "selected_minus_policy": float(np.mean(list(delta.values()))),
            "paired_cluster_bootstrap_95_ci": _cluster_bootstrap(delta, args.seed + 13 + len(delta)),
        }

    metrics = {
        "unique_state_action_rows": len(rows),
        "binary_chance_outcome_samples": int(y.size),
        "chance_outcome_split": (
            "first_half_train_second_half_evaluation"
            if args.split_chance_repeats else "all_outcomes"
        ),
        "team_clusters": int(cluster_ids.size),
        "train_clusters": len(train_clusters),
        "validation_clusters": len(validation_clusters),
        "test_clusters": len(test_clusters),
        "observation_dim": observation_dim,
        "action_width": action_width,
        "action_feature_dim": action_feature_dim,
        "context_dim": context_dim,
        "use_policy_log_probability": use_policy_logp,
        "extra_feature_dim": extra_feature_dim,
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_log),
        "train_outcome_mean": train_mean,
        "test_action_outcome_brier": _brier(test_probabilities, test_y),
        "test_action_outcome_log_loss": _log_loss(test_probabilities, test_y),
        "constant_train_rate_brier": _brier(test_baseline_p, test_y),
        "calibration_bins": _calibration(test_probabilities, test_y),
        "heldout_candidate_selection": selection,
        "selection_by_opponent": selection_by_opponent,
    }
    checkpoint = {
        "schema_version": 1,
        "model_type": "action_conditioned_win_probability",
        "target_description": "P1 win after this legal action and frozen policy continuation against the collected opponent mixture",
        "opponent_identity_is_model_input": False,
        "observation_dim": observation_dim,
        "action_width": action_width,
        "action_feature_dim": action_feature_dim,
        "context_dim": context_dim,
        "use_policy_log_probability": use_policy_logp,
        "extra_feature_dim": extra_feature_dim,
        "state_dict": best_state,
        "metrics": metrics,
        "training_data": str(args.data),
        "seed": args.seed,
    }
    target_path = args.output_dir / "action_win_model.pt"
    with tempfile.NamedTemporaryFile(dir=args.output_dir, suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        torch.save(checkpoint, temp_path)
        temp_path.replace(target_path)
    finally:
        temp_path.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "data": str(args.data),
        "checkpoint": str(target_path),
        "split_seed": args.seed,
        "test_fraction": args.test_fraction,
        "metrics": metrics,
        "epoch_log": epoch_log,
        "selected_actions": chosen_actions,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(target_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
