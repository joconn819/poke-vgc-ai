#!/usr/bin/env python3
"""Train a closed-sheet action value model with a worst-response selector.

Each candidate action has matched full-match outcomes against every named
response policy from the same captured state. The model predicts each response
head from player-visible state/action features; deployment chooses the action
with the best minimum predicted win probability. Opponent response identity is
not an input to the model or the deployed choice.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_action_conditioned_win_model import (
    ActionWinModel,
    _inputs,
)


def _cluster_interval(values: dict[int, float], seed: int,
                      rounds: int = 20000) -> list[float] | None:
    rows = np.asarray(list(values.values()), dtype=np.float64)
    if rows.size == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rows[rng.integers(0, rows.size, (rounds, rows.size))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def _binary_metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    if outcomes.size == 0:
        return {"samples": 0, "brier": None, "log_loss": None}
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "samples": int(outcomes.size),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
        "log_loss": float(np.mean(-(
            outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)
        ))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20282021)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.4:
        parser.error("test-fraction must be between 0.05 and 0.4")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", [])
    if not raw_rows:
        raise ValueError("counterfactual dataset contains no rows")

    response_names = sorted(raw_rows[0].get("outcomes_by_opponent", {}))
    if len(response_names) < 2:
        raise ValueError("data must include at least two paired opponent response classes")

    # Collapse semantically duplicate action rows while requiring their
    # player-view state and response-specific chance outcomes to agree.
    deduplicated: dict[tuple[int, int, tuple[int, int]], dict] = {}
    for row in raw_rows:
        outcomes_by_response = row.get("outcomes_by_opponent")
        if not outcomes_by_response or set(outcomes_by_response) != set(response_names):
            raise ValueError("every candidate must have all paired response outcomes")
        if row.get("native_action_features") is None or row.get("public_context_features") is None:
            raise ValueError("response-conditioned training requires action and public-context features")
        key = (int(row["team_cluster"]), int(row["state_repeat"]),
               (int(row["action"][0]), int(row["action"][1])))
        normalized = {
            name: [int(value) for value in outcomes_by_response[name]]
            for name in response_names
        }
        prior = deduplicated.get(key)
        if prior is None:
            deduplicated[key] = {
                **row,
                "outcomes_by_opponent": normalized,
                "candidate_names": {str(row["candidate"])},
            }
        else:
            for field in ("observation", "legal_action_mask", "native_action_features",
                          "public_context_features"):
                if prior[field] != row[field]:
                    raise ValueError(f"duplicate action has inconsistent {field}: {key}")
            if prior["outcomes_by_opponent"] != normalized:
                raise ValueError(f"duplicate action has inconsistent outcomes: {key}")
            prior["candidate_names"].add(str(row["candidate"]))
    rows = list(deduplicated.values())

    repeats = {
        len(values)
        for row in rows
        for values in row["outcomes_by_opponent"].values()
    }
    if len(repeats) != 1 or next(iter(repeats)) < 2:
        raise ValueError("each candidate/response needs the same number of at least two chance outcomes")
    chance_count = next(iter(repeats))
    midpoint = chance_count // 2
    if midpoint < 1 or chance_count - midpoint < 1:
        raise ValueError("need both training and evaluation chance repeats")

    team_clusters = np.asarray(sorted({int(row["team_cluster"]) for row in rows}))
    shuffled = rng.permutation(team_clusters)
    test_count = max(1, round(len(shuffled) * args.test_fraction))
    test_teams = set(map(int, shuffled[:test_count]))
    remaining_teams = rng.permutation(shuffled[test_count:])
    validation_count = max(1, round(len(remaining_teams) * 0.20))
    validation_teams = set(map(int, remaining_teams[:validation_count]))
    train_teams = set(map(int, remaining_teams[validation_count:]))

    action_width = len(rows[0]["legal_action_mask"]) // 2
    observation_dim = len(rows[0]["observation"])
    action_feature_dim = len(rows[0]["native_action_features"][0][0])
    context_dim = len(rows[0]["public_context_features"])
    use_policy_logp = all(row.get("policy_log_probability") is not None for row in rows)
    extra_feature_dim = 2 * action_feature_dim + context_dim + (2 if use_policy_logp else 0)
    for row in rows:
        if len(row["legal_action_mask"]) != 2 * action_width:
            raise ValueError("inconsistent action mask width")
        for slot, action in enumerate(row["action"]):
            if row["legal_action_mask"][slot * action_width + int(action)] != 1:
                raise ValueError(f"illegal recorded action in team cluster {row['team_cluster']}")

    features = np.stack([
        _inputs(row["observation"], row["action"], action_width,
                row["native_action_features"], row["public_context_features"],
                row.get("policy_log_probability") if use_policy_logp else None,
                use_policy_logp)
        for row in rows
    ])
    labels_by_response = {
        name: np.asarray([
            [float(value) for value in row["outcomes_by_opponent"][name]]
            for row in rows
        ], dtype=np.float32)
        for name in response_names
    }
    row_teams = np.asarray([int(row["team_cluster"]) for row in rows])
    train_row_mask = np.isin(row_teams, list(train_teams))
    validation_row_mask = np.isin(row_teams, list(validation_teams))
    test_row_mask = np.isin(row_teams, list(test_teams))

    # Use response-major ordering consistently in train, validation, and test.
    train_y = np.concatenate([
        labels_by_response[name][train_row_mask, :midpoint].reshape(-1)
        for name in response_names
    ]).astype(np.float32)
    train_x = np.concatenate([
        np.repeat(features[train_row_mask], midpoint, axis=0)
        for _ in response_names
    ], axis=0)
    validation_x = np.concatenate([
        np.repeat(features[validation_row_mask], midpoint, axis=0)
        for _ in response_names
    ], axis=0)
    validation_y = np.concatenate([
        labels_by_response[name][validation_row_mask, :midpoint].reshape(-1)
        for name in response_names
    ]).astype(np.float32)
    # Expand response-major examples so each training row supervises only its
    # matching output head.
    response_ids = np.concatenate([
        np.full(int(train_row_mask.sum()) * midpoint, index, dtype=np.int64)
        for index in range(len(response_names))
    ])
    validation_response_ids = np.concatenate([
        np.full(int(validation_row_mask.sum()) * midpoint, index, dtype=np.int64)
        for index in range(len(response_names))
    ])

    model = ActionWinModel(observation_dim, action_width, extra_feature_dim,
                           response_count=len(response_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    x_tensor = torch.from_numpy(train_x).to(device)
    y_tensor = torch.from_numpy(train_y).to(device)
    response_tensor = torch.from_numpy(response_ids).to(device)
    val_x_tensor = torch.from_numpy(validation_x).to(device)
    val_y = validation_y
    val_response_tensor = torch.from_numpy(validation_response_ids).to(device)
    best_brier = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), args.batch_size):
            batch_indices = torch.as_tensor(order[start:start + args.batch_size], device=device)
            logits_all = model(x_tensor[batch_indices])
            logits = logits_all.gather(1, response_tensor[batch_indices, None]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_tensor[batch_indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_all = model(val_x_tensor)
            val_logits = val_all.gather(1, val_response_tensor[:, None]).squeeze(1)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        validation_brier = float(np.mean((val_prob - val_y) ** 2))
        epoch_log.append({"epoch": epoch, "validation_brier": validation_brier})
        if validation_brier < best_brier - 1e-5:
            best_brier = validation_brier
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no best response-conditioned model was produced")
    model.load_state_dict(best_state)
    model.eval()

    selected_records = []
    equal_weight_records = []
    test_prob_by_response: dict[str, list[float]] = {name: [] for name in response_names}
    test_label_by_response: dict[str, list[float]] = {name: [] for name in response_names}
    state_rows: dict[tuple[int, int], list[int]] = {}
    for row_index, row in enumerate(rows):
        if not test_row_mask[row_index]:
            continue
        state_rows.setdefault((int(row["team_cluster"]), int(row["state_repeat"])), []).append(row_index)
        x = torch.as_tensor(features[row_index:row_index + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x)).cpu().numpy()[0]
        for response_index, name in enumerate(response_names):
            test_prob_by_response[name].extend([float(probabilities[response_index])] * (chance_count - midpoint))
            test_label_by_response[name].extend(
                [float(value) for value in labels_by_response[name][row_index, midpoint:]]
            )

    response_metrics = {}
    baseline_metrics = {}
    for response_index, name in enumerate(response_names):
        probs = np.asarray(test_prob_by_response[name], dtype=np.float32)
        labels = np.asarray(test_label_by_response[name], dtype=np.float32)
        response_metrics[name] = _binary_metrics(probs, labels)
        train_labels = labels_by_response[name][train_row_mask, :midpoint].reshape(-1)
        base = np.full_like(labels, float(train_labels.mean()))
        baseline_metrics[name] = _binary_metrics(base, labels)

    policy_deltas_by_team = {name: {} for name in response_names}
    selected_rates_by_team = {name: {} for name in response_names}
    policy_rates_by_team = {name: {} for name in response_names}
    selected_worst_delta_by_team: dict[int, list[float]] = {}
    policy_worst_by_team: dict[int, list[float]] = {}
    selected_worst_by_team: dict[int, list[float]] = {}
    equal_weight_worst_delta_by_team: dict[int, list[float]] = {}
    equal_weight_response_delta_by_team = {name: {} for name in response_names}
    robust_selected_actions = 0
    equal_weight_selected_actions = 0
    for (team_cluster, state_repeat), indexes in state_rows.items():
        candidates = [rows[index] for index in indexes]
        feature_batch = torch.as_tensor(features[indexes], dtype=torch.float32, device=device)
        with torch.no_grad():
            predictions = torch.sigmoid(model(feature_batch)).cpu().numpy()
        robust_scores = predictions.min(axis=1)
        selected_position = int(np.argmax(robust_scores))
        policy_positions = [
            pos for pos, row in enumerate(candidates)
            if "policy" in row.get("candidate_names", {str(row["candidate"])})
        ]
        if not policy_positions:
            continue
        policy_position = policy_positions[0]
        selected_row = candidates[selected_position]
        policy_row = candidates[policy_position]
        selected_worst = []
        policy_worst = []
        record = {
            "team_cluster": team_cluster,
            "state_repeat": state_repeat,
            "selected_action": selected_row["action"],
            "selected_candidate": sorted(selected_row.get("candidate_names", {selected_row["candidate"]})),
            "policy_action": policy_row["action"],
            "predicted_response_probabilities": {
                name: float(predictions[selected_position, i])
                for i, name in enumerate(response_names)
            },
            "predicted_worst_response_probability": float(robust_scores[selected_position]),
            "observed_by_response": {},
        }
        if selected_position != policy_position:
            robust_selected_actions += 1
        for response_index, name in enumerate(response_names):
            selected_rate = float(np.mean(selected_row["outcomes_by_opponent"][name][midpoint:]))
            policy_rate = float(np.mean(policy_row["outcomes_by_opponent"][name][midpoint:]))
            selected_worst.append(selected_rate)
            policy_worst.append(policy_rate)
            selected_rates_by_team[name].setdefault(team_cluster, []).append(selected_rate)
            policy_rates_by_team[name].setdefault(team_cluster, []).append(policy_rate)
            policy_deltas_by_team[name].setdefault(team_cluster, []).append(selected_rate - policy_rate)
            record["observed_by_response"][name] = {
                "selected_win_rate": selected_rate,
                "policy_win_rate": policy_rate,
                "selected_minus_policy": selected_rate - policy_rate,
            }
        selected_min = min(selected_worst)
        policy_min = min(policy_worst)
        selected_worst_by_team.setdefault(team_cluster, []).append(selected_min)
        policy_worst_by_team.setdefault(team_cluster, []).append(policy_min)
        selected_worst_delta_by_team.setdefault(team_cluster, []).append(selected_min - policy_min)
        record["observed_worst_response"] = {
            "selected_win_rate": selected_min,
            "policy_win_rate": policy_min,
            "selected_minus_policy": selected_min - policy_min,
        }
        selected_records.append(record)

        # Compare a simple equal-weight opponent mixture with maximin using
        # exactly the same held-out states, candidates, and chance outcomes.
        equal_weight_scores = predictions.mean(axis=1)
        equal_position = int(np.argmax(equal_weight_scores))
        equal_selected = candidates[equal_position]
        equal_record = {
            "team_cluster": team_cluster,
            "state_repeat": state_repeat,
            "selected_action": equal_selected["action"],
            "policy_action": policy_row["action"],
            "predicted_response_probabilities": {
                name: float(predictions[equal_position, i])
                for i, name in enumerate(response_names)
            },
            "observed_by_response": {},
        }
        if equal_position != policy_position:
            equal_weight_selected_actions += 1
        equal_selected_outcomes = []
        equal_policy_outcomes = []
        for name in response_names:
            selected_rate = float(np.mean(equal_selected["outcomes_by_opponent"][name][midpoint:]))
            policy_rate = float(np.mean(policy_row["outcomes_by_opponent"][name][midpoint:]))
            equal_selected_outcomes.append(selected_rate)
            equal_policy_outcomes.append(policy_rate)
            equal_weight_response_delta_by_team[name].setdefault(team_cluster, []).append(
                selected_rate - policy_rate
            )
            equal_record["observed_by_response"][name] = {
                "selected_win_rate": selected_rate,
                "policy_win_rate": policy_rate,
                "selected_minus_policy": selected_rate - policy_rate,
            }
        equal_weight_worst_delta_by_team.setdefault(team_cluster, []).append(
            min(equal_selected_outcomes) - min(equal_policy_outcomes)
        )
        equal_record["observed_worst_response"] = {
            "selected_win_rate": min(equal_selected_outcomes),
            "policy_win_rate": min(equal_policy_outcomes),
            "selected_minus_policy": min(equal_selected_outcomes) - min(equal_policy_outcomes),
        }
        equal_record["predicted_mean_response_probability"] = float(
            equal_weight_scores[equal_position]
        )
        equal_weight_records.append(equal_record)

    def reduce_team(values: dict[int, list[float]]) -> dict[int, float]:
        return {team: float(np.mean(team_values)) for team, team_values in values.items()}

    selection_by_response = {}
    for index, name in enumerate(response_names):
        deltas = reduce_team(policy_deltas_by_team[name])
        selection_by_response[name] = {
            "team_clusters": len(deltas),
            "selected_minus_policy": float(np.mean(list(deltas.values()))) if deltas else None,
            "paired_team_cluster_95_ci": _cluster_interval(deltas, args.seed + index + 1),
        }
    worst_deltas = reduce_team(selected_worst_delta_by_team)
    worst_policy = reduce_team(policy_worst_by_team)
    worst_selected = reduce_team(selected_worst_by_team)
    equal_weight_worst_delta = reduce_team(equal_weight_worst_delta_by_team)
    equal_weight_by_response = {
        name: {
            "team_clusters": len(deltas := reduce_team(equal_weight_response_delta_by_team[name])),
            "selected_minus_policy": float(np.mean(list(deltas.values()))) if deltas else None,
            "paired_team_cluster_95_ci": _cluster_interval(deltas, args.seed + 201 + index),
        }
        for index, name in enumerate(response_names)
    }
    metrics = {
        "team_split": {
            "train": len(train_teams),
            "validation": len(validation_teams),
            "test": len(test_teams),
            "unit": "team_cluster",
        },
        "states": len(rows),
        "test_states_selected": len(selected_records),
        "test_action_rows": int(test_row_mask.sum()),
        "response_names": response_names,
        "chance_split": {
            "training_repeats_per_response": midpoint,
            "test_repeats_per_response": chance_count - midpoint,
            "unit": "common chance index",
        },
        "best_epoch": best_epoch,
        "validation_action_outcome_brier": best_brier,
        "action_outcome_metrics_by_response": response_metrics,
        "constant_train_rate_metrics_by_response": baseline_metrics,
        "robust_selection": {
            "aggregation": "minimum predicted response win probability",
            "action_changed_from_policy_states": robust_selected_actions,
            "test_states": len(selected_records),
            "selected_worst_response_win_rate": float(np.mean(list(worst_selected.values()))) if worst_selected else None,
            "policy_worst_response_win_rate": float(np.mean(list(worst_policy.values()))) if worst_policy else None,
            "selected_minus_policy_worst_response": float(np.mean(list(worst_deltas.values()))) if worst_deltas else None,
            "selected_minus_policy_worst_response_95_ci": _cluster_interval(worst_deltas, args.seed + 101),
            "by_response": selection_by_response,
        },
        "equal_weight_selection": {
            "aggregation": "mean predicted response win probability",
            "action_changed_from_policy_states": equal_weight_selected_actions,
            "test_states": len(equal_weight_records),
            "selected_minus_policy_worst_response": float(np.mean(list(equal_weight_worst_delta.values()))) if equal_weight_worst_delta else None,
            "selected_minus_policy_worst_response_95_ci": _cluster_interval(equal_weight_worst_delta, args.seed + 301),
            "by_response": equal_weight_by_response,
        },
    }
    checkpoint = {
        "schema_version": 2,
        "model_type": "response_conditioned_action_win_probability",
        "target_description": "P(P1 wins after candidate joint action and frozen policy continuation, separately against each sampled response class)",
        "opponent_identity_is_model_input": False,
        "opponent_response_names": response_names,
        "robust_aggregation": "minimum",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.output_dir / "response_conditioned_action_value.pt"
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
        "metrics": metrics,
        "epoch_log": epoch_log,
        "selected_actions": selected_records,
        "equal_weight_actions": equal_weight_records,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(target_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
