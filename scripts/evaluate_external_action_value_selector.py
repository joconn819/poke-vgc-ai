#!/usr/bin/env python3
"""Evaluate a fitted action-value checkpoint on independent counterfactual states."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.train_action_conditioned_win_model import ActionWinModel, _inputs


def _cluster_ci(values: dict[int, float], seed: int, rounds: int = 20_000) -> list[float] | None:
    rows = np.asarray(list(values.values()), dtype=np.float64)
    if rows.size == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rows[rng.integers(0, rows.size, (rounds, rows.size))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bootstrap-seed", type=int, default=202610491)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_response_names = list(checkpoint["opponent_response_names"])
    evaluation_response_names = sorted(payload["rows"][0]["outcomes_by_opponent"])
    if not set(model_response_names).issubset(evaluation_response_names):
        raise ValueError("evaluation data lacks one or more fitted response classes")

    model = ActionWinModel(
        checkpoint["observation_dim"], checkpoint["action_width"],
        checkpoint["extra_feature_dim"], response_count=len(model_response_names),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Collapse semantic aliases so every distinct legal action is ranked once.
    states: dict[tuple[int, int], dict[tuple[int, int], dict]] = defaultdict(dict)
    for source in payload["rows"]:
        state_key = (int(source["team_cluster"]), int(source["state_repeat"]))
        action_key = tuple(map(int, source["action"]))
        target = states[state_key].get(action_key)
        if target is None:
            target = dict(source)
            target["candidate_names"] = {str(source["candidate"])}
            states[state_key][action_key] = target
        else:
            for field in ("observation", "legal_action_mask", "native_action_features",
                          "public_context_features", "outcomes_by_opponent"):
                if target[field] != source[field]:
                    raise ValueError(f"duplicate action has inconsistent {field}: {state_key}, {action_key}")
            target["candidate_names"].add(str(source["candidate"]))

    action_width = int(checkpoint["action_width"])
    use_policy_logp = bool(checkpoint["use_policy_log_probability"])
    records_by_method = {
        "maximin": [], "equal_weight": [],
        "family_maximin": [], "family_equal_weight": [],
    }
    action_changes = {name: 0 for name in records_by_method}
    per_team_response: dict[str, dict[int, list[float]]] = {
        method: {name: defaultdict(list) for name in evaluation_response_names}
        for method in records_by_method
    }
    per_team_worst = {method: defaultdict(list) for method in records_by_method}

    for (team_cluster, state_repeat), action_map in sorted(states.items()):
        candidates = list(action_map.values())
        if not candidates:
            continue
        features = np.stack([
            _inputs(
                row["observation"], row["action"], action_width,
                row["native_action_features"], row["public_context_features"],
                row.get("policy_log_probability") if use_policy_logp else None,
                use_policy_logp,
            )
            for row in candidates
        ])
        with torch.no_grad():
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            predictions = torch.sigmoid(model(feature_tensor)).cpu().numpy()
        response_index = {name: index for index, name in enumerate(model_response_names)}
        fitted_predictions = predictions[:, [response_index[name] for name in model_response_names]]
        # Related checkpoint labels belong to one policy family. Average their
        # predicted values before family-level choice so adding more snapshots
        # does not silently increase that family's weight.
        def family(name: str) -> str:
            return "ppo" if name == "policy" or name.startswith("policy_round") else name

        family_names = sorted({family(name) for name in model_response_names})
        family_predictions = np.stack([
            fitted_predictions[:, [i for i, name in enumerate(model_response_names)
                                  if family(name) == family_name]].mean(axis=1)
            for family_name in family_names
        ], axis=1)
        maximin_position = int(np.argmax(fitted_predictions.min(axis=1)))
        equal_weight_position = int(np.argmax(fitted_predictions.mean(axis=1)))
        family_maximin_position = int(np.argmax(family_predictions.min(axis=1)))
        family_equal_weight_position = int(np.argmax(family_predictions.mean(axis=1)))
        policy_positions = [
            index for index, row in enumerate(candidates)
            if "policy" in row.get("candidate_names", {str(row["candidate"])})
        ]
        if not policy_positions:
            continue
        policy_position = policy_positions[0]
        for method, selected_position in (
            ("maximin", maximin_position), ("equal_weight", equal_weight_position),
            ("family_maximin", family_maximin_position),
            ("family_equal_weight", family_equal_weight_position),
        ):
            selected = candidates[selected_position]
            baseline = candidates[policy_position]
            action_changes[method] += int(selected_position != policy_position)
            selected_rates = {}
            baseline_rates = {}
            for name in evaluation_response_names:
                selected_rate = float(np.mean(selected["outcomes_by_opponent"][name]))
                baseline_rate = float(np.mean(baseline["outcomes_by_opponent"][name]))
                selected_rates[name] = selected_rate
                baseline_rates[name] = baseline_rate
                per_team_response[method][name][team_cluster].append(selected_rate - baseline_rate)
            selected_worst = min(selected_rates.values())
            baseline_worst = min(baseline_rates.values())
            worst_delta = selected_worst - baseline_worst
            per_team_worst[method][team_cluster].append(worst_delta)
            records_by_method[method].append({
                "team_cluster": team_cluster,
                "state_repeat": state_repeat,
                "selected_action": selected["action"],
                "policy_action": baseline["action"],
                "observed_by_response": {
                    name: {
                        "selected_win_rate": selected_rates[name],
                        "policy_win_rate": baseline_rates[name],
                        "selected_minus_policy": selected_rates[name] - baseline_rates[name],
                    }
                    for name in evaluation_response_names
                },
                "observed_worst_response": {
                    "selected_win_rate": selected_worst,
                    "policy_win_rate": baseline_worst,
                    "selected_minus_policy": worst_delta,
                },
            })

    def reduce(values: dict[int, list[float]]) -> dict[int, float]:
        return {key: statistics.mean(items) for key, items in values.items()}

    metrics_by_method = {}
    for method in records_by_method:
        worst_values = reduce(per_team_worst[method])
        response_values = {
            name: reduce(per_team_response[method][name])
            for name in evaluation_response_names
        }
        metrics_by_method[method] = {
            "action_changed_from_policy_states": action_changes[method],
            "test_states": len(records_by_method[method]),
            "selected_minus_policy_worst_response": statistics.mean(worst_values.values()) if worst_values else None,
            "selected_minus_policy_worst_response_95_ci": _cluster_ci(
                worst_values, args.bootstrap_seed + (0 if method == "maximin" else 1)
            ),
            "by_response": {
                name: {
                    "team_clusters": len(values),
                    "selected_minus_policy": statistics.mean(values.values()) if values else None,
                    "paired_team_cluster_95_ci": _cluster_ci(
                        values, args.bootstrap_seed + 10 + index
                    ),
                }
                for index, (name, values) in enumerate(response_values.items())
            },
        }

    report = {
        "schema_version": 1,
        "experiment": "Independent action-value selector evaluation",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "seed": checkpoint.get("seed"),
        "metrics": {
            "team_split": {"test": len({key[0] for key in states}), "unit": "team_cluster"},
            "states": len(states),
            "validation_action_outcome_brier": checkpoint["metrics"]["validation_action_outcome_brier"],
            "response_names": model_response_names,
            "evaluation_response_names": evaluation_response_names,
            "heldout_response_names": sorted(set(evaluation_response_names) - set(model_response_names)),
            "robust_selection": metrics_by_method["maximin"],
            "equal_weight_selection": metrics_by_method["equal_weight"],
            "family_maximin_selection": metrics_by_method["family_maximin"],
            "family_equal_weight_selection": metrics_by_method["family_equal_weight"],
        },
        "selected_actions": records_by_method["maximin"],
        "equal_weight_actions": records_by_method["equal_weight"],
        "family_maximin_actions": records_by_method["family_maximin"],
        "family_equal_weight_actions": records_by_method["family_equal_weight"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "models": report["metrics"]["response_names"],
        "heldout_responses": report["metrics"]["heldout_response_names"],
        "test_clusters": report["metrics"]["team_split"]["test"],
        "maximin": metrics_by_method["maximin"]["selected_minus_policy_worst_response"],
        "equal_weight": metrics_by_method["equal_weight"]["selected_minus_policy_worst_response"],
        "family_maximin": metrics_by_method["family_maximin"]["selected_minus_policy_worst_response"],
        "family_equal_weight": metrics_by_method["family_equal_weight"]["selected_minus_policy_worst_response"],
    }, indent=2))


if __name__ == "__main__":
    main()
