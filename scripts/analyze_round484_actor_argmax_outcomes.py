#!/usr/bin/env python3
"""Evaluate actor greedy actions from paired counterfactual outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_headless_showdown_self_play import _load_policy
from src.training.self_play import PolicyAgent


def cluster_ci(values: list[float], seed: int) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), (20000, len(x)))].mean(axis=1)
    return np.quantile(sample, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=202610485)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text())
    policy = _load_policy(args.checkpoint, args.device)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in payload["rows"]:
        grouped.setdefault((int(row["team_cluster"]), int(row["state_repeat"])), []).append(row)

    responses = sorted(payload["paired_response_classes"])
    deltas: dict[str, list[float]] = {name: [] for name in responses}
    worst_deltas: list[float] = []
    chosen_changed = 0
    explicit_alias_states = 0
    alias_argmax_consistent = 0
    candidate_mass: list[float] = []
    chosen_names: dict[str, int] = {}
    for key, rows in sorted(grouped.items()):
        rows_by_action: dict[tuple[int, int], dict] = {}
        for row in rows:
            rows_by_action.setdefault(tuple(map(int, row["action"])), row)
        actions = list(rows_by_action)
        template = rows[0]
        actor = PolicyAgent(
            policy, template["player_team"], template["opponent_public_team"],
            deterministic=True, device=args.device,
            public_context_features=(
                np.asarray(template["public_context_features"], dtype=np.float32)
                if template.get("public_context_features") is not None else None
            ),
        )
        scores = np.asarray(actor.log_prob_native_actions(
            np.asarray(template["observation"], dtype=np.float32),
            np.asarray(template["legal_action_mask"], dtype=np.int8),
            [list(action) for action in actions],
        ), dtype=np.float64)
        probabilities = np.exp(scores)
        candidate_mass.append(float(probabilities.sum()))
        selected = actions[int(np.argmax(scores))]
        baseline = next((tuple(map(int, row["action"])) for row in rows
                         if row["candidate"] == "policy"), None)
        if baseline is None:
            raise ValueError(f"state {key} has no policy baseline action")
        chosen_changed += selected != baseline
        named = next((tuple(map(int, row["action"])) for row in rows
                      if row["candidate"] == "candidate_policy_argmax"), None)
        if named is not None:
            explicit_alias_states += 1
            alias_argmax_consistent += selected == named
        selected_name = rows_by_action[selected]["candidate"]
        chosen_names[selected_name] = chosen_names.get(selected_name, 0) + 1
        base_row = rows_by_action[baseline]
        selected_row = rows_by_action[selected]
        state_deltas = []
        for response in responses:
            test_values = np.asarray(selected_row["outcomes_by_opponent"][response][
                len(selected_row["outcomes_by_opponent"][response]) // 2:
            ], dtype=np.float64)
            base_values = np.asarray(base_row["outcomes_by_opponent"][response][
                len(base_row["outcomes_by_opponent"][response]) // 2:
            ], dtype=np.float64)
            delta = float(np.mean(test_values - base_values))
            deltas[response].append(delta)
            state_deltas.append(delta)
        worst_deltas.append(min(state_deltas))

    result = {
        "method": "Round482 actor's highest-probability action from the sampled candidate set; the collector adds the full legal-action argmax whenever it is not already represented. Only the final half of paired chance repeats is used.",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "states": len(grouped),
        "responses": responses,
        "explicit_candidate_policy_argmax_rows": explicit_alias_states,
        "explicit_alias_matches_actor_scored_argmax": alias_argmax_consistent,
        "actor_action_changed_from_round430": chosen_changed,
        "mean_probability_mass_on_candidate_set": float(np.mean(candidate_mass)),
        "chosen_candidate_names": chosen_names,
        "mean_delta_by_response": {
            response: float(np.mean(values)) for response, values in deltas.items()
        },
        "cluster_bootstrap_95_ci_by_response": {
            response: cluster_ci(values, args.seed + index)
            for index, (response, values) in enumerate(deltas.items())
        },
        "mean_statewise_worst_response_delta": float(np.mean(worst_deltas)),
        "statewise_worst_cluster_bootstrap_95_ci": cluster_ci(worst_deltas, args.seed + 100),
        "scope_limit": "one-step first-decision action intervention with Round430 as the subsequent player policy; offline counterfactuals, not complete matches",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
