#!/usr/bin/env python3
"""Measure held-out one-step action selection as the chance-sample budget varies."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RESPONSES = ("max_damage", "public_counterplay", "policy")
SAMPLE_BUDGETS = (1, 2, 4)


def _ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def _evaluate_split(
    states: dict[int, dict[tuple[int, int], dict[str, Any]]],
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired: dict[str, list[dict[str, Any]]] = {"equal_weight": [], "maximin": []}
    changed = {"equal_weight": 0, "maximin": 0}
    for cluster, actions in states.items():
        baseline = next(action for action, row in actions.items() if row["candidate"] == "policy")

        def response_means(row: dict[str, Any], indices: tuple[int, ...]) -> list[float]:
            return [
                statistics.mean(row["outcomes_by_opponent"][response][index] for index in indices)
                for response in RESPONSES
            ]

        train_scores = {action: response_means(row, train_indices) for action, row in actions.items()}
        test_scores = {action: response_means(row, test_indices) for action, row in actions.items()}
        selections = {
            "equal_weight": max(actions, key=lambda action: statistics.mean(train_scores[action])),
            "maximin": max(actions, key=lambda action: min(train_scores[action])),
        }
        for name, selected in selections.items():
            changed[name] += selected != baseline
            base_score, selected_score = test_scores[baseline], test_scores[selected]
            paired[name].append({
                "response_delta": [new - old for old, new in zip(base_score, selected_score)],
                "composite_delta": statistics.mean(selected_score) - statistics.mean(base_score),
                "worst_delta": min(selected_score) - min(base_score),
            })

    report: dict[str, Any] = {}
    for name, rows in paired.items():
        response_deltas = [
            statistics.mean(row["response_delta"][index] for row in rows)
            for index in range(len(RESPONSES))
        ]
        composite = [row["composite_delta"] for row in rows]
        worst = [row["worst_delta"] for row in rows]
        report[name] = {
            "selected_nonbaseline_actions": changed[name],
            "heldout_delta_by_response_pp": {
                response: round(delta * 100, 3)
                for response, delta in zip(RESPONSES, response_deltas)
            },
            "heldout_equal_weight_delta_pp": round(statistics.mean(composite) * 100, 3),
            "heldout_equal_weight_ci95_pp": [
                round(value * 100, 3)
                for value in _ci(composite, seed=bootstrap_seed, samples=bootstrap_samples)
            ],
            "heldout_statewise_worst_response_delta_pp": round(statistics.mean(worst) * 100, 3),
            "heldout_statewise_worst_response_ci95_pp": [
                round(value * 100, 3)
                for value in _ci(worst, seed=bootstrap_seed + 1, samples=bootstrap_samples)
            ],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202610487)
    parser.add_argument("--games-per-second", type=float, default=51.195)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    states: dict[int, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in data["rows"]:
        action = tuple(int(value) for value in row["action"])
        states[int(row["team_cluster"])].setdefault(action, row)
    if not states:
        raise SystemExit("No action states found")

    splits: dict[str, Any] = {}
    mean_candidates = statistics.mean(len(actions) for actions in states.values())
    for budget in SAMPLE_BUDGETS:
        first_train = tuple(range(budget))
        first_test = tuple(range(budget, 8))
        second_train = tuple(range(8 - budget, 8))
        second_test = tuple(range(0, 8 - budget))
        branch_count = mean_candidates * len(RESPONSES) * budget
        splits[str(budget)] = {
            "samples_per_action_response": budget,
            "sample_split_sensitivity": {
                "first_repeats_train": _evaluate_split(
                    states, first_train, first_test,
                    bootstrap_seed=args.seed + budget * 100,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "last_repeats_train": _evaluate_split(
                    states, second_train, second_test,
                    bootstrap_seed=args.seed + budget * 100 + 1,
                    bootstrap_samples=args.bootstrap_samples,
                ),
            },
            "mean_candidate_actions_per_state": round(mean_candidates, 3),
            "estimated_counterfactual_branches_per_decision": round(branch_count, 1),
            "estimated_wall_seconds_at_measured_collection_rate": round(
                branch_count / args.games_per_second, 3
            ),
        }
    report = {
        "schema_version": 1,
        "experiment": "Round487 candidate action chance-sample budget",
        "source": str(args.input),
        "source_checkpoint": data["checkpoint"],
        "states": len(states),
        "chance_repeats_available": 8,
        "response_classes": list(RESPONSES),
        "method": (
            "For each state and each sample budget, choose a legal joint action by equal-weight mean "
            "or maximin over the three response classes using either the first or last k repeats. "
            "Evaluate on the disjoint remaining repeats. Confidence intervals bootstrap whole team clusters."
        ),
        "measured_collection_rate_games_per_second": args.games_per_second,
        "budgets": splits,
        "interpretation_limit": (
            "Offline one-step action ceiling with reference-policy continuation and only three sampled "
            "response classes. Estimated latency extrapolates collection throughput and does not equal "
            "optimized in-process search latency."
        ),
        "bootstrap_samples": args.bootstrap_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        budget: values["sample_split_sensitivity"]
        for budget, values in splits.items()
    }
    print(json.dumps({"output": str(args.output), "states": len(states), "budgets": compact}, indent=2))


if __name__ == "__main__":
    main()
