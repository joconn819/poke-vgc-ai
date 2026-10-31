#!/usr/bin/env python3
"""Audit whether Round484 candidate actions contain transferable local gains.

For each saved state, choose an action using half of its sampled outcomes, then
score it on the other half. This is an offline one-step diagnostic: each
counterfactual branch resumes with the reference policy, so it is not a full
policy or Nash-equilibrium evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RESPONSES = ("max_damage", "public_counterplay", "policy")
SPLITS = {
    "first_half_train_last_half_test": ((0, 1, 2, 3), (4, 5, 6, 7)),
    "even_train_odd_test": ((0, 2, 4, 6), (1, 3, 5, 7)),
}


def _mean_response_scores(row: dict[str, Any], indices: tuple[int, ...]) -> list[float]:
    return [
        statistics.mean(row["outcomes_by_opponent"][response][i] for i in indices)
        for response in RESPONSES
    ]


def _bootstrap_ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def _run_split(
    states: dict[int, dict[tuple[int, int], dict[str, Any]]],
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {"mean": [], "maximin": []}
    changed = {"mean": 0, "maximin": 0}
    for cluster, candidates in states.items():
        baseline_action = next(
            action for action, row in candidates.items() if row["candidate"] == "policy"
        )
        train_scores = {
            action: _mean_response_scores(row, train_indices)
            for action, row in candidates.items()
        }
        test_scores = {
            action: _mean_response_scores(row, test_indices)
            for action, row in candidates.items()
        }
        selected = {
            "mean": max(candidates, key=lambda a: statistics.mean(train_scores[a])),
            "maximin": max(candidates, key=lambda a: min(train_scores[a])),
        }
        for objective, action in selected.items():
            if action != baseline_action:
                changed[objective] += 1
            baseline = test_scores[baseline_action]
            chosen = test_scores[action]
            records[objective].append(
                {
                    "cluster": cluster,
                    "response_delta": [c - b for c, b in zip(chosen, baseline)],
                    "equal_weight_delta": statistics.mean(chosen) - statistics.mean(baseline),
                    "worst_response_delta": min(chosen) - min(baseline),
                }
            )

    output: dict[str, Any] = {}
    for objective, rows in records.items():
        response_delta = [
            statistics.mean(row["response_delta"][i] for row in rows)
            for i in range(len(RESPONSES))
        ]
        equal_weight = [row["equal_weight_delta"] for row in rows]
        worst_response = [row["worst_response_delta"] for row in rows]
        offset = 0 if objective == "mean" else 100
        output[objective] = {
            "states": len(rows),
            "selected_nonbaseline_action_states": changed[objective],
            "heldout_delta_by_response_pp": {
                response: round(value * 100, 3)
                for response, value in zip(RESPONSES, response_delta)
            },
            "heldout_equal_weight_delta_pp": round(statistics.mean(equal_weight) * 100, 3),
            "heldout_equal_weight_delta_ci95_pp": [
                round(x * 100, 3)
                for x in _bootstrap_ci(equal_weight, seed=seed + offset + 1, samples=bootstrap_samples)
            ],
            "heldout_statewise_worst_response_delta_pp": round(
                statistics.mean(worst_response) * 100, 3
            ),
            "heldout_statewise_worst_response_delta_ci95_pp": [
                round(x * 100, 3)
                for x in _bootstrap_ci(worst_response, seed=seed + offset + 2, samples=bootstrap_samples)
            ],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=4_852_026)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    states: dict[int, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    category_state_coverage: dict[str, set[int]] = defaultdict(set)
    for row in data["rows"]:
        cluster = int(row["team_cluster"])
        action = tuple(int(x) for x in row["action"])
        # Candidate labels may alias the same legal joint action; score each
        # actual action once and preserve its first, reproducible label.
        states[cluster].setdefault(action, row)
        name = str(row["candidate"])
        for category in ("switch", "protect", "setup", "status", "redirection", "legal_perturbation"):
            if name.startswith(category + "_"):
                category_state_coverage[category].add(cluster)

    if not states:
        raise SystemExit("No candidate states found")
    for candidates in states.values():
        for row in candidates.values():
            if any(len(row["outcomes_by_opponent"].get(response, [])) != 8 for response in RESPONSES):
                raise SystemExit("Expected exactly 8 independent repeats for each response class")

    report = {
        "schema_version": 1,
        "experiment": "Round484 heldout-repeat candidate action selection audit",
        "source": str(args.input),
        "source_checkpoint": data["checkpoint"],
        "candidate_policy_checkpoint": data["candidate_policy_checkpoint"],
        "states": len(states),
        "responses": list(RESPONSES),
        "distinct_candidate_actions_per_state": {
            "mean": round(statistics.mean(len(c) for c in states.values()), 3),
            "min": min(map(len, states.values())),
            "max": max(map(len, states.values())),
        },
        "candidate_category_state_coverage": {
            category: len(clusters) for category, clusters in sorted(category_state_coverage.items())
        },
        "method": (
            "For every state, select one candidate action using four of its eight outcome repeats, "
            "averaging equally across Max Damage, Public Counterplay, and Round476 policy responses. "
            "Score the selected action on the disjoint four repeats. The mean objective maximizes "
            "equal-weight response-class win rate; maximin maximizes the lowest sampled response-class "
            "win rate. Paired percentile bootstrap resamples whole team clusters."
        ),
        "interpretation_limit": (
            "This measures a local candidate-action ceiling only. After the chosen first action, each "
            "branch continues with the reference policy; the three response classes are a finite league, "
            "not all legal opponent actions. It is neither a full-policy evaluation nor a Nash certificate."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "splits": {
            name: _run_split(
                states, train_indices, test_indices,
                seed=args.seed + index * 1_000,
                bootstrap_samples=args.bootstrap_samples,
            )
            for index, (name, (train_indices, test_indices)) in enumerate(SPLITS.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": len(states), "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()
