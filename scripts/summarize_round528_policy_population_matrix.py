#!/usr/bin/env python3
"""Summarize the completed Round528 restricted policy payoff matrix."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def maximin_mix(payoff):
    """Solve a small zero-sum matrix game by enumerating basic supports."""
    n = payoff.shape[0]
    best_p = None
    best_value = -np.inf
    tol = 1e-8
    for k in range(1, n + 1):
        supports = list(itertools.combinations(range(n), k))
        for rows in supports:
            for cols in supports:
                sub = payoff[np.ix_(rows, cols)].T
                system = np.block([
                    [sub, -np.ones((k, 1))],
                    [np.ones((1, k)), np.zeros((1, 1))],
                ])
                try:
                    solution = np.linalg.solve(system, np.r_[np.zeros(k), 1.0])
                except np.linalg.LinAlgError:
                    continue
                ps, active_value = solution[:-1], float(solution[-1])
                if np.any(ps < -tol):
                    continue
                p = np.zeros(n)
                p[list(rows)] = np.maximum(ps, 0.0)
                if p.sum() <= tol:
                    continue
                p /= p.sum()
                guarantee = float(np.min(p @ payoff))
                if guarantee + tol < active_value:
                    continue
                if guarantee > best_value:
                    best_p, best_value = p, guarantee
    if best_p is None:
        raise RuntimeError("Could not find a feasible matrix-game support")
    return best_p, best_value


def main():
    root = Path(sys.argv[1])
    requested, repeats, workers, seed = map(int, sys.argv[2:6])
    manifest = json.loads((root / "policies.json").read_text())
    names = manifest["names"]
    n = len(names)
    cluster_scores = {}
    throughput, errors, adjustments = {}, {}, {}
    total_games = 0
    total_seconds = 0.0

    for i, a in enumerate(names):
        for j in range(i + 1, n):
            b = names[j]
            d = json.loads((root / f"{a}_vs_{b}" / "paired_summary.json").read_text())
            rows = {}
            for row in d["results"]:
                if row.get("complete"):
                    rows.setdefault(int(row["team_seed"]), []).append(
                        float(row["checkpoint_points"]) / 2.0
                    )
            cluster_scores[(i, j)] = {k: float(np.mean(v)) for k, v in rows.items()}
            key = f"{a}_vs_{b}"
            throughput[key] = float(d["aggregate_games_per_second"])
            errors[key] = int(d["timeouts_or_errors"])
            adjustments[key] = int(d["action_adjustments"])
            total_games += int(d["games_completed"])
            total_seconds += float(d["parallel_wall_seconds"])

    common = sorted(set.intersection(*(set(x) for x in cluster_scores.values())))
    if len(common) < max(32, requested // 2):
        raise SystemExit(f"only {len(common)} common completed team clusters")

    matrix = np.full((n, n), 0.5, dtype=np.float64)
    pair_arrays = {}
    ci = {}
    for pair, scores in cluster_scores.items():
        vals = np.asarray([scores[k] for k in common], dtype=np.float64)
        pair_arrays[pair] = vals
        matrix[pair[0], pair[1]] = float(vals.mean())
        matrix[pair[1], pair[0]] = 1.0 - matrix[pair[0], pair[1]]
        rng = np.random.default_rng(seed + pair[0] * 100 + pair[1])
        ix = rng.integers(0, len(vals), (10000, len(vals)))
        boot = vals[ix].mean(axis=1)
        ci[f"{names[pair[0]]}_vs_{names[pair[1]]}"] = [
            float(x) for x in np.quantile(boot, [0.025, 0.975])
        ]

    point_mix, point_value = maximin_mix(matrix)
    rng = np.random.default_rng(seed + 528)
    bootstrap_replicates = 500
    bootstrap_indexes = rng.integers(
        0, len(common), (bootstrap_replicates, len(common))
    )
    mixes = np.zeros((bootstrap_replicates, n))
    values = np.zeros(bootstrap_replicates)
    for k, sample_ix in enumerate(bootstrap_indexes):
        sample_matrix = np.full((n, n), 0.5)
        for (i, j), vals in pair_arrays.items():
            sample_matrix[i, j] = float(vals[sample_ix].mean())
            sample_matrix[j, i] = 1.0 - sample_matrix[i, j]
        mixes[k], values[k] = maximin_mix(sample_matrix)

    result = {
        "status": "Complete; restricted policy-population payoff screen",
        "method": (
            "Unordered head-to-head games among policy checkpoints, both seats, "
            "common preview selector, common generated teams across pairings, "
            "and team-cluster bootstrap. Randomness inside Showdown is server-side. "
            "The maximin mix is only for this finite checkpoint population."
        ),
        "policies": names,
        "checkpoints": manifest["checkpoints"],
        "pairs_requested_per_matchup": requested,
        "repeats_per_team_cluster": repeats,
        "workers": workers,
        "seed": seed,
        "common_team_clusters": len(common),
        "payoff_matrix_row_score_rate": matrix.tolist(),
        "payoff_matrix_ci95_by_pair": ci,
        "restricted_maximin_mix": {
            name: float(point_mix[i]) for i, name in enumerate(names)
        },
        "restricted_maximin_value": point_value,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_maximin_value_ci95": [
            float(x) for x in np.quantile(values, [0.025, 0.975])
        ],
        "bootstrap_mean_strategy": {
            name: float(mixes[:, i].mean()) for i, name in enumerate(names)
        },
        "strategy_weight_ci95": {
            name: [float(x) for x in np.quantile(mixes[:, i], [0.025, 0.975])]
            for i, name in enumerate(names)
        },
        "strategy_support_frequency_gt_5pct": {
            name: float((mixes[:, i] > 0.05).mean()) for i, name in enumerate(names)
        },
        "timeouts_or_errors_by_matchup": errors,
        "action_adjustments_by_matchup": adjustments,
        "aggregate_games_completed": total_games,
        "sum_matchup_worker_wall_seconds": total_seconds,
        "sequential_end_to_end_games_per_second": total_games / max(total_seconds, 1e-9),
        "matchup_games_per_second": throughput,
        "interpretation": (
            "A restricted empirical matrix is useful only if bootstrap samples "
            "show a stable robust mix. It is not a Nash equilibrium of the full "
            "hidden-information VGC game and does not establish tournament strength."
        ),
        "reproduction": "scripts/evaluate_round528_policy_population_matrix.sh",
        "summarizer": "scripts/summarize_round528_policy_population_matrix.py",
    }
    (root / "payoff_matrix.csv").write_text(
        "policy," + ",".join(names) + "\n" + "\n".join(
            name + "," + ",".join(f"{x:.6f}" for x in matrix[i])
            for i, name in enumerate(names)
        ) + "\n"
    )
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        key: result[key] for key in [
            "common_team_clusters", "restricted_maximin_mix",
            "restricted_maximin_value", "bootstrap_maximin_value_ci95",
            "bootstrap_mean_strategy", "strategy_support_frequency_gt_5pct",
            "aggregate_games_completed",
            "sequential_end_to_end_games_per_second",
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
