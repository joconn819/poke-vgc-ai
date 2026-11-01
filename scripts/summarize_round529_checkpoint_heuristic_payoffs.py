#!/usr/bin/env python3
"""Summarize Round529 candidate-policy payoffs against policy and heuristic responses."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def maximin_mix(payoff):
    """Maximize a mixture over candidate rows against all opponent columns."""
    n, m = payoff.shape
    best_p, best_value = None, -np.inf
    tol = 1e-8
    for k in range(1, min(n, m) + 1):
        row_supports = list(itertools.combinations(range(n), k))
        col_supports = list(itertools.combinations(range(m), k))
        for rows in row_supports:
            for cols in col_supports:
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
        raise RuntimeError("Could not find a feasible rectangular-game support")
    return best_p, best_value



def approximate_equilibria(matrices, iterations=20000):
    """Batched regret matching for bootstrap matrices; returns mixtures and bounds."""
    batch, n, m = matrices.shape
    row_regrets = np.zeros((batch, n))
    col_regrets = np.zeros((batch, m))
    row_sum = np.zeros((batch, n))
    col_sum = np.zeros((batch, m))
    uniform_row = np.full((batch, n), 1.0 / n)
    uniform_col = np.full((batch, m), 1.0 / m)
    col_payoff = 1.0 - matrices.transpose(0, 2, 1)
    for _ in range(iterations):
        positive = np.maximum(row_regrets, 0.0)
        denom = positive.sum(axis=1, keepdims=True)
        row_strategy = np.divide(positive, denom, out=uniform_row.copy(), where=denom > 0)
        positive = np.maximum(col_regrets, 0.0)
        denom = positive.sum(axis=1, keepdims=True)
        col_strategy = np.divide(positive, denom, out=uniform_col.copy(), where=denom > 0)
        row_payoffs = np.einsum('bij,bj->bi', matrices, col_strategy)
        row_current = np.einsum('bi,bi->b', row_strategy, row_payoffs)
        row_regrets += row_payoffs - row_current[:, None]
        col_payoffs = np.einsum('bij,bj->bi', col_payoff, row_strategy)
        col_current = np.einsum('bi,bi->b', col_strategy, col_payoffs)
        col_regrets += col_payoffs - col_current[:, None]
        row_sum += row_strategy
        col_sum += col_strategy
    row_mix = row_sum / iterations
    col_mix = col_sum / iterations
    lower = np.min(np.einsum('bi,bij->bj', row_mix, matrices), axis=1)
    upper = np.max(np.einsum('bij,bj->bi', matrices, col_mix), axis=1)
    return row_mix, col_mix, lower, upper


def read_cluster_scores(path):
    d = json.loads(path.read_text())
    rows = {}
    for row in d["results"]:
        if row.get("complete"):
            rows.setdefault(int(row["team_seed"]), []).append(
                float(row["checkpoint_points"]) / 2.0
            )
    return {seed: float(np.mean(values)) for seed, values in rows.items()}, d


def main():
    root = Path(sys.argv[1])
    requested, repeats, workers, seed = map(int, sys.argv[2:6])
    manifest = json.loads((root / "policies.json").read_text())
    policies = manifest["names"]
    heuristics = ["max_damage", "counterplay", "public_counterplay"]
    opponents = policies + heuristics
    n, m = len(policies), len(opponents)
    base_root = Path(
        "training/outputs/phase12_headless/"
        "round528_policy_population_matrix_128x1_20261031"
    )

    scores = {}
    errors, adjustments, throughput = {}, {}, {}
    for i, policy in enumerate(policies):
        for heuristic in heuristics:
            key = f"{policy}_vs_{heuristic}"
            scores[(i, n + heuristics.index(heuristic))], d = read_cluster_scores(
                root / key / "paired_summary.json"
            )
            errors[key] = int(d["timeouts_or_errors"])
            adjustments[key] = int(d["action_adjustments"])
            throughput[key] = float(d["aggregate_games_per_second"])

    # Import all policy-vs-policy matchup data, reversing row perspective where needed.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lo, hi = sorted((i, j))
            folder = base_root / f"{policies[lo]}_vs_{policies[hi]}"
            base_scores, _ = read_cluster_scores(folder / "paired_summary.json")
            scores[(i, j)] = (
                base_scores if i < j
                else {team_seed: 1.0 - value for team_seed, value in base_scores.items()}
            )

    common = sorted(set.intersection(*(set(x) for x in scores.values())))
    if len(common) < max(32, requested // 2):
        raise SystemExit(f"only {len(common)} common completed team clusters")

    cells = {}
    matrix = np.full((n, m), 0.5, dtype=np.float64)
    intervals = {}
    for i in range(n):
        for j in range(m):
            if j < n and i == j:
                vals = np.full(len(common), 0.5)
            else:
                vals = np.asarray([scores[(i, j)][k] for k in common])
            cells[(i, j)] = vals
            matrix[i, j] = float(vals.mean())
            rng = np.random.default_rng(seed + i * 100 + j)
            ix = rng.integers(0, len(vals), (10000, len(vals)))
            boot = vals[ix].mean(axis=1)
            intervals[f"{policies[i]}_vs_{opponents[j]}"] = [
                float(x) for x in np.quantile(boot, [0.025, 0.975])
            ]

    point_mix, point_value = maximin_mix(matrix)
    point_adversary_mix, point_adversary_value = maximin_mix(1.0 - matrix.T)
    bootstrap_replicates = 200
    regret_matching_iterations = 20000
    rng = np.random.default_rng(seed + 529)
    sample_ix = rng.integers(0, len(common), (bootstrap_replicates, len(common)))
    bootstrap_matrices = np.empty((bootstrap_replicates, n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            bootstrap_matrices[:, i, j] = cells[(i, j)][sample_ix].mean(axis=1)
    mixes, adversary_mixes, lower_values, upper_values = approximate_equilibria(
        bootstrap_matrices, iterations=regret_matching_iterations
    )
    values = lower_values
    adversary_values = 1.0 - upper_values
    equilibrium_gaps = np.maximum(upper_values - lower_values, 0.0)

    total_games = 0
    total_seconds = 0.0
    for dpath in root.glob("*/paired_summary.json"):
        d = json.loads(dpath.read_text())
        total_games += int(d["games_completed"])
        total_seconds += float(d["parallel_wall_seconds"])

    result = {
        "status": "Complete; checkpoint policies vs fixed heuristic and policy responses",
        "method": (
            "Seven policy checkpoints are the candidate strategy rows. Opponent "
            "columns are those seven checkpoints plus Max Damage, Counterplay, and "
            "Public Counterplay. Scores use both player seats, shared team/preview "
            "selection, 128 requested team clusters, and a common team-cluster "
            "bootstrap. This restricted maximin is not a full-game Nash solution."
        ),
        "policies": policies,
        "opponents": opponents,
        "heuristics": heuristics,
        "candidate_checkpoints": manifest["checkpoints"],
        "pairs_requested_per_matchup": requested,
        "repeats_per_team_cluster": repeats,
        "workers": workers,
        "seed": seed,
        "common_team_clusters": len(common),
        "payoff_matrix_candidate_score_rate": matrix.tolist(),
        "payoff_matrix_ci95_by_pair": intervals,
        "restricted_maximin_mix": {
            name: float(point_mix[i]) for i, name in enumerate(policies)
        },
        "restricted_maximin_value": point_value,
        "restricted_adversary_mix": {
            name: float(point_adversary_mix[j]) for j, name in enumerate(opponents)
        },
        "restricted_adversary_value": point_adversary_value,
        "bootstrap_adversary_value_ci95": [
            float(x) for x in np.quantile(adversary_values, [0.025, 0.975])
        ],
        "bootstrap_mean_adversary_strategy": {
            name: float(adversary_mixes[:, j].mean()) for j, name in enumerate(opponents)
        },
        "adversary_strategy_weight_ci95": {
            name: [float(x) for x in np.quantile(adversary_mixes[:, j], [0.025, 0.975])]
            for j, name in enumerate(opponents)
        },
        "adversary_support_frequency_gt_5pct": {
            name: float((adversary_mixes[:, j] > 0.05).mean()) for j, name in enumerate(opponents)
        },
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_regret_matching_iterations": regret_matching_iterations,
        "bootstrap_equilibrium_gap_mean": float(equilibrium_gaps.mean()),
        "bootstrap_equilibrium_gap_ci95": [float(x) for x in np.quantile(equilibrium_gaps, [0.025, 0.975])],
        "bootstrap_maximin_value_ci95": [
            float(x) for x in np.quantile(values, [0.025, 0.975])
        ],
        "bootstrap_mean_strategy": {
            name: float(mixes[:, i].mean()) for i, name in enumerate(policies)
        },
        "strategy_weight_ci95": {
            name: [float(x) for x in np.quantile(mixes[:, i], [0.025, 0.975])]
            for i, name in enumerate(policies)
        },
        "strategy_support_frequency_gt_5pct": {
            name: float((mixes[:, i] > 0.05).mean()) for i, name in enumerate(policies)
        },
        "timeouts_or_errors_by_matchup": errors,
        "action_adjustments_by_matchup": adjustments,
        "matchup_games_per_second": throughput,
        "aggregate_games_completed": total_games,
        "sum_matchup_worker_wall_seconds": total_seconds,
        "sequential_end_to_end_games_per_second": total_games / max(total_seconds, 1e-9),
        "matchups_completed": len(throughput),
        "interpretation": (
            "Use this matrix to ask whether any empirical checkpoint mixture can "
            "raise its lower bound against both past policies and fixed heuristic "
            "responses. Bootstrap support must be stable before using the mixture "
            "for training. It does not imply Nash equilibrium in full VGC."
        ),
        "reproduction": "scripts/evaluate_round529_checkpoint_heuristic_payoffs.sh",
        "summarizer": "scripts/summarize_round529_checkpoint_heuristic_payoffs.py",
    }
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "payoff_matrix.csv").write_text(
        "policy," + ",".join(opponents) + "\n" + "\n".join(
            policy + "," + ",".join(f"{x:.6f}" for x in matrix[i])
            for i, policy in enumerate(policies)
        ) + "\n"
    )
    print(json.dumps({
        key: result[key] for key in [
            "common_team_clusters", "restricted_maximin_mix",
            "restricted_maximin_value", "bootstrap_maximin_value_ci95",
            "restricted_adversary_mix", "restricted_adversary_value",
            "bootstrap_adversary_value_ci95", "bootstrap_mean_adversary_strategy",
            "adversary_support_frequency_gt_5pct",
            "bootstrap_mean_strategy", "strategy_weight_ci95",
            "aggregate_games_completed", "sequential_end_to_end_games_per_second",
            "bootstrap_equilibrium_gap_mean", "bootstrap_equilibrium_gap_ci95",
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
