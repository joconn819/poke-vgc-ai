#!/usr/bin/env python3
"""Build a restricted empirical zero-sum payoff game from Round582 match cells."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def read_scores(path):
    data = json.loads(path.read_text())
    by_seed = {}
    for row in data['results']:
        if row.get('complete'):
            by_seed.setdefault(int(row['team_seed']), []).append(float(row['checkpoint_points']) / 2.0)
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}, data


def maximin_mix(payoff):
    n, m = payoff.shape
    best_mix, best_value = None, -np.inf
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.combinations(range(m), k):
                sub = payoff[np.ix_(rows, cols)].T
                system = np.block([[sub, -np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
                try:
                    solution = np.linalg.solve(system, np.r_[np.zeros(k), 1.0])
                except np.linalg.LinAlgError:
                    continue
                probs, value = solution[:-1], float(solution[-1])
                if np.any(probs < -1e-8):
                    continue
                mix = np.zeros(n)
                mix[list(rows)] = np.maximum(probs, 0.0)
                if mix.sum() <= 1e-8:
                    continue
                mix /= mix.sum()
                guarantee = float(np.min(mix @ payoff))
                if guarantee + 1e-8 >= value and guarantee > best_value:
                    best_mix, best_value = mix, guarantee
    if best_mix is None:
        raise RuntimeError('No feasible restricted-game maximin strategy found')
    return best_mix, best_value


def regret_mixed_equilibria(matrices, iterations=10000):
    batch, n, m = matrices.shape
    row_regrets = np.zeros((batch, n)); col_regrets = np.zeros((batch, m))
    row_sum = np.zeros((batch, n)); col_sum = np.zeros((batch, m))
    uniform_row = np.full((batch, n), 1/n); uniform_col = np.full((batch, m), 1/m)
    col_payoff = 1 - matrices.transpose(0, 2, 1)
    for _ in range(iterations):
        positive = np.maximum(row_regrets, 0); denom = positive.sum(axis=1, keepdims=True)
        row_strategy = np.divide(positive, denom, out=uniform_row.copy(), where=denom > 0)
        positive = np.maximum(col_regrets, 0); denom = positive.sum(axis=1, keepdims=True)
        col_strategy = np.divide(positive, denom, out=uniform_col.copy(), where=denom > 0)
        row_payoffs = np.einsum('bij,bj->bi', matrices, col_strategy)
        row_regrets += row_payoffs - np.einsum('bi,bi->b', row_strategy, row_payoffs)[:, None]
        col_payoffs = np.einsum('bij,bj->bi', col_payoff, row_strategy)
        col_regrets += col_payoffs - np.einsum('bi,bi->b', col_strategy, col_payoffs)[:, None]
        row_sum += row_strategy; col_sum += col_strategy
    row_mix, col_mix = row_sum/iterations, col_sum/iterations
    lower = np.min(np.einsum('bi,bij->bj', row_mix, matrices), axis=1)
    upper = np.max(np.einsum('bij,bj->bi', matrices, col_mix), axis=1)
    return row_mix, col_mix, lower, upper


def main():
    root = Path(sys.argv[1]); requested, repeats, workers, seed = map(int, sys.argv[2:6])
    manifest = json.loads((root/'policies.json').read_text())
    policies = manifest['names']; heuristics = ['max_damage', 'counterplay', 'public_counterplay']
    opponents = policies + heuristics; n, m = len(policies), len(opponents)
    cells, diagnostics = {}, {}
    for i, row_name in enumerate(policies):
        for j, col_name in enumerate(opponents):
            if j < n and j == i:
                continue
            if j < n and j < i:
                path = root/f'{col_name}_vs_{row_name}'/'paired_summary.json'
                scores, _ = read_scores(path)
                scores = {k: 1-v for k, v in scores.items()}
                key = f'{col_name}_vs_{row_name}'
            else:
                path = root/f'{row_name}_vs_{col_name}'/'paired_summary.json'
                scores, _ = read_scores(path); key = f'{row_name}_vs_{col_name}'
            cells[(i,j)] = scores
            raw = json.loads(path.read_text())
            diagnostics[key] = {
                'games_completed': int(raw['games_completed']),
                'timeouts_or_errors': int(raw['timeouts_or_errors']),
                'action_adjustments': int(raw['action_adjustments']),
                'games_per_second': float(raw['aggregate_games_per_second']),
            }
    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    if len(common) < max(32, requested//2):
        raise SystemExit(f'Only {len(common)} common team-pair clusters across all matrix cells')
    matrix = np.full((n,m), .5); arrays = {}; intervals = {}
    for (i,j), values in cells.items():
        arr = np.asarray([values[k] for k in common]); arrays[(i,j)] = arr
        matrix[i,j] = arr.mean()
        boot_ix = np.random.default_rng(seed + i*100 + j).integers(0, len(arr), (10000,len(arr)))
        intervals[f'{policies[i]}_vs_{opponents[j]}'] = [float(x) for x in np.quantile(arr[boot_ix].mean(axis=1), [.025,.975])]
    pure_mix, pure_value = maximin_mix(matrix)
    point_response_mix, point_response_value = maximin_mix(1-matrix.T)
    B = 300; rng = np.random.default_rng(seed+582)
    indexes = rng.integers(0, len(common), (B,len(common)))
    bootstrap = np.broadcast_to(matrix, (B,n,m)).copy()
    for (i,j), arr in arrays.items(): bootstrap[:,i,j] = arr[indexes].mean(axis=1)
    row_mixes, col_mixes, lower, upper = regret_mixed_equilibria(bootstrap)
    report = {
        'status':'Complete; fresh restricted empirical policy game',
        'method':'Eight historical PPO policies as candidate rows; those policies plus Max Damage, Counterplay, and Public Counterplay as response columns. Each matchup uses the same 128 team-pair clusters and both seat orders; each cluster score averages the two side-swapped outcomes. Team-cluster bootstrap accounts for matchup variation. This is a restricted game over checkpoint policies and three fixed heuristics, not a full VGC Nash certificate.',
        'policies':policies,'opponents':opponents,'heuristics':heuristics,
        'checkpoints':manifest['checkpoints'],'requested_clusters_per_matchup':requested,
        'outcome_repeats':repeats,'workers':workers,'seed':seed,
        'common_team_clusters':len(common),
        'payoff_matrix_candidate_score_rate':matrix.tolist(),
        'payoff_matrix_ci95_by_cell':intervals,
        'restricted_maximin_mix':{p:float(pure_mix[i]) for i,p in enumerate(policies)},
        'restricted_maximin_value':float(pure_value),
        'restricted_adversary_mix':{p:float(point_response_mix[j]) for j,p in enumerate(opponents)},
        'restricted_adversary_value':float(point_response_value),
        'bootstrap_equilibrium_row_mix_mean':{p:float(row_mixes[:,i].mean()) for i,p in enumerate(policies)},
        'bootstrap_equilibrium_row_mix_ci95':{p:[float(x) for x in np.quantile(row_mixes[:,i],[.025,.975])] for i,p in enumerate(policies)},
        'bootstrap_equilibrium_response_mix_mean':{p:float(col_mixes[:,j].mean()) for j,p in enumerate(opponents)},
        'bootstrap_equilibrium_response_mix_ci95':{p:[float(x) for x in np.quantile(col_mixes[:,j],[.025,.975])] for j,p in enumerate(opponents)},
        'bootstrap_restricted_value_ci95':[float(x) for x in np.quantile(lower,[.025,.975])],
        'bootstrap_equilibrium_gap_mean':float(np.maximum(upper-lower,0).mean()),
        'direct_heuristic_scores':{h:{policies[i]:float(matrix[i,n+heuristics.index(h)]) for i in range(n)} for h in heuristics},
        'timeouts_errors_and_speed':diagnostics,
        'interpretation':'Use this fresh matrix to select an opponent mixture for the next best-response PPO iteration and to identify whether any current checkpoint mixture approaches 50% against the available responses. A value below 50% confirms that this restricted population remains exploitable; it does not estimate exploitability against policies absent from the matrix.',
        'reproduction':'scripts/evaluate_round582_current_policy_meta_game.sh',
        'summarizer':'scripts/summarize_round582_current_policy_meta_game.py',
    }
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'payoff_matrix.csv').write_text('policy,'+','.join(opponents)+'\n'+'\n'.join(p+','+','.join(f'{x:.6f}' for x in matrix[i]) for i,p in enumerate(policies))+'\n')
    print(json.dumps({k:report[k] for k in ['common_team_clusters','restricted_maximin_mix','restricted_maximin_value','restricted_adversary_mix','restricted_adversary_value','bootstrap_restricted_value_ci95','bootstrap_equilibrium_response_mix_mean','bootstrap_equilibrium_gap_mean','direct_heuristic_scores']},indent=2))


if __name__ == '__main__': main()
