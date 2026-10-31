#!/usr/bin/env python3
"""Audit a dense empirical-game target on Round479 held-out chance outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_counterfactual_policy_update import _load_rows, _target_distribution


def regret_matching(payoffs: np.ndarray, iterations: int = 5000) -> np.ndarray:
    """Approximate a row-player Nash mixture in a small zero-sum matrix."""
    n_rows, n_cols = payoffs.shape
    row_regret = np.zeros(n_rows)
    col_regret = np.zeros(n_cols)
    row_sum = np.zeros(n_rows)
    for _ in range(iterations):
        positive = np.maximum(row_regret, 0)
        row = positive / positive.sum() if positive.sum() else np.full(n_rows, 1 / n_rows)
        positive = np.maximum(col_regret, 0)
        col = positive / positive.sum() if positive.sum() else np.full(n_cols, 1 / n_cols)
        value = float(row @ payoffs @ col)
        row_regret += payoffs @ col - value
        col_regret += value - row @ payoffs
        row_sum += row
    return row_sum / iterations


def summarize(states: list[dict], target_by_state: list[np.ndarray], responses: list[str],
              train_repeats_by_state: dict[tuple[int, int], int], seed: int) -> dict:
    per_state = []
    for state, target in zip(states, target_by_state):
        actions = state['actions']
        base_i = next(i for i, a in enumerate(actions)
                      if tuple(a['action']) == state['base_action'])
        train_repeats = train_repeats_by_state[(state['team_cluster'], state['state_repeat'])]
        heldout = np.stack([[np.mean(a['outcomes'][r][train_repeats:]) for r in responses]
                            for a in actions])
        response_delta = target @ (heldout - heldout[base_i])
        picked_i = int(np.argmax(target))
        pure_delta = heldout[picked_i] - heldout[base_i]
        per_state.append({
            'cluster': state['team_cluster'],
            'response_delta': response_delta,
            'worst_response_delta': float(response_delta.min()),
            'pure_argmax_worst_delta': float(pure_delta.min()),
            'pure_argmax_changed': picked_i != base_i,
        })
    by_cluster: dict[int, list[dict]] = {}
    for row in per_state:
        by_cluster.setdefault(row['cluster'], []).append(row)
    cluster_worst = np.asarray([np.mean([x['worst_response_delta'] for x in rows])
                                for rows in by_cluster.values()])
    rng = np.random.default_rng(seed)
    bootstrap = cluster_worst[rng.integers(0, len(cluster_worst),
                                           (20000, len(cluster_worst)))].mean(axis=1)
    response_deltas = np.stack([x['response_delta'] for x in per_state])
    response_ci = {}
    for index, name in enumerate(responses):
        per_cluster = np.asarray([
            np.mean([x['response_delta'][index] for x in rows])
            for rows in by_cluster.values()
        ])
        response_bootstrap = per_cluster[
            rng.integers(0, len(per_cluster), (20000, len(per_cluster)))
        ].mean(axis=1)
        response_ci[name] = {
            'mean_delta': float(per_cluster.mean()),
            'cluster_bootstrap_95_ci': np.quantile(response_bootstrap, [0.025, 0.975]).tolist(),
        }
    return {
        'delta_by_response': response_ci,
        'mean_statewise_worst_response_delta': float(cluster_worst.mean()),
        'statewise_worst_cluster_bootstrap_95_ci': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        'positive_statewise_worst_response_states': int(sum(x['worst_response_delta'] > 0 for x in per_state)),
        'pure_argmax_changed_states': int(sum(x['pure_argmax_changed'] for x in per_state)),
        'pure_argmax_mean_statewise_worst_delta': float(np.mean([x['pure_argmax_worst_delta'] for x in per_state])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=202610481)
    ap.add_argument('--train-repeats', type=int,
                    help='number of leading chance repeats used for targets; default is half')
    args = ap.parse_args()
    states, responses = _load_rows(args.data)
    train_repeats_by_state = {}
    repeat_groups: dict[str, dict[str, int]] = {}
    for state in states:
        key = (state['team_cluster'], state['state_repeat'])
        repeats = len(state['actions'][0]['outcomes'][responses[0]])
        train_repeats = args.train_repeats or repeats // 2
        if not 1 <= train_repeats < repeats:
            ap.error(f'--train-repeats must be in [1, {repeats - 1}] for state {key}')
        train_repeats_by_state[key] = train_repeats
        group = repeat_groups.setdefault(str(repeats), {
            'states': 0, 'train_chance_repeats': train_repeats,
            'heldout_chance_repeats': repeats - train_repeats,
        })
        group['states'] += 1
    matrix_targets = []; robust_targets = []; base_targets = []; incumbent_mass = []; entropy = []
    for state in states:
        train_repeats = train_repeats_by_state[(state['team_cluster'], state['state_repeat'])]
        train = np.stack([[np.mean(a['outcomes'][r][:train_repeats]) for r in responses]
                          for a in state['actions']])
        base_i = next(i for i, a in enumerate(state['actions'])
                      if tuple(a['action']) == state['base_action'])
        mixed = regret_matching(train, args.iterations)
        robust = _target_distribution(state, responses, slice(0, train_repeats), 0.0, 0.15)
        base = np.zeros(len(state['actions'])); base[base_i] = 1.0
        matrix_targets.append(mixed); robust_targets.append(robust); base_targets.append(base)
        incumbent_mass.append(float(mixed[base_i]))
        entropy.append(float(-(mixed[mixed > 0] * np.log(mixed[mixed > 0])).sum()))
    result = {
        'method': 'Regret matching against a restricted opponent policy pool; not a full legal-action Nash equilibrium.',
        'data': str(args.data), 'states': len(states),
        'candidate_actions_mean': float(np.mean([len(s['actions']) for s in states])),
        'response_classes': responses, 'chance_split_by_repeat_count': repeat_groups,
        'regret_matching_iterations': args.iterations,
        'regret_matching_target': summarize(states, matrix_targets, responses,
                                            train_repeats_by_state, args.seed),
        'positive_worst_advantage_target': summarize(states, robust_targets, responses,
                                                     train_repeats_by_state, args.seed),
        'incumbent_policy_control': summarize(states, base_targets, responses,
                                              train_repeats_by_state, args.seed),
        'mean_regret_target_entropy_nats': float(np.mean(entropy)),
        'mean_regret_target_mass_on_incumbent': float(np.mean(incumbent_mass)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
