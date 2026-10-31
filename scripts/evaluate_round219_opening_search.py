#!/usr/bin/env python3
"""Paired closed-sheet opening-search screen against Max Damage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_headless_showdown_self_play import _load_policy, _load_team_selector, run_game
from src.env.headless_showdown import HeadlessShowdownWorker
from src.models.state_value import load_state_value_checkpoint
from src.search.local_showdown import LocalShowdownSearchPool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--selector', type=Path,
                   help='optional team selector; omit only with --random-preview')
    p.add_argument('--random-preview', action='store_true',
                   help='use the same random team preview as the standalone baseline')
    p.add_argument('--showdown-path', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--format', default='gen9championsdoubles2026mcclosed')
    p.add_argument('--clusters', type=int, default=128)
    p.add_argument('--player-strategy', choices=('policy', 'public_counterplay'), default='policy')
    p.add_argument('--opponent-strategy', choices=('max_damage', 'public_counterplay'), default='max_damage',
                   help='opponent policy used for both raw and search arms')
    p.add_argument('--counterplay-incoming-weight', type=float, default=1.0)
    p.add_argument('--counterplay-ko-bonus', type=float, default=0.35,
                   help='default 0.35 matches the strongest measured public Counterplay baseline')
    p.add_argument('--start-cluster', type=int, default=0)
    p.add_argument('--seed', type=int, default=20262190)
    p.add_argument('--hidden-worlds', type=int, default=2)
    p.add_argument('--candidates', type=int, default=4)
    p.add_argument('--response-candidates', type=int,
                   help='opponent response shortlist width; defaults to --candidates')
    p.add_argument('--search-workers', type=int, default=1,
                   help='parallel local Showdown search workers per decision')
    p.add_argument('--candidate-damage-samples', type=int, default=4,
                   help='fixed damage estimates per move/target during candidate scoring')
    p.add_argument('--chance-samples', type=int, default=1,
                   help='independent Showdown outcome samples per candidate joint action')
    p.add_argument('--max-chance-samples', type=int,
                   help='adaptive midgame chance-sample cap; requires --action-selection maximin')
    p.add_argument('--action-selection', choices=('maximin', 'equilibrium'), default='equilibrium')
    p.add_argument('--decision-timeout', type=float, default=5.0)
    p.add_argument('--search-mode', choices=('material', 'learned'), default='material')
    p.add_argument('--search-at', choices=('opening', 'midgame'), default='opening')
    p.add_argument('--state-value-checkpoint', type=Path,
                   help='closed-sheet state-value checkpoint required for learned search')
    args = p.parse_args()
    if args.random_preview == (args.selector is not None):
        p.error('provide exactly one of --selector or --random-preview')
    if args.search_workers < 1 or args.candidate_damage_samples < 1 or args.chance_samples < 1:
        p.error('--search-workers, --candidate-damage-samples, and --chance-samples must be positive')
    if args.max_chance_samples is not None:
        if args.max_chance_samples < args.chance_samples:
            p.error('--max-chance-samples must be at least --chance-samples')
        if args.max_chance_samples > args.chance_samples and (
            args.search_at != 'midgame' or args.action_selection != 'maximin'
        ):
            p.error('adaptive chance sampling requires --search-at midgame and --action-selection maximin')
    if (args.search_mode == 'learned') != (args.state_value_checkpoint is not None):
        p.error('--state-value-checkpoint is required only with --search-mode learned')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = _load_policy(args.checkpoint, 'cpu')
    evaluator = (
        load_state_value_checkpoint(args.state_value_checkpoint, map_location='cpu')
        if args.state_value_checkpoint is not None else None
    )
    if args.random_preview:
        selector, selector_config = None, None
    else:
        selector, selector_config = _load_team_selector(args.selector)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    response_candidates = (
        args.candidates if args.response_candidates is None
        else args.response_candidates
    )
    started = time.perf_counter()
    with HeadlessShowdownWorker(args.showdown_path) as worker, LocalShowdownSearchPool(
        format_id=args.format, showdown_path=args.showdown_path,
        workers=args.search_workers, timeout_seconds=args.decision_timeout,
    ) as pool:
        for cluster in range(args.start_cluster, args.start_cluster + args.clusters):
            seed = args.seed + cluster * 10007
            teams = (worker.generate_team(seed, args.format), worker.generate_team(seed + 1, args.format))
            for side in (0, 1):
                for mode in ('raw', 'opening_search'):
                    p1_strategy = args.player_strategy if side == 0 else args.opponent_strategy
                    p2_strategy = args.opponent_strategy if side == 0 else args.player_strategy
                    result = run_game(
                        game=cluster, seed=seed + (100003 if side else 0),
                        format_id=args.format, showdown_path=args.showdown_path,
                        output_path=args.out.with_suffix('.runner.jsonl'),
                        model=model, opponent_model=None,
                        player_strategy=p1_strategy,
                        opponent_strategy=p2_strategy,
                        temperature=1.0, device='cpu', telemetry=None, worker=worker,
                        trainable_side='player1' if side == 0 else 'player2',
                        counterplay_incoming_weight=args.counterplay_incoming_weight,
                        counterplay_ko_bonus=args.counterplay_ko_bonus,
                        deterministic_policy=True, teams_override=teams,
                        swap_teams=bool(side), persist_record=False,
                        team_selection_model=selector, team_selection_config=selector_config,
                        opening_search=(
                            args.search_mode
                            if mode == 'opening_search' and args.search_at == 'opening'
                            else 'none'
                        ),
                        midgame_search=(
                            args.search_mode
                            if mode == 'opening_search' and args.search_at == 'midgame'
                            else 'none'
                        ),
                        state_value_evaluator=evaluator,
                        search_hidden_worlds=args.hidden_worlds,
                        search_max_candidates=args.candidates,
                        search_max_response_candidates=response_candidates,
                        search_chance_samples=args.chance_samples,
                        search_action_selection=args.action_selection,
                        search_workers=args.search_workers,
                        search_candidate_damage_samples=args.candidate_damage_samples,
                        search_pool=pool if mode == 'opening_search' else None,
                        midgame_search_max_decisions=(
                            1 if args.search_at == 'midgame' else 0
                        ),
                        midgame_search_chance_samples=args.chance_samples,
                        midgame_search_max_chance_samples=args.max_chance_samples,
                        search_response_coverage_diagnostic=(mode == 'opening_search'),
                    )
                    player_name = 'Headless P1' if side == 0 else 'Headless P2'
                    rows.append({
                        'cluster': cluster, 'side': side, 'mode': mode,
                        'win': int(result.get('winner') == player_name),
                        'completed': bool(result.get('winner')),
                        'steps': result.get('steps'),
                        'action_adjustments': result.get('action_adjustments'),
                        'opening_search_report': result.get('opening_search_report'),
                        'midgame_search_reports': result.get('midgame_search_reports', []),
                        'elapsed_seconds': result.get('elapsed_seconds'),
                    })
            if (cluster + 1) % 8 == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({'clusters': cluster + 1, 'matches': len(rows),
                                  'matches_per_second': len(rows) / elapsed}), flush=True)
    summary = {}
    for side in (0, 1):
        groups = {m: {r['cluster']: r['win'] for r in rows if r['side'] == side and r['mode'] == m}
                  for m in ('raw', 'opening_search')}
        summary[str(side)] = {}
        for mode, values in groups.items():
            summary[str(side)][mode] = {'win_rate': float(np.mean(list(values.values())))}
        for mode in ('opening_search',):
            d = np.array([
                groups[mode][i] - groups['raw'][i]
                for i in range(args.start_cluster, args.start_cluster + args.clusters)
            ])
            rng = np.random.default_rng(args.seed + side)
            boot = d[rng.integers(0, d.size, (20000, d.size))].mean(axis=1)
            summary[str(side)][mode]['paired_delta_vs_raw'] = float(d.mean())
            summary[str(side)][mode]['paired_cluster_bootstrap_95_ci'] = np.quantile(boot, [0.025, 0.975]).tolist()
    reports = []
    for row in rows:
        if row['mode'] != 'opening_search':
            continue
        if args.search_at == 'opening':
            reports.append(row['opening_search_report'])
        elif row['midgame_search_reports']:
            reports.append(row['midgame_search_reports'][0])
    decision_latencies = [
        x.get('search_adapter_wall_ms') for x in reports
        if x and x.get('search_adapter_wall_ms') is not None
    ]
    engine_latencies = [
        x.get('elapsed_ms') for x in reports
        if x and x.get('elapsed_ms') is not None
    ]
    elapsed = time.perf_counter() - started
    report = {
        'schema_version': 1, 'format': args.format, 'checkpoint': str(args.checkpoint),
        'selector': 'random' if args.random_preview else str(args.selector),
        'opponent': args.opponent_strategy,
        'counterplay_config': {
            'incoming_weight': args.counterplay_incoming_weight,
            'ko_bonus': args.counterplay_ko_bonus,
        },
        'information_boundary': 'search receives only the acting player’s full own team, public preview, and player-visible battle state; hidden opponent information is sanitized and sampled from a closed-sheet belief',
        'seed': args.seed, 'start_cluster': args.start_cluster,
        'clusters': args.clusters, 'matches': len(rows),
        'elapsed_seconds': elapsed, 'matches_per_second': len(rows) / elapsed,
        'search_config': {'mode': args.search_mode, 'hidden_worlds': args.hidden_worlds,
                          'search_at': args.search_at,
                          'candidates': args.candidates,
                          'chance_samples': args.chance_samples,
                          'max_chance_samples': args.max_chance_samples,
                          'response_candidates': response_candidates,
                          'search_workers': args.search_workers,
                          'candidate_damage_samples': args.candidate_damage_samples,
                          'selection': args.action_selection,
                          'decision_timeout': args.decision_timeout,
                          'state_value_checkpoint': (
                              str(args.state_value_checkpoint)
                              if args.state_value_checkpoint is not None else None
                          )},
        'opening_search_latency_ms': {
            'metric': 'full search adapter wall time, including hidden-world setup, worker search, matrix selection, and action translation',
            'count': len(decision_latencies),
            'median': float(np.median(decision_latencies)) if decision_latencies else None,
            'p95': float(np.quantile(decision_latencies, .95)) if decision_latencies else None,
            'failures': sum(bool(x and x.get('error')) for x in reports),
            'completed_searches': sum(bool(x and not x.get('error')) for x in reports),
        },
        'opening_search_engine_latency_ms': {
            'metric': 'Showdown inner rollout loop only; excludes hidden-world setup and adapter work',
            'count': len(engine_latencies),
            'median': float(np.median(engine_latencies)) if engine_latencies else None,
            'p95': float(np.quantile(engine_latencies, .95)) if engine_latencies else None,
        },
        'summary': summary, 'rows': rows,
    }
    args.out.write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
