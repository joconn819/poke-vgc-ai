#!/usr/bin/env python3
"""Merge compatible counterfactual corpora with disjoint cluster identifiers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'refusing to overwrite {args.output}')
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    signatures = ('checkpoint', 'opponent_checkpoint', 'team_selection_checkpoint',
                  'format', 'decision_step', 'candidate_mode',
                  'paired_response_classes', 'player_strategy', 'preview_strategy')
    reference = payloads[0]
    for index, payload in enumerate(payloads[1:], start=1):
        for field in signatures:
            if payload.get(field) != reference.get(field):
                parser.error(f'incompatible dataset {args.inputs[index]}: {field} differs')
    rows = []
    clusters_total = 0
    source_info = []
    for path, payload in zip(args.inputs, payloads):
        clusters = sorted({int(row['team_cluster']) for row in payload['rows']})
        if len(clusters) != payload.get('clusters_requested'):
            parser.error(f'cluster count mismatch in {path}')
        remap = {cluster: clusters_total + offset for offset, cluster in enumerate(clusters)}
        for row in payload['rows']:
            item = dict(row)
            item['team_cluster'] = remap[int(row['team_cluster'])]
            item['source_dataset'] = str(path)
            rows.append(item)
        clusters_total += len(clusters)
        source_info.append({
            'path': str(path), 'seed': payload.get('seed'),
            'clusters': len(clusters), 'states_captured': payload.get('states_captured'),
            'chance_repeats': payload.get('chance_repeats'),
            'matches_executed': payload.get('matches_executed'),
            'elapsed_seconds': payload.get('elapsed_seconds'),
        })
    output = {
        'schema_version': 1,
        **{field: reference.get(field) for field in signatures},
        'dataset_role': 'combined train/evaluation corpus; keep chance repeats split within each state and team clusters split across train/validation/test',
        'clusters_requested': clusters_total,
        'states_captured': len({(int(row['team_cluster']), int(row['state_repeat'])) for row in rows}),
        'source_datasets': source_info,
        'rows': rows,
    }
    # Summarize actual outcome lengths; merged corpora may use different repeat
    # counts per state and consumers split chance outcomes state by state.
    output['chance_repeats_by_source'] = {}
    repeat_by_state = {}
    for row in output['rows']:
        state_key = (int(row['team_cluster']), int(row['state_repeat']))
        count = len(next(iter(row['outcomes_by_opponent'].values())))
        if state_key in repeat_by_state and repeat_by_state[state_key] != count:
            parser.error(f'inconsistent chance repeat count within state {state_key}')
        repeat_by_state[state_key] = count
    for count in repeat_by_state.values():
        group = output['chance_repeats_by_source'].setdefault(str(count), {'states': 0})
        group['states'] += 1
    output['chance_repeats_by_source'] = {
        key: {**value, 'train': int(key) // 2, 'test': int(key) - int(key) // 2}
        for key, value in output['chance_repeats_by_source'].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(',', ':')) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key != 'rows'}, indent=2))

if __name__ == '__main__':
    main()
