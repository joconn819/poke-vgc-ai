#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
BASE="${BASE:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
CANDIDATE="${CANDIDATE:-$ROOT/round504_round476_balanced_population_ppo_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round506_maxdamage_512x2_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-202611506}"
for file in "$BASE" "$CANDIDATE" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT"
run_arm() {
  local name="$1" checkpoint="$2"
  venv/bin/python scripts/evaluate_native_paired_parallel.py \
    --checkpoint "$checkpoint" --opponent max_damage --pairs 512 \
    --outcome-repeats 2 --workers 8 --seed "$SEED" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --battle-format gen9championsdoubles2026mcclosed --device cpu \
    --output-dir "$OUTPUT_ROOT/$name"
}
run_arm round430 "$BASE"
run_arm round504 "$CANDIDATE"
venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
  --starting-summary "$OUTPUT_ROOT/round430/paired_summary.json" \
  --candidate-summary "$OUTPUT_ROOT/round504/paired_summary.json" \
  --output "$OUTPUT_ROOT/paired_comparison.json" --seed "$SEED"
venv/bin/python - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows={}
for name in ('round430','round504'):
    d=json.loads((root/name/'paired_summary.json').read_text())
    clusters={}
    for r in d['results']:
        if r.get('complete'):
            clusters.setdefault(r['team_seed'],[]).append(r['checkpoint_points']/2)
    rows[name]={
        'score_pct': round(100*sum(map(sum,clusters.values()))/sum(map(len,clusters.values())),3),
        'clusters':len(clusters), 'complete_matches':d['games_completed'],
        'games_per_second':round(d['aggregate_games_per_second'],3),
        'timeouts_or_errors':d['timeouts_or_errors'],
        'action_adjustments':d['action_adjustments'],
        'cluster_score_means':{str(k):sum(v)/len(v) for k,v in clusters.items()},
    }
    # Reproducible cluster-bootstrap interval around absolute score.
    import numpy as np
    vals=np.asarray([np.mean(v) for v in clusters.values()])
    rng=np.random.default_rng(506430 if name=='round430' else 506504)
    boot=np.asarray([rng.choice(vals,len(vals),replace=True).mean() for _ in range(20000)])
    rows[name]['score_ci95_pct']=[round(float(x*100),3) for x in np.quantile(boot,[.025,.975])]
paired=json.loads((root/'paired_comparison.json').read_text())
payload={
 'status':'Complete' if all(x['timeouts_or_errors']==0 for x in rows.values()) else 'Complete with incomplete cases',
 'baseline':'Round430 hard-opponent auxiliary PPO',
 'candidate':'Round504 balanced-population PPO',
 'opponent':'Max Damage',
 'method':'512 fresh team-pair clusters, two outcome repeats per cluster, side-swapped; matched teams and previews. Paired and absolute confidence intervals bootstrap team-pair clusters. Showdown RNG is server-side and not identical between checkpoint arms.',
 'round430':rows['round430'], 'round504':rows['round504'],
 'paired_delta_pp':round(100*paired['candidate_minus_starting_rate'],3),
 'paired_ci95_pp':[round(100*x,3) for x in paired['difference_95_percentile_interval']],
 'matched_team_clusters':paired['matched_team_pair_clusters'],
 'matched_cases':paired['matched_cases'],
 'promotion':'Focused matchup diagnostic; interpret below-50 scores as a weakness, independently of whether this PPO update changed it.'
}
(root/'report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:{x:v[x] for x in ('score_pct','score_ci95_pct','clusters','timeouts_or_errors')} for k,v in rows.items()},indent=2))
print('paired delta',payload['paired_delta_pp'],payload['paired_ci95_pp'])
PY
