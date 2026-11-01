#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
BASE="${BASE:-$ROOT/round536_opponent_balanced_loss_20k_20261031/checkpoints/latest.pt}"
CANDIDATE="${CANDIDATE:-$ROOT/round542_opening14_response_20k_20261101_retry1/checkpoints/latest.pt}"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
R430="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
R473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
R476="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
R485="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
R524="$ROOT/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round543_round542_opening14_response_league_128x2_20261101}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611543}"
for file in "$BASE" "$CANDIDATE" "$SELECTOR" "$R430" "$R473" "$R476" "$R485" "$R524"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite output: $OUTPUT_ROOT" >&2; exit 1; }
mkdir "$OUTPUT_ROOT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_arm() {
  local matchup="$1" opponent="$2" opponent_checkpoint="$3" model="$4" seed="$5"
  local checkpoint="$BASE"
  [[ "$model" != round542 ]] || checkpoint="$CANDIDATE"
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$checkpoint" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats 2 --workers "$WORKERS" --seed "$seed"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu
    --output-dir "$OUTPUT_ROOT/$matchup/$model")
  if [[ -n "$opponent_checkpoint" ]]; then command+=(--opponent-checkpoint "$opponent_checkpoint"); fi
  "${command[@]}"
}
run_matchup() {
  local matchup="$1" opponent="$2" opponent_checkpoint="$3" seed="$4"
  echo "Evaluating $matchup (seed $seed)" >&2
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round536 "$seed"
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round542 "$seed"
  venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$OUTPUT_ROOT/$matchup/round536/paired_summary.json" \
    --candidate-summary "$OUTPUT_ROOT/$matchup/round542/paired_summary.json" \
    --output "$OUTPUT_ROOT/$matchup/paired_comparison.json" --seed "$seed"
}
run_matchup random random "" "$SEED"
run_matchup max_damage max_damage "" "$((SEED + 1))"
run_matchup counterplay counterplay "" "$((SEED + 2))"
run_matchup public_counterplay public_counterplay "" "$((SEED + 3))"
run_matchup round430 policy "$R430" "$((SEED + 4))"
run_matchup round473 policy "$R473" "$((SEED + 5))"
run_matchup round476 policy "$R476" "$((SEED + 6))"
run_matchup round524 policy "$R524" "$((SEED + 7))"
venv/bin/python - "$OUTPUT_ROOT" "$BASE" "$CANDIDATE" "$PAIRS" "$WORKERS" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows={}
for path in sorted(root.glob('*/paired_comparison.json')):
    comp=json.loads(path.read_text())
    base=json.loads((path.parent/'round536'/'paired_summary.json').read_text())
    cand=json.loads((path.parent/'round542'/'paired_summary.json').read_text())
    rows[path.parent.name]={
      'round536_score_pct':round(base['side_swapped_score_rate']*100,3),
      'round542_score_pct':round(cand['side_swapped_score_rate']*100,3),
      'paired_delta_pp':round(comp['candidate_minus_starting_rate']*100,3),
      'cluster_ci95_pp':[round(x*100,3) for x in comp['difference_95_percentile_interval']],
      'matched_team_clusters':comp['matched_team_pair_clusters'],
      'matched_cases':comp['matched_cases'],
      'round536_games_per_second':round(base['aggregate_games_per_second'],3),
      'round542_games_per_second':round(cand['aggregate_games_per_second'],3),
      'round536_errors':base['timeouts_or_errors'],'round542_errors':cand['timeouts_or_errors'],
      'round536_action_adjustments':base['action_adjustments'],'round542_action_adjustments':cand['action_adjustments']}
if len(rows)!=8: raise SystemExit(f'expected 8 matchup reports, found {len(rows)}')
payload={'round':543,'status':'complete_screen_no_promotion','candidate':'Round542 PPO trained against frozen policy opponents using Max Damage actions at decisions 1 and 4','baseline':'Round536 opponent-balanced-loss PPO trained with the same rollout distribution and optimizer, without the opponent overlay','method':'128 fresh team-pair clusters per opponent; two outcome repeats; side-swapped; identical seed, team selector and opponent selector between candidate and baseline. Paired intervals bootstrap team-pair clusters.','format':'gen9championsdoubles2026mcclosed','clusters_per_opponent':int(sys.argv[4]),'outcome_repeats':2,'workers':int(sys.argv[5]),'base_checkpoint':sys.argv[2],'candidate_checkpoint':sys.argv[3],'opponents':rows,'promotion':'Screen only. Require independent confirmation and lower-tail review before promotion.','reproduction':'scripts/evaluate_round542_opening14_response_128x2.sh'}
(root/'report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:{'raw':v['round536_score_pct'],'candidate':v['round542_score_pct'],'delta':v['paired_delta_pp'],'ci':v['cluster_ci95_pp']} for k,v in rows.items()},indent=2))
PY
