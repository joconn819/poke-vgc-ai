#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/training/outputs/phase12_headless/round540_round430_opening14_nineclass_128x1_20261101}"
BASE="${BASE:-$ROOT/training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/training/outputs/phase12_headless/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
R430="$BASE"
R473="$ROOT/training/outputs/phase12_headless/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
R476="$ROOT/training/outputs/phase12_headless/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
R485="$ROOT/training/outputs/phase12_headless/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
R524="$ROOT/training/outputs/phase12_headless/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611540}"

for path in "$BASE" "$SELECTOR" "$R473" "$R476" "$R485" "$R524"; do
  [[ -s "$path" ]] || { echo "Missing required artifact: $path" >&2; exit 1; }
done
[[ -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_arm() {
  local opponent_name="$1" strategy="$2" opponent_checkpoint="$3" seed="$4" arm="$5"
  local directory="$OUT/$opponent_name/$arm"
  mkdir -p "$directory"
  local args=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$BASE" --opponent "$strategy" --pairs "$PAIRS"
    --outcome-repeats 1 --workers "$WORKERS" --seed "$seed"
    --output-dir "$directory" --team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu)
  if [[ -n "$opponent_checkpoint" ]]; then
    args+=(--opponent-checkpoint "$opponent_checkpoint"
      --opponent-team-selection-checkpoint "$SELECTOR")
  fi
  if [[ "$arm" == opening14 ]]; then
    args+=(--max-damage-action-steps 1 4 --intervention-target evaluated_checkpoint)
  fi
  (cd "$ROOT" && "${args[@]}")
}

run_matchup() {
  local name="$1" strategy="$2" checkpoint="$3" seed="$4"
  echo "Round540 $name paired arms (seed $seed)" >&2
  run_arm "$name" "$strategy" "$checkpoint" "$seed" raw
  run_arm "$name" "$strategy" "$checkpoint" "$seed" opening14
  venv/bin/python "$ROOT/scripts/summarize_paired_checkpoint_comparison.py" \
    --starting-summary "$OUT/$name/raw/paired_summary.json" \
    --candidate-summary "$OUT/$name/opening14/paired_summary.json" \
    --output "$OUT/$name/paired_comparison.json" --seed "$seed"
}

run_matchup random random "" "$SEED"
run_matchup max_damage max_damage "" "$((SEED+1))"
run_matchup counterplay counterplay "" "$((SEED+2))"
run_matchup public_counterplay public_counterplay "" "$((SEED+3))"
run_matchup round430 policy "$R430" "$((SEED+4))"
run_matchup round473 policy "$R473" "$((SEED+5))"
run_matchup round476 policy "$R476" "$((SEED+6))"
run_matchup round485 policy "$R485" "$((SEED+7))"
run_matchup round524 policy "$R524" "$((SEED+8))"

venv/bin/python - "$OUT" "$BASE" "$PAIRS" "$WORKERS" "$SEED" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows={}
for path in sorted(root.glob('*/paired_comparison.json')):
    comp=json.loads(path.read_text())
    baseline=json.loads((path.parent/'raw'/'paired_summary.json').read_text())
    candidate=json.loads((path.parent/'opening14'/'paired_summary.json').read_text())
    rows[path.parent.name]={
      'raw_score_pct':round(baseline['side_swapped_score_rate']*100,3),
      'opening14_score_pct':round(candidate['side_swapped_score_rate']*100,3),
      'paired_delta_pp':round(comp['candidate_minus_starting_rate']*100,3),
      'paired_cluster_ci95_pp':[round(x*100,3) for x in comp['difference_95_percentile_interval']],
      'matched_clusters':comp['matched_team_pair_clusters'],'matched_cases':comp['matched_cases'],
      'raw_games_per_second':round(baseline['aggregate_games_per_second'],3),
      'opening14_games_per_second':round(candidate['aggregate_games_per_second'],3),
      'raw_timeouts_or_errors':baseline['timeouts_or_errors'],
      'opening14_timeouts_or_errors':candidate['timeouts_or_errors'],
      'raw_action_adjustments':baseline['action_adjustments'],
      'opening14_action_adjustments':candidate['action_adjustments'],
    }
if len(rows)!=9: raise SystemExit(f'expected 9 matchups, completed {len(rows)}')
report={'round':540,'status':'complete_screening_no_promotion','candidate':'Round430 with Max Damage actions at decision indices 1 and 4','baseline':'Raw Round430','method':'128 fresh team-pair clusters per opponent, one outcome repeat per cluster, side-swapped, paired seeds/teams/previews/selectors. Paired intervals bootstrap team-pair clusters; Showdown RNG is server-side and not identical across arms.','format':'gen9championsdoubles2026mcclosed','opponents':rows,'pairs_per_opponent':int(sys.argv[3]),'workers':int(sys.argv[4]),'seed_base':int(sys.argv[5]),'reproduction':'scripts/evaluate_round540_opening14_nineclass_128x1.sh','promotion':'Screen only. Do not promote without a fresh repeated confirmation and operational integration checks.'}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{'raw':v['raw_score_pct'],'opening14':v['opening14_score_pct'],'delta':v['paired_delta_pp'],'ci':v['paired_cluster_ci95_pp']} for k,v in rows.items()},indent=2))
PY
