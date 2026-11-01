#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
CHECKPOINT="${CHECKPOINT:-$ROOT/round520_opening5_teacher_population_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT:-$ROOT/round523_round520_opening_traces_128x2_20261031}"
SEED="${SEED:-202611528}"
WORKERS="${WORKERS:-8}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"

for file in "$CHECKPOINT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing trace run: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

common=(--checkpoint "$CHECKPOINT" --opponent max_damage
  --team-selection-checkpoint "$SELECTOR"
  --opponent-team-selection-checkpoint "$SELECTOR"
  --pairs 128 --outcome-repeats 2 --workers "$WORKERS" --seed "$SEED"
  --battle-format gen9championsdoubles2026mcclosed --device cpu --trace-decisions)
for mode in raw first5; do
  args=()
  if [[ "$mode" == first5 ]]; then
    args+=(--intervention-target evaluated_checkpoint
      --max-damage-action-steps 0 1 2 3 4)
  fi
  PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    venv/bin/python scripts/evaluate_native_paired_parallel.py \
      "${common[@]}" "${args[@]}" --output-dir "$OUT/$mode"
  PYTHONPATH=. venv/bin/python scripts/analyze_action_loss_traces.py \
    --input-dir "$OUT/$mode" \
    --output-json "$OUT/${mode}_trace_analysis.json" \
    --output-md "$OUT/${mode}_trace_analysis.md" --seed "$((SEED + 1))"
done
venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
  --starting-summary "$OUT/raw/paired_summary.json" \
  --candidate-summary "$OUT/first5/paired_summary.json" \
  --output "$OUT/paired_comparison.json" --seed "$((SEED + 1))"
venv/bin/python - "$OUT" "$SEED" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
raw=json.loads((root/'raw/paired_summary.json').read_text())
first5=json.loads((root/'first5/paired_summary.json').read_text())
paired=json.loads((root/'paired_comparison.json').read_text())
report={
  'status':'Complete; traced first-five intervention replication',
  'method':'128 fresh team-pair clusters, two repeats, side-swapped; paired cluster bootstrap; traces enabled.',
  'raw_score_pct':round(raw['side_swapped_score_rate']*100,3),
  'first5_score_pct':round(first5['side_swapped_score_rate']*100,3),
  'paired_delta_pp':round(paired['candidate_minus_starting_rate']*100,3),
  'paired_ci95_pp':[round(v*100,3) for v in paired['difference_95_percentile_interval']],
  'clusters':paired['matched_team_pair_clusters'],
  'raw_games_per_second':round(raw['aggregate_games_per_second'],3),
  'first5_games_per_second':round(first5['aggregate_games_per_second'],3),
  'raw_timeouts_or_errors':raw['timeouts_or_errors'],
  'first5_timeouts_or_errors':first5['timeouts_or_errors'],
  'seed':int(sys.argv[2]),
  'trace_analysis_raw':'raw_trace_analysis.json',
  'trace_analysis_first5':'first5_trace_analysis.json',
}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(report,indent=2))
PY
