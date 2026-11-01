#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="${OUT:-$ROOT/round529_checkpoint_heuristic_payoffs_128x1_20261031}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-2026115290}"

names=(round430 round472 round473 round476 round485 round524 round102)
checkpoints=(
  "$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
  "$ROOT/round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
  "$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
)
heuristics=(max_damage counterplay public_counterplay)
for file in "${checkpoints[@]}" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing run directory: $OUT" >&2; exit 1; }
[[ "$PAIRS" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAIRS, REPEATS, and WORKERS must be positive integers" >&2; exit 2;
}
mkdir -p "$OUT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

venv/bin/python - "$OUT/policies.json" "${names[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path('training/outputs/phase12_headless')
paths={
'round430':root/'round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt',
'round472':root/'round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt',
'round473':root/'round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt',
'round476':root/'round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt',
'round485':root/'round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt',
'round524':root/'round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt',
'round102':root/'direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt',
}
Path(sys.argv[1]).write_text(json.dumps({'names':sys.argv[2:],'checkpoints':{n:str(paths[n]) for n in sys.argv[2:]}},indent=2)+'\n')
PY

for ((i=0; i<${#names[@]}; i++)); do
  for heuristic in "${heuristics[@]}"; do
    matchup="${names[$i]}_vs_$heuristic"
    echo "Evaluating $matchup ($PAIRS team clusters)" >&2
    venv/bin/python scripts/evaluate_native_paired_parallel.py \
      --checkpoint "${checkpoints[$i]}" --opponent "$heuristic" \
      --team-selection-checkpoint "$SELECTOR" \
      --opponent-team-selection-checkpoint "$SELECTOR" \
      --pairs "$PAIRS" --outcome-repeats "$REPEATS" \
      --workers "$WORKERS" --seed "$SEED" --device cpu \
      --battle-format gen9championsdoubles2026mcclosed \
      --output-dir "$OUT/$matchup"
  done
done

venv/bin/python scripts/summarize_round529_checkpoint_heuristic_payoffs.py \
  "$OUT" "$PAIRS" "$REPEATS" "$WORKERS" "$SEED"
