#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="${OUT:-$ROOT/round582_current_policy_meta_game_128x1_20261101}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611582}"

names=(round430 round530 round531 round532 round536 round542 round574 round579)
checkpoints=(
  "$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
  "$ROOT/round530_rectangular_robust_response_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round531_iterative_robust_response_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round532_maxdamage_floor_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round536_opponent_balanced_loss_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round542_opening14_response_20k_20261101_retry1/checkpoints/latest.pt"
  "$ROOT/round574_robust_ppo_8pass_20k_20261101/checkpoints/latest.pt"
  "$ROOT/round579_robust_ppo_2x_lr_20k_20261101/checkpoints/latest.pt"
)
for file in "${checkpoints[@]}" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing matrix run: $OUT" >&2; exit 1; }
[[ "$PAIRS" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAIRS, REPEATS, and WORKERS must be positive integers" >&2; exit 2;
}
mkdir -p "$OUT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

venv/bin/python - "$OUT/policies.json" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path
root = Path('training/outputs/phase12_headless')
paths = {
    'round430': root/'round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt',
    'round530': root/'round530_rectangular_robust_response_20k_20261031/checkpoints/latest.pt',
    'round531': root/'round531_iterative_robust_response_20k_20261031/checkpoints/latest.pt',
    'round532': root/'round532_maxdamage_floor_20k_20261031/checkpoints/latest.pt',
    'round536': root/'round536_opponent_balanced_loss_20k_20261031/checkpoints/latest.pt',
    'round542': root/'round542_opening14_response_20k_20261101_retry1/checkpoints/latest.pt',
    'round574': root/'round574_robust_ppo_8pass_20k_20261101/checkpoints/latest.pt',
    'round579': root/'round579_robust_ppo_2x_lr_20k_20261101/checkpoints/latest.pt',
}
names = sys.argv[2:]
Path(sys.argv[1]).write_text(json.dumps({'names': names, 'checkpoints': {n: str(paths[n]) for n in names}}, indent=2)+'\n')
PY

run_match() {
  local row="$1" row_checkpoint="$2" opponent="$3" opponent_checkpoint="$4" output="$5"
  echo "Round582: $row vs $opponent ($PAIRS team-pair clusters)" >&2
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$row_checkpoint" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats "$REPEATS" --workers "$WORKERS" --seed "$SEED"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu --output-dir "$output")
  if [[ "$opponent" == policy ]]; then command+=(--opponent-checkpoint "$opponent_checkpoint"); fi
  "${command[@]}"
}

for ((i=0; i<${#names[@]}; i++)); do
  for ((j=i+1; j<${#names[@]}; j++)); do
    run_match "${names[$i]}" "${checkpoints[$i]}" policy "${checkpoints[$j]}" "$OUT/${names[$i]}_vs_${names[$j]}"
  done
  for opponent in max_damage counterplay public_counterplay; do
    run_match "${names[$i]}" "${checkpoints[$i]}" "$opponent" "" "$OUT/${names[$i]}_vs_${opponent}"
  done
done

venv/bin/python scripts/summarize_round582_current_policy_meta_game.py "$OUT" "$PAIRS" "$REPEATS" "$WORKERS" "$SEED"
