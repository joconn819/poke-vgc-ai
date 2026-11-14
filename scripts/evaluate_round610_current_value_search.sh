#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
POLICY="${POLICY:-$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt}"
VALUE="${VALUE:-$BASE/round609_round600h_state_value_20k_20261114/evaluator/best.pt}"
SELECTOR="${SELECTOR:-$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT:-$BASE/round610_current_value_search_64x1_20261114}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-610160}"
CLUSTERS="${CLUSTERS:-64}"
RESPONSE_COVERAGE_DIAGNOSTIC="${RESPONSE_COVERAGE_DIAGNOSTIC:-0}"
RESPONSE_SELECTION_STRATEGY="${RESPONSE_SELECTION_STRATEGY:-semantic}"
MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
MAX_RESPONSE_CANDIDATES="${MAX_RESPONSE_CANDIDATES:-4}"
MAX_DAMAGE_RESPONSE_CANDIDATES="${MAX_DAMAGE_RESPONSE_CANDIDATES:-1}"

for file in "$POLICY" "$VALUE" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing local Showdown checkout: $SHOWDOWN" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite evaluation output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

run_opponent() {
  local opponent="$1"
  local diagnostic_args=()
  if [[ "$RESPONSE_COVERAGE_DIAGNOSTIC" == "1" ]]; then
    diagnostic_args+=(--response-coverage-diagnostic)
  fi
  PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_midgame_search_side_swapped.py" \
      --checkpoint "$POLICY" \
      --value-checkpoint "$VALUE" \
      --team-selection-checkpoint "$SELECTOR" \
      --opponent-strategy "$opponent" \
      --output "$OUT/${opponent}.json" \
      --showdown-path "$SHOWDOWN" \
      --seed "$SEED" --clusters "$CLUSTERS" \
      --search-workers 2 --hidden-worlds 2 \
      --max-candidates "$MAX_CANDIDATES" --max-response-candidates "$MAX_RESPONSE_CANDIDATES" \
      --max-damage-response-candidates "$MAX_DAMAGE_RESPONSE_CANDIDATES" \
      --response-selection-strategy "$RESPONSE_SELECTION_STRATEGY" \
      --chance-samples 4 --search-selection equilibrium \
      --decision-timeout 10 "${diagnostic_args[@]}"
}

run_opponent max_damage
run_opponent public_counterplay
echo "Round610 policy-only vs one-decision current-value equilibrium search complete: $OUT"
