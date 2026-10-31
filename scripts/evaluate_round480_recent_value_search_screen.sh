#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
RUN="$ROOT/round480_recent_mixed_state_value_20261031"
CHECKPOINT="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
OUT="${OUT:-$RUN/midgame_search_screen_32_replay.json}"

for file in "$CHECKPOINT" "$SELECTOR" "$RUN/evaluator/best.pt"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing report: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/evaluate_round219_opening_search.py \
    --checkpoint "$CHECKPOINT" --selector "$SELECTOR" \
    --showdown-path "$SHOWDOWN_PATH" --clusters 32 --seed 202610480 \
    --hidden-worlds 2 --candidates 4 --response-candidates 4 \
    --chance-samples 1 --action-selection maximin --search-mode learned \
    --search-at midgame --state-value-checkpoint "$RUN/evaluator/best.pt" \
    --out "$OUT"
