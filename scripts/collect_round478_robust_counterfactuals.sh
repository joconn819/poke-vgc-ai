#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
POLICY="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
OPPONENT="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
OUT="$ROOT/round478_earlygame_robust_counterfactuals_128x4_20261031_retry1"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"

for file in "$POLICY" "$OPPONENT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing output: $OUT" >&2; exit 1; }

env PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/collect_midgame_action_counterfactuals_parallel.py \
    --checkpoint "$POLICY" \
    --opponent-checkpoint "$OPPONENT" \
    --team-selection-checkpoint "$SELECTOR" \
    --include-public-counterplay-response \
    --output-dir "$OUT" \
    --clusters 128 \
    --workers 4 \
    --seed 202610478 \
    --states-per-cluster 1 \
    --chance-repeats 4 \
    --decision-step 1 \
    --candidate-mode broad \
    --device cpu \
    --showdown-path "$SHOWDOWN_PATH"

echo "Round478 robust counterfactual data complete: $OUT"
