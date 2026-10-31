#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
POLICY="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
ACTOR="$ROOT/round482_absolute_joint_regret_actor_384states_20261031_fullaction/counterfactual_policy.pt"
OPPONENT="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
OUT="$ROOT/round484_public_preview_actor_counterfactuals_128x8_20261031"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"

for file in "$POLICY" "$ACTOR" "$OPPONENT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing collection: $OUT" >&2; exit 1; }

PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/collect_midgame_action_counterfactuals_parallel.py \
    --checkpoint "$POLICY" \
    --candidate-policy-checkpoint "$ACTOR" \
    --opponent-checkpoint "$OPPONENT" \
    --team-selection-checkpoint "$SELECTOR" \
    --output-dir "$OUT" --clusters 128 --workers 4 \
    --states-per-cluster 1 --chance-repeats 8 --decision-step 1 \
    --candidate-mode broad --seed 202610485 --device cpu \
    --include-public-counterplay-response --showdown-path "$SHOWDOWN_PATH"

echo "Round484 public-preview counterfactual collection complete: $OUT"
