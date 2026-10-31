#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
POLICY="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
ACTOR="$ROOT/round482_absolute_joint_regret_actor_384states_20261031_fullaction/counterfactual_policy.pt"
OPPONENT="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
OUT="$ROOT/round489_wide_response_action_value_128x8_20261031"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"

for file in "$POLICY" "$ACTOR" "$OPPONENT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing collection: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

for STEP in 1 3 5; do
  case "$STEP" in
    1) SEED=202610501 ;;
    3) SEED=202610503 ;;
    5) SEED=202610505 ;;
  esac
  PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    venv/bin/python scripts/collect_midgame_action_counterfactuals_parallel.py \
      --checkpoint "$POLICY" \
      --candidate-policy-checkpoint "$ACTOR" \
      --opponent-checkpoint "$OPPONENT" \
      --team-selection-checkpoint "$SELECTOR" \
      --output-dir "$OUT/step_${STEP}" --clusters 128 --workers 8 \
      --states-per-cluster 1 --chance-repeats 8 --decision-step "$STEP" \
      --candidate-mode broad --seed "$SEED" --device cpu \
      --include-public-counterplay-response --include-random-response \
      --include-legacy-counterplay-response --showdown-path "$SHOWDOWN_PATH"
done

PYTHONPATH=. venv/bin/python scripts/merge_round488_action_value_data.py \
  "$OUT/data.json" "$OUT/step_1/data.json" "$OUT/step_3/data.json" "$OUT/step_5/data.json"
echo "Round489 wide-response action-value data complete: $OUT/data.json"
