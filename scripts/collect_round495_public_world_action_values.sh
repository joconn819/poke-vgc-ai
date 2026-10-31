#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
POLICY="${POLICY_CHECKPOINT:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
OPPONENT="${OPPONENT_CHECKPOINT:-$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${TEAM_SELECTOR_CHECKPOINT:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT_DIR:-$ROOT/round495_public_world_action_values_24x4x2_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
CLUSTERS="${CLUSTERS:-24}"
WORKERS="${WORKERS:-8}"
HIDDEN_WORLDS="${HIDDEN_WORLDS:-4}"
HIDDEN_SET_PRIOR="${HIDDEN_SET_PRIOR:-showdown_random_champions}"
CHANCE_REPEATS="${CHANCE_REPEATS:-2}"

for file in "$POLICY" "$OPPONENT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing collection: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

for STEP in 1 3 5; do
  case "$STEP" in
    1) SEED=2026109501 ;;
    3) SEED=2026109503 ;;
    5) SEED=2026109505 ;;
  esac
  PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    venv/bin/python scripts/collect_midgame_action_counterfactuals_parallel.py \
      --checkpoint "$POLICY" --opponent-checkpoint "$OPPONENT" \
      --team-selection-checkpoint "$SELECTOR" \
      --output-dir "$OUT/step_${STEP}" --clusters "$CLUSTERS" --workers "$WORKERS" \
      --states-per-cluster 1 --chance-repeats "$CHANCE_REPEATS" \
      --hidden-worlds "$HIDDEN_WORLDS" --decision-step "$STEP" \
      --hidden-set-prior "$HIDDEN_SET_PRIOR" \
      --candidate-mode broad --seed "$SEED" --device cpu \
      --include-public-counterplay-response --include-random-response \
      --include-legacy-counterplay-response --showdown-path "$SHOWDOWN_PATH"
done

PYTHONPATH=. venv/bin/python scripts/merge_round488_action_value_data.py \
  "$OUT/data.json" "$OUT/step_1/data.json" "$OUT/step_3/data.json" "$OUT/step_5/data.json"
echo "Round495 public-world action-value data complete: $OUT/data.json"
