#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
ROUND476_CHECKPOINT="${ROUND476_CHECKPOINT:-$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
ROUND102_CHECKPOINT="${ROUND102_CHECKPOINT:-$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
SELECTOR_CHECKPOINT="${TEAM_SELECTOR_CHECKPOINT:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT_BASE="${OUT_BASE:-$ROOT/round498_policy_ensemble_48x4x2_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
CLUSTERS="${CLUSTERS:-48}"
WORKERS="${WORKERS:-8}"
HIDDEN_WORLDS="${HIDDEN_WORLDS:-4}"
CHANCE_REPEATS="${CHANCE_REPEATS:-2}"

for file in "$POLICY_CHECKPOINT" "$ROUND476_CHECKPOINT" "$ROUND102_CHECKPOINT" "$SELECTOR_CHECKPOINT"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }

for PRIOR in showdown_random_champions uniform_unique_full_sets; do
  OUT="${OUT_BASE}_${PRIOR}"
  [[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing collection: $OUT" >&2; exit 1; }
  mkdir -p "$OUT"
  for STEP in 1 3 5; do
    case "$STEP" in
      1) SEED=2026110101 ;;
      3) SEED=2026110103 ;;
      5) SEED=2026110105 ;;
    esac
    PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      venv/bin/python scripts/collect_midgame_action_counterfactuals_parallel.py \
        --checkpoint "$POLICY_CHECKPOINT" \
        --opponent-checkpoint "$ROUND476_CHECKPOINT" \
        --additional-opponent-checkpoint "policy_round430=$POLICY_CHECKPOINT" \
        --additional-opponent-checkpoint "policy_round102=$ROUND102_CHECKPOINT" \
        --team-selection-checkpoint "$SELECTOR_CHECKPOINT" \
        --output-dir "$OUT/step_${STEP}" --clusters "$CLUSTERS" --workers "$WORKERS" \
        --states-per-cluster 1 --chance-repeats "$CHANCE_REPEATS" \
        --hidden-worlds "$HIDDEN_WORLDS" --decision-step "$STEP" \
        --hidden-set-prior "$PRIOR" --candidate-mode broad --seed "$SEED" --device cpu \
        --include-public-counterplay-response --include-random-response \
        --include-legacy-counterplay-response --showdown-path "$SHOWDOWN_PATH"
  done
  PYTHONPATH=. venv/bin/python scripts/merge_round488_action_value_data.py \
    "$OUT/data.json" "$OUT/step_1/data.json" "$OUT/step_3/data.json" "$OUT/step_5/data.json"
  echo "Round498 policy-ensemble data complete: $OUT/data.json"
done
