#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round479_robust_counterfactual_screen_128x2_20261031"
BASE="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
CANDIDATE="$ROOT/round479_robust_counterfactual_policy_20261031/counterfactual_policy.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
HISTORICAL="$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-20680479}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-8}"

for file in "$BASE" "$CANDIDATE" "$SELECTOR" "$HISTORICAL"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing evaluation: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

run_arm() {
  local policy_name="$1" checkpoint="$2" opponent="$3"
  local output="$OUT/$policy_name/$opponent"
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$checkpoint" --opponent "$opponent"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --pairs "$PAIRS" --outcome-repeats 2 --workers "$WORKERS"
    --seed "$SEED" --device cpu --output-dir "$output")
  if [[ "$opponent" == "policy" ]]; then
    command+=(--opponent-checkpoint "$HISTORICAL")
  fi
  PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "${command[@]}"
}

opponents=(max_damage public_counterplay policy)
for opponent in "${opponents[@]}"; do
  run_arm round430 "$BASE" "$opponent"
done
for opponent in "${opponents[@]}"; do
  run_arm round479 "$CANDIDATE" "$opponent"
done

index=0
for opponent in "${opponents[@]}"; do
  PYTHONPATH=. venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$OUT/round430/$opponent/paired_summary.json" \
    --candidate-summary "$OUT/round479/$opponent/paired_summary.json" \
    --output "$OUT/$opponent/round479_minus_round430.json" \
    --seed "$((SEED + 1 + index))" --bootstrap-replicates 20000
  index=$((index + 1))
done

printf 'Round479 robust counterfactual screen complete: %s\n' "$OUT"
