#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
CHECKPOINT="${CHECKPOINT:-$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt}"
PARENT="${PARENT:-$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt}"
ROUND585="${ROUND585:-$BASE/round585_public_pool_20k_20261113/checkpoints/latest.pt}"
ROUND102="${ROUND102:-$BASE/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT:-$BASE/round606_round600h_first5_transfer_league_128x1_20261114}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-606160}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-4}"

for file in "$CHECKPOINT" "$PARENT" "$ROUND585" "$ROUND102" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing local Showdown checkout: $SHOWDOWN" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

run_arm() {
  local opponent_name="$1" opponent="$2" opponent_checkpoint="${3:-}" mode="$4"
  local out="$OUT/${opponent_name}_$mode"
  local args=(--checkpoint "$CHECKPOINT" --pairs "$PAIRS" --outcome-repeats 1
    --workers "$WORKERS" --seed "$SEED" --output-dir "$out"
    --battle-format gen9championsdoubles2026mcclosed --episode-timeout "${TIMEOUT:-180}" --device cpu
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR")
  if [[ "$opponent" == policy ]]; then
    args+=(--opponent policy --opponent-checkpoint "$opponent_checkpoint")
  else
    args+=(--opponent "$opponent")
  fi
  if [[ "$mode" == first5 ]]; then
    args+=(--intervention-target evaluated_checkpoint --max-damage-action-steps 0 1 2 3 4)
  fi
  POKEMON_SHOWDOWN_PATH="$SHOWDOWN" PYTHONPATH="$ROOT" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_native_paired_parallel.py" "${args[@]}"
}

run_matchup() {
  local name="$1" opponent="$2" checkpoint="${3:-}"
  run_arm "$name" "$opponent" "$checkpoint" raw
  run_arm "$name" "$opponent" "$checkpoint" first5
  PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" \
    "$ROOT/scripts/summarize_paired_checkpoint_comparison.py" \
    --starting-summary "$OUT/${name}_raw/paired_summary.json" \
    --candidate-summary "$OUT/${name}_first5/paired_summary.json" \
    --output "$OUT/${name}_paired_comparison.json" --seed "$((SEED + 1))"
}

run_matchup max_damage max_damage
run_matchup public_counterplay public_counterplay
run_matchup random random
run_matchup round600h policy "$PARENT"
run_matchup round585 policy "$ROUND585"
run_matchup round102 policy "$ROUND102"
printf 'Round606 Round600h raw vs first-five transfer screen complete: %s\n' "$OUT"
