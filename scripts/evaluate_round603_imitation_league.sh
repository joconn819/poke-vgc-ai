#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
CHECKPOINT="${CHECKPOINT:-$BASE/round603_round600h_opening_teacher_w1_20k_20261114/checkpoints/latest.pt}"
PARENT="${PARENT:-$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt}"
ROUND585="${ROUND585:-$BASE/round585_public_pool_20k_20261113/checkpoints/latest.pt}"
ROUND102="${ROUND102:-$BASE/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT:-$BASE/round603_fresh_league_screen_128x1_20261114}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-603160}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-4}"

for file in "$CHECKPOINT" "$PARENT" "$ROUND585" "$ROUND102" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

run_cell() {
  local name="$1" opponent="$2" opponent_checkpoint="${3:-}"
  local args=(--checkpoint "$CHECKPOINT" --pairs "$PAIRS" --outcome-repeats 1
    --workers "$WORKERS" --seed "$SEED" --output-dir "$OUT/$name"
    --battle-format gen9championsdoubles2026mcclosed --device cpu
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR")
  if [[ "$opponent" == policy ]]; then
    args+=(--opponent policy --opponent-checkpoint "$opponent_checkpoint")
  else
    args+=(--opponent "$opponent")
  fi
  POKEMON_SHOWDOWN_PATH="$SHOWDOWN" PYTHONPATH="$ROOT" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_native_paired_parallel.py" "${args[@]}"
}

run_cell vs_max_damage max_damage
run_cell vs_public_counterplay public_counterplay
run_cell vs_random random
run_cell vs_round600h policy "$PARENT"
run_cell vs_round585 policy "$ROUND585"
run_cell vs_round102 policy "$ROUND102"
printf 'Round603 fresh league screen complete: %s\n' "$OUT"
