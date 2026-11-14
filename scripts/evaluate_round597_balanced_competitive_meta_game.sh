#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${MANIFEST:-$ROOT/training/outputs/phase12_headless/round595_competitive_event_splits_top256_20261113/team_pair_manifest.json}"
SELECTOR="${SELECTOR:-$ROOT/training/outputs/phase12_headless/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
ROUND430="${ROUND430:-$ROOT/training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
OUT="${OUT:-$ROOT/training/outputs/phase12_headless/round597_balanced_competitive_meta_game_35x2_20261113}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-35}"
REPEATS="${REPEATS:-2}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-597100}"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

run_cell() {
  local name="$1" player="$2" opponent="$3"
  shift 3
  local extra=("$@")
  PYTHONPATH="$ROOT" POKEMON_SHOWDOWN_PATH="$SHOWDOWN" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_counterplay_paired.py" \
      --pairs "$PAIRS" --repeats "$REPEATS" --workers "$WORKERS" \
      --seed "$SEED" --player "$player" --opponent "$opponent" \
      --balance-team-assignment \
      --team-selection-checkpoint "$SELECTOR" \
      --player-preview-strategy learned --opponent-preview-strategy learned \
      --team-pair-manifest "$MANIFEST" --team-pair-split development \
      --output-dir "$OUT/$name" "${extra[@]}"
}

run_cell round430_vs_round430 policy policy \
  --checkpoint "$ROUND430" --opponent-checkpoint "$ROUND430"
run_cell round430_vs_public_counterplay policy public_counterplay \
  --checkpoint "$ROUND430"
run_cell round430_vs_max_damage policy max_damage \
  --checkpoint "$ROUND430"
run_cell round430_vs_random policy random \
  --checkpoint "$ROUND430"
run_cell public_counterplay_vs_public_counterplay public_counterplay public_counterplay
run_cell public_counterplay_vs_max_damage public_counterplay max_damage
run_cell public_counterplay_vs_random public_counterplay random
run_cell max_damage_vs_max_damage max_damage max_damage
run_cell max_damage_vs_random max_damage random
run_cell random_vs_random random random

echo "All Round597 competitive matrix cells completed. Summaries: $OUT/*/paired_summary.json"
