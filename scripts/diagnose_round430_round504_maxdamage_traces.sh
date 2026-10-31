#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round507_maxdamage_trace_diagnostic_128_20261031}"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_arm() {
  local name="$1" checkpoint="$2"
  venv/bin/python scripts/evaluate_native_paired_parallel.py \
    --checkpoint "$checkpoint" --opponent max_damage --pairs 128 \
    --outcome-repeats 1 --workers 8 --seed 202611507 \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --battle-format gen9championsdoubles2026mcclosed --device cpu \
    --trace-decisions --output-dir "$OUTPUT_ROOT/$name"
}
run_arm round430 "$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
run_arm round504 "$ROOT/round504_round476_balanced_population_ppo_20k_20261031/checkpoints/latest.pt"
