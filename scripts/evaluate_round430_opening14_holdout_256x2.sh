#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${BASE:-$ROOT/training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
R476="${R476:-$ROOT/training/outputs/phase12_headless/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/training/outputs/phase12_headless/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
OUT_ROOT="${OUT_ROOT:-$ROOT/training/outputs/phase12_headless/round430_opening14_holdout_256x2_20261031}"
WORKERS="${WORKERS:-8}"
PAIRS="${PAIRS:-256}"
REPEATS="${REPEATS:-2}"
SEED_LEGACY="${SEED_LEGACY:-2026115061}"
SEED_R476="${SEED_R476:-2026115077}"

for path in "$BASE" "$R476" "$SELECTOR"; do
  [[ -s "$path" ]] || { echo "Missing required artifact: $path" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }

run_arm() {
  local matchup="$1" opponent="$2" seed="$3" mode="$4"
  local out="$OUT_ROOT/${matchup}_${mode}"
  [[ ! -e "$out" ]] || { echo "Refusing to overwrite $out" >&2; exit 1; }
  local cmd=(env PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$BASE" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats "$REPEATS" --workers "$WORKERS" --seed "$seed"
    --output-dir "$out" --team-selection-checkpoint "$SELECTOR"
    --device cuda)
  if [[ "$opponent" == policy ]]; then
    cmd+=(--opponent-checkpoint "$R476"
      --opponent-team-selection-checkpoint "$SELECTOR")
  fi
  if [[ "$mode" == opening14 ]]; then
    cmd+=(--max-damage-action-steps 1 4 --intervention-target evaluated_checkpoint)
  fi
  (cd "$ROOT" && "${cmd[@]}")
}

for pair in "legacy counterplay $SEED_LEGACY" "round476 policy $SEED_R476"; do
  read -r name opponent seed <<<"$pair"
  run_arm "$name" "$opponent" "$seed" opening14
  run_arm "$name" "$opponent" "$seed" raw
 done
