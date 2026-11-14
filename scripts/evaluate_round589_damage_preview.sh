#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="training/outputs/phase12_headless"
OUT="${OUTPUT_ROOT:-$ROOT/round589_damage_preview_vs_learned_1024x2_20261113}"
SEED="${SEED:-232116001}"
PAIRS="${PAIRS:-1024}"
REPEATS="${REPEATS:-2}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
export PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
venv/bin/python scripts/evaluate_counterplay_paired.py --player public_counterplay \
  --pairs "$PAIRS" --repeats "$REPEATS" --workers 8 --seed "$SEED" \
  --counterplay-incoming-weight 1.5 --counterplay-ko-bonus 0.35 \
  --team-selection-checkpoint "$SELECTOR" \
  --player-preview-strategy damage --opponent-preview-strategy learned --output-dir "$OUT"
