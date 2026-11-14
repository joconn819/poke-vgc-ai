#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
OUT="$BASE/round600h_loss_trace_diagnostic_12x1_20261114_rerun"
CHECKPOINT="$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt"
SELECTOR="$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
[[ -s "$CHECKPOINT" && -s "$SELECTOR" && -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing Round600h checkpoint, preview selector, or local Showdown checkout" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
for opponent in max_damage public_counterplay; do
  POKEMON_SHOWDOWN_PATH="$SHOWDOWN" PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_counterplay_paired.py" \
    --pairs 12 --repeats 1 --workers 4 --seed 600900 --player policy \
    --opponent "$opponent" --checkpoint "$CHECKPOINT" --balance-team-assignment \
    --episode-timeout 180 --team-selection-checkpoint "$SELECTOR" \
    --player-preview-strategy learned --opponent-preview-strategy learned \
    --trace-decisions --output-dir "$OUT/vs_$opponent"
done
printf 'Round600h loss traces: %s\n' "$OUT"
