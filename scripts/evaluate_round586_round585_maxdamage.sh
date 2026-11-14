#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="training/outputs/phase12_headless"
OUT="${OUTPUT_ROOT:-$ROOT/round586_round585_maxdamage_512x2_20261113}"
SEED="${SEED:-231914001}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
BASE="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
CANDIDATE="$ROOT/round585_public_pool_20k_20261113/checkpoints/latest.pt"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
export PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for name in round430 round585; do
  checkpoint="$BASE"
  [[ "$name" != round585 ]] || checkpoint="$CANDIDATE"
  venv/bin/python scripts/evaluate_native_paired_parallel.py --checkpoint "$checkpoint" \
    --opponent max_damage --pairs 512 --outcome-repeats 2 --workers 8 --seed "$SEED" \
    --team-selection-checkpoint "$SELECTOR" --opponent-team-selection-checkpoint "$SELECTOR" \
    --battle-format gen9championsdoubles2026mcclosed --device cpu --output-dir "$OUT/$name"
done
venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
  --starting-summary "$OUT/round430/paired_summary.json" --candidate-summary "$OUT/round585/paired_summary.json" \
  --output "$OUT/paired_comparison.json" --seed "$SEED"
