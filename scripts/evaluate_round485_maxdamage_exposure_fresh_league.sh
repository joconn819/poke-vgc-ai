#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round485_maxdamage_exposure_fresh_league_256x2_20261031"
R473_EVAL="$ROOT/round473_fresh_seed_league_256x2_20261031"
CANDIDATE="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
HISTORICAL="$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-20680473}"
PAIRS="${PAIRS:-256}"
WORKERS="${WORKERS:-8}"

for file in "$CANDIDATE" "$SELECTOR" "$HISTORICAL"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing evaluation: $OUT" >&2; exit 1; }
for opponent in max_damage counterplay public_counterplay random policy; do
  [[ -s "$R473_EVAL/round473/$opponent/paired_summary.json" ]] || {
    echo "Missing Round473 comparison artifact for $opponent" >&2; exit 1;
  }
  [[ -s "$R473_EVAL/round430/$opponent/paired_summary.json" ]] || {
    echo "Missing Round430 comparison artifact for $opponent" >&2; exit 1;
  }
done
mkdir -p "$OUT"

for opponent in max_damage counterplay public_counterplay random policy; do
  output="$OUT/round485/$opponent"
  mkdir -p "$OUT/$opponent"
  command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$CANDIDATE" --opponent "$opponent"
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

  PYTHONPATH=. venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$R473_EVAL/round473/$opponent/paired_summary.json" \
    --candidate-summary "$OUT/round485/$opponent/paired_summary.json" \
    --output "$OUT/$opponent/round485_minus_round473.json" \
    --seed "$((SEED + 1))" --bootstrap-replicates 20000
  PYTHONPATH=. venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$R473_EVAL/round430/$opponent/paired_summary.json" \
    --candidate-summary "$OUT/round485/$opponent/paired_summary.json" \
    --output "$OUT/$opponent/round485_minus_round430.json" \
    --seed "$((SEED + 2))" --bootstrap-replicates 20000
done

printf 'Round485 matched fresh-seed evaluation complete: %s\n' "$OUT"
