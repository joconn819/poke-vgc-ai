#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
DATA="$ROOT/round482_combined_counterfactual_data_384states_20261031/data.json"
POLICY="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
OUT="$ROOT/round482_regret_matching_policy_384states_20261031"

[[ -s "$DATA" ]] || { echo "Missing merged counterfactual data: $DATA" >&2; exit 1; }
[[ -s "$POLICY" ]] || { echo "Missing source checkpoint: $POLICY" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing update: $OUT" >&2; exit 1; }

PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_counterfactual_policy_update.py \
    --data "$DATA" --checkpoint "$POLICY" --output-dir "$OUT" \
    --seed 202610482 --test-fraction 0.2 --epochs 12 --patience 3 \
    --batch-size 16 --learning-rate 1e-5 --kl-coefficient 0.05 \
    --target-objective regret_matching --regret-matching-iterations 5000 \
    --device cuda

echo "Round482 regret-matching candidate complete: $OUT"
