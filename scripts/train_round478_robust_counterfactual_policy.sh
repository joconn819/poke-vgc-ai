#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
DATA="$ROOT/round478_earlygame_robust_counterfactuals_128x4_20261031_retry1/data.json"
POLICY="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
OUT="$ROOT/round478_robust_counterfactual_policy_20261031"

[[ -s "$DATA" ]] || { echo "Missing counterfactual data: $DATA" >&2; exit 1; }
[[ -s "$POLICY" ]] || { echo "Missing policy checkpoint: $POLICY" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing output: $OUT" >&2; exit 1; }

env PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_counterfactual_policy_update.py \
    --data "$DATA" \
    --checkpoint "$POLICY" \
    --output-dir "$OUT" \
    --seed 202610479 \
    --test-fraction 0.20 \
    --epochs 12 \
    --patience 3 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --advantage-temperature 0.15 \
    --kl-coefficient 0.05 \
    --device cuda

echo "Round478 candidate policy complete: $OUT"
