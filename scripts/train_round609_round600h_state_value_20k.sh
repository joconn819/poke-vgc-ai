#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
RUN="$BASE/round600h_round585_competitive_team_pool_20k_20261113"
OUT="$BASE/round609_round600h_state_value_20k_20261114"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))

[[ -s "$RUN/rounds.jsonl" ]] || { echo "Missing completed Round600h rollout manifest: $RUN/rounds.jsonl" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite evaluator run: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 "$ROOT" | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size "$ROOT" | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES || project_bytes + RESERVE_BYTES >= total_bytes / 2 )); then
  echo "Round609 stopped: insufficient storage reserve or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$ROOT/venv/bin/python" "$ROOT/scripts/train_state_value.py" \
    --rounds-root "$RUN" \
    --disk-backed-cache "$OUT/cache" \
    --output-dir "$OUT/evaluator" \
    --epochs 20 --batch-size 256 --lr 1e-4 --patience 5 \
    --test-fraction 0.2 --calibration-fraction 0.15 \
    --turn-stratified-calibration --seed 202611609 --device cuda \
    2>&1 | tee "$OUT/train.log"

echo "Round609 current-policy state-value training complete: $OUT"
