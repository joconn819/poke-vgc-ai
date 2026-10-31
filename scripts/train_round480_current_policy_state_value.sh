#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
RUN472="$ROOT/round472_fresh_robust_mixed_ppo_20k_20261031"
RUN473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031"
RUN476="$ROOT/round476_refreshed_population_ppo_20k_20261031"
OUT="$ROOT/round480_recent_mixed_state_value_20261031"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))

for run in "$RUN472" "$RUN473" "$RUN476"; do
  [[ -s "$run/rounds.jsonl" ]] || { echo "Missing completed rollout manifest: $run/rounds.jsonl" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing evaluator run: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES || project_bytes + RESERVE_BYTES >= total_bytes / 2 )); then
  echo "Round480 stopped: insufficient storage reserve or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_state_value.py \
    --rounds-root "$RUN472" \
    --rounds-root "$RUN473" \
    --rounds-root "$RUN476" \
    --disk-backed-cache "$OUT/cache" \
    --output-dir "$OUT/evaluator" \
    --epochs 20 --batch-size 256 --lr 1e-4 --patience 5 \
    --test-fraction 0.2 --calibration-fraction 0.15 \
    --turn-stratified-calibration --seed 202610480 --device cuda \
    2>&1 | tee "$OUT/train.log"

echo "Round480 recent mixed-policy state-value training complete: $OUT"
