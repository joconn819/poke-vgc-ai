#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/training/outputs/phase12_headless/round602f_round600h_opening_teacher_w1_20k_20261114"
DATA="$RUN/rollouts"
START="$ROOT/training/outputs/phase12_headless/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt"
OUT="$ROOT/training/outputs/phase12_headless/round604_onepass_imitation_from_round602f_20261114"
mapfile -t TRAJECTORIES < <(find "$DATA" -maxdepth 1 -type f -name '*.jsonl.gz' | sort)
[[ ${#TRAJECTORIES[@]} -eq 16 ]] || { echo "Expected 16 audited rollout shards, found ${#TRAJECTORIES[@]}" >&2; exit 1; }
[[ -s "$START" ]] || { echo "Missing Round600h starting checkpoint" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 1; }
mkdir -p "$OUT/checkpoints"
ARGS=()
for trajectory in "${TRAJECTORIES[@]}"; do ARGS+=(--trajectory-path "$trajectory"); done
cd "$ROOT"
PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$ROOT/venv/bin/python" scripts/train_native_rl.py \
    "${ARGS[@]}" \
    --checkpoint-dir "$OUT/checkpoints" \
    --init-from-checkpoint "$START" \
    --updates 1 --epochs 1 --minibatch-size 512 --lr 1e-5 \
    --training-team trainable --reward-mode terminal --device cuda \
    --seed 231114604 --run-id round604-onepass-imitation-from-round602f \
    --teacher-loss-weight 1.0 --teacher-imitation-only \
    --entropy-weight 0 --win-probability-loss-weight 0 --hidden-set-loss-weight 0 \
    --off-policy-behavior-log-probs \
    --telemetry-path "$OUT/telemetry.jsonl" \
    2>&1 | tee "$OUT/stdout.log"
[[ -s "$OUT/checkpoints/update_000001.pt" ]] || { echo "Round604 did not write its checkpoint" >&2; exit 1; }
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" - "$OUT/checkpoints/update_000001.pt" "$OUT/checkpoints/latest.pt" <<'PY'
import os, sys, tempfile
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
fd, tmp = tempfile.mkstemp(prefix=".latest.", dir=dst.parent)
os.close(fd)
try:
    Path(tmp).write_bytes(src.read_bytes())
    os.replace(tmp, dst)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
PY
echo "Round604 imitation-only checkpoint complete: $OUT/checkpoints/latest.pt"
