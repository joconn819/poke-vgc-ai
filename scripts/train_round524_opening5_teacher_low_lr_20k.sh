#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round524_opening5_teacher_low_lr_20k_20261031"
START="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
POOL465="$ROOT/round465_round464_continue_pool430_464_lr5e5_20k_20261030/checkpoints/latest.pt"
POOL472="$ROOT/round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt"
POOL473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
POOL102="$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))

for file in "$START" "$POOL465" "$POOL472" "$POOL473" "$POOL102" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || {
  echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2
  exit 1
}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || \
      project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round524 stopped: insufficient reserve or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/policy_opponent_mixture.json" <<JSON
{
  "$OUT/inputs/start_round_0001.pt": 0.20,
  "$POOL465": 0.20,
  "$POOL472": 0.20,
  "$POOL473": 0.20,
  "$POOL102": 0.20
}
JSON

# Fresh held-out evidence shows that replacing the first five policy actions
# with Max Damage choices Round520 reached the KL stop after only 177 of 1,296 planned optimizer steps.
# Hold the data mixture, teacher-label rate and loss weight fixed; lower only the
# learning rate to test whether a full optimization pass improves the checkpoint.
# Teacher-forced actions are excluded from PPO's policy-gradient loss.
PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.10 --max-damage-fraction 0.20 \
    --counterplay-fraction 0.25 --public-counterplay-fraction 0.25 \
    --opponent-checkpoint "$POOL465" --opponent-checkpoint "$POOL472" \
    --opponent-checkpoint "$POOL473" --opponent-checkpoint "$POOL102" \
    --opponent-pool-size 5 \
    --policy-opponent-mixture-json "$OUT/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --first-turn-max-damage-teacher-fraction 0.50 \
    --max-damage-teacher-steps 1 2 3 4 5 \
    --teacher-loss-weight 0.50 --exclude-teacher-actions-from-ppo \
    --output-root "$OUT" --seed 202611524 \
    --epochs 4 --minibatch-size 512 --learning-rate 1.5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"

PYTHONPATH=. venv/bin/python scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round524 training complete: $OUT"
