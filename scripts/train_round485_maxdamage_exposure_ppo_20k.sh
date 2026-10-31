#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031"
START="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
POOL465="$ROOT/round465_round464_continue_pool430_464_lr5e5_20k_20261030/checkpoints/latest.pt"
POOL468="$ROOT/round468_robust_mixed_ppo_20k_20261030/checkpoints/latest.pt"
POOL471="$ROOT/round471_high_update_replay_20k_20261031/checkpoints/latest.pt"
POOL472="$ROOT/round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))

for file in "$START" "$POOL465" "$POOL468" "$POOL471" "$POOL472" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required training artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing run directory: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round485 stopped: insufficient free space or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/policy_opponent_mixture.json" <<JSON
{
  "$OUT/inputs/start_round_0001.pt": 0.20,
  "$POOL465": 0.20,
  "$POOL468": 0.20,
  "$POOL471": 0.20,
  "$POOL472": 0.20
}
JSON

# Controlled exposure ablation against Round473: same Round430 learner start,
# Round402 team selectors, five-policy pool, PPO settings, and game count.
# Change only the opponent proportions: 70% Max Damage, 10% legacy
# Counterplay, 10% Public Counterplay, and 10% policy pool. Promotion remains
# gated on fresh paired evaluation across all five opponent classes.
PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.0 --max-damage-fraction 0.70 \
    --counterplay-fraction 0.10 --public-counterplay-fraction 0.10 \
    --opponent-checkpoint "$POOL465" --opponent-checkpoint "$POOL468" \
    --opponent-checkpoint "$POOL471" --opponent-checkpoint "$POOL472" \
    --opponent-pool-size 5 \
    --policy-opponent-mixture-json "$OUT/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --output-root "$OUT" --seed 202610485 \
    --epochs 4 --minibatch-size 512 --learning-rate 5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"

PYTHONPATH=. venv/bin/python scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round485 training complete: $OUT"
