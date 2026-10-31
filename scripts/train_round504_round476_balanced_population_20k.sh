#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round504_round476_balanced_population_ppo_20k_20261031"
START="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
POOL430="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
POOL473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
POOL485="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
POOL102="$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))

for file in "$START" "$POOL430" "$POOL473" "$POOL485" "$POOL102" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required training artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing run directory: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round504 stopped: insufficient free space or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/policy_opponent_mixture.json" <<JSON
{
  "$OUT/inputs/start_round_0001.pt": 0.20,
  "$POOL430": 0.20,
  "$POOL473": 0.20,
  "$POOL485": 0.20,
  "$POOL102": 0.20
}
JSON

# Continue from Round476, evenly sample five frozen PPO policies within the
# policy-response portion, and reserve 70% of games for four heuristic classes.
# No one response class or checkpoint dominates the training distribution.
PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.10 --max-damage-fraction 0.20 \
    --counterplay-fraction 0.20 --public-counterplay-fraction 0.20 \
    --opponent-checkpoint "$POOL430" --opponent-checkpoint "$POOL473" \
    --opponent-checkpoint "$POOL485" --opponent-checkpoint "$POOL102" \
    --opponent-pool-size 5 \
    --policy-opponent-mixture-json "$OUT/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --output-root "$OUT" --seed 202611504 \
    --epochs 4 --minibatch-size 512 --learning-rate 5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"

PYTHONPATH=. venv/bin/python scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round504 training complete: $OUT"
