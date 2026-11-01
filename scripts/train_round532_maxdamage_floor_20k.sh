#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="$ROOT/round532_maxdamage_floor_20k_20261031"
START="$ROOT/round531_iterative_robust_response_20k_20261031/checkpoints/latest.pt"
POOL430="$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
POOL473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
POOL476="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
POOL485="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
POOL524="$ROOT/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="/home/justin/p/poke_learning/pokemon-showdown"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))

for file in "$START" "$POOL430" "$POOL473" "$POOL476" "$POOL485" "$POOL524" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required training artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing run directory: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round532 stopped: insufficient free space or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/policy_opponent_mixture.json" <<JSON
{
  "$OUT/inputs/start_round_0001.pt": 0.1666666667,
  "$POOL430": 0.1666666667,
  "$POOL473": 0.1666666667,
  "$POOL476": 0.1666666667,
  "$POOL485": 0.1666666667,
  "$POOL524": 0.1666666665
}
JSON

# Controlled opponent-mixture ablation from the completed Round531 policy.
# Preserve the same six-policy pool and heuristic ratio while raising the
# Max Damage training share from 11.25% to 30%.
PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.0 \
    --max-damage-fraction 0.30 \
    --counterplay-fraction 0.34873 \
    --public-counterplay-fraction 0.24477 \
    --opponent-checkpoint "$POOL430" --opponent-checkpoint "$POOL473" \
    --opponent-checkpoint "$POOL476" \
    --opponent-checkpoint "$POOL485" --opponent-checkpoint "$POOL524" \
    --opponent-pool-size 6 \
    --policy-opponent-mixture-json "$OUT/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --output-root "$OUT" --seed 202611532 \
    --epochs 4 --minibatch-size 512 --learning-rate 5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"

PYTHONPATH=. venv/bin/python scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round532 training complete: $OUT"
