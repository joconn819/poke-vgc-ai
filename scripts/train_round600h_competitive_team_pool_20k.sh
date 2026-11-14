#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/training/outputs/phase12_headless/round600h_round585_competitive_team_pool_20k_20261113"
BASE="$ROOT/training/outputs/phase12_headless"
START="$BASE/round585_public_pool_20k_20261113/checkpoints/latest.pt"
POOL473="$BASE/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
POOL476="$BASE/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
POOL485="$BASE/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
POOL524="$BASE/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
SELECTOR="$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
MANIFEST="$BASE/round595_competitive_event_splits_top256_20261113/team_pair_manifest.json"
TEAM_POOL="$ROOT/training/inputs/round595_development_team_pool.json"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))
for file in "$START" "$POOL473" "$POOL476" "$POOL485" "$POOL524" "$SELECTOR" "$MANIFEST"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 "$ROOT" | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size "$ROOT" | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round600 stopped: insufficient free space or project would exceed half the filesystem." >&2; exit 1
fi
mkdir -p "$OUT" "$(dirname "$TEAM_POOL")"
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" "$ROOT/scripts/build_competitive_training_pool.py" --manifest "$MANIFEST" --output "$TEAM_POOL" | tee "$OUT/team_pool_manifest.json"
cat > "$OUT/policy_opponent_mixture.json" <<JSON
{
  "$OUT/inputs/start_round_0001.pt": 0.20,
  "$POOL473": 0.20,
  "$POOL476": 0.20,
  "$POOL485": 0.20,
  "$POOL524": 0.20
}
JSON
# Keep the Round585 opponent mixture and PPO settings fixed. Change only the
# starting checkpoint and the rollout team distribution: half tournament teams,
# half local RandomChampions teams. Final-event rosters are excluded.
cd "$ROOT"
PYTHONPATH="$ROOT" POKEMON_SHOWDOWN_PATH="$SHOWDOWN" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$ROOT/venv/bin/python" scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.0 --max-damage-fraction 0.50 \
    --counterplay-fraction 0.0 --public-counterplay-fraction 0.25 \
    --opponent-checkpoint "$POOL473" --opponent-checkpoint "$POOL476" \
    --opponent-checkpoint "$POOL485" --opponent-checkpoint "$POOL524" \
    --opponent-pool-size 5 --policy-opponent-mixture-json "$OUT/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --competitive-team-pool-json "$TEAM_POOL" --competitive-team-fraction 0.50 \
    --output-root "$OUT" --seed 231113600 \
    --epochs 4 --minibatch-size 512 --learning-rate 5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round600 training complete: $OUT"
