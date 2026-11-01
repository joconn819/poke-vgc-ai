#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
MATRIX="$ROOT/round582_current_policy_meta_game_128x1_20261101"
OUT="$ROOT/round583_psro_response_mix_20k_20261101_retry1"
START="$ROOT/round532_maxdamage_floor_20k_20261031/checkpoints/latest.pt"
SELECTOR="$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))
ROLLOUT_PEAK_ESTIMATE_BYTES=$((5 * 1024 * 1024 * 1024))

names=(round430 round530 round531 round532 round536 round542 round574 round579)
checkpoints=(
  "$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
  "$ROOT/round530_rectangular_robust_response_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round531_iterative_robust_response_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round532_maxdamage_floor_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round536_opponent_balanced_loss_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round542_opening14_response_20k_20261101_retry1/checkpoints/latest.pt"
  "$ROOT/round574_robust_ppo_8pass_20k_20261101/checkpoints/latest.pt"
  "$ROOT/round579_robust_ppo_2x_lr_20k_20261101/checkpoints/latest.pt"
)
for file in "$START" "$SELECTOR" "$MATRIX/report.json" "${checkpoints[@]}"; do
  [[ -s "$file" ]] || { echo "Missing required training artifact: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing run directory: $OUT" >&2; exit 1; }
project_bytes="$(du -s -B1 . | awk '{print $1}')"
free_bytes="$(df -B1 --output=avail . | tail -n 1 | tr -d ' ')"
total_bytes="$(df -B1 --output=size . | tail -n 1 | tr -d ' ')"
if (( free_bytes < RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES || project_bytes + RESERVE_BYTES + ROLLOUT_PEAK_ESTIMATE_BYTES >= total_bytes / 2 )); then
  echo "Round583 stopped: insufficient free space or project would exceed half of filesystem." >&2
  exit 1
fi

mkdir -p "$OUT/inputs"
venv/bin/python - "$MATRIX/report.json" "$OUT/inputs" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
names = sys.argv[3:]
response = report['bootstrap_equilibrium_response_mix_mean']
row = report['bootstrap_equilibrium_row_mix_mean']
# Allocate 90% to the empirically derived adversarial response mixture and
# 10% to the row-side population mixture. The checkpoint-opponent JSON stores
# their combined conditional distribution within PPO-vs-policy games.
overall = {name: 0.1*row[name] + 0.9*response[name] for name in names}
policy_mass = sum(overall.values())
if policy_mass <= 0:
    raise SystemExit('equilibrium target has no policy-opponent mass')
# train_headless_ppo.py adds its initial checkpoint as the first pool member.
# Represent that checkpoint with the exact generated start_round_0001 path and
# pass the other seven matrix members explicitly to keep the pool size at eight.
pool_paths = {name: report['checkpoints'][name] for name in names if name != 'round532'}
pool_paths['round532'] = str(out/'start_round_0001.pt')
policy_mix = {pool_paths[name]: overall[name]/policy_mass for name in names}
heuristic_fraction = {
    'max_damage': 0.9*response['max_damage'],
    'counterplay': 0.9*response['counterplay'],
    'public_counterplay': 0.9*response['public_counterplay'],
}
if abs(policy_mass + sum(heuristic_fraction.values()) - 1) > 1e-6:
    raise SystemExit('opponent mixture does not sum to one')
payload = {
    'source_matrix': sys.argv[1],
    'method': '90% bootstrap empirical dual equilibrium response mixture + 10% bootstrap row equilibrium population mixture',
    'overall_policy_opponent_fraction': policy_mass,
    'heuristic_fractions': heuristic_fraction,
    'conditional_policy_opponent_mixture': policy_mix,
    'row_mixture': row,
    'response_mixture': response,
    'start_checkpoint': 'round532, represented by inputs/start_round_0001.pt in the opponent pool',
    'target_game_restricted_to_round582_policy_population': True,
}
(out/'opponent_target.json').write_text(json.dumps(payload, indent=2)+'\n')
(out/'policy_opponent_mixture.json').write_text(json.dumps(policy_mix, indent=2)+'\n')
PY
mapfile -t HEURISTIC_FRACTIONS < <(venv/bin/python - "$OUT/inputs/opponent_target.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))['heuristic_fractions']
for name in ('max_damage','counterplay','public_counterplay'):
 print(f'{d[name]:.10f}')
PY
)

PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/train_headless_ppo.py \
    --rounds 1 --start-round 1 --episodes-per-round 20000 \
    --random-fraction 0.0 \
    --max-damage-fraction "${HEURISTIC_FRACTIONS[0]}" \
    --counterplay-fraction "${HEURISTIC_FRACTIONS[1]}" \
    --public-counterplay-fraction "${HEURISTIC_FRACTIONS[2]}" \
    --opponent-checkpoint "${checkpoints[0]}" --opponent-checkpoint "${checkpoints[1]}" \
    --opponent-checkpoint "${checkpoints[2]}" --opponent-checkpoint "${checkpoints[4]}" \
    --opponent-checkpoint "${checkpoints[5]}" --opponent-checkpoint "${checkpoints[6]}" \
    --opponent-checkpoint "${checkpoints[7]}" --opponent-pool-size 8 \
    --policy-opponent-mixture-json "$OUT/inputs/policy_opponent_mixture.json" \
    --initial-checkpoint "$START" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --output-root "$OUT" --seed 202611583 \
    --epochs 4 --minibatch-size 512 --learning-rate 5e-5 \
    --reward-mode terminal --device cuda --rollout-workers 8 \
    2>&1 | tee "$OUT/stdout.log"

PYTHONPATH=. venv/bin/python scripts/compress_rollout_manifest.py --run-dir "$OUT"
echo "Round583 PSRO best-response training complete: $OUT"
