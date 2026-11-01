#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
DATA="$ROOT/round536_opponent_balanced_loss_20k_20261031"
OUT="$ROOT/round538_opponent_group_norm_replay_20261031_retry1"
START="$DATA/inputs/start_round_0001.pt"
[[ -s "$START" ]] || { echo "Missing initial checkpoint: $START" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 1; }
mkdir -p "$OUT/preflight" "$OUT/checkpoints/round_0001"
mapfile -t ROLLOUTS < <(find "$DATA/rollouts" -maxdepth 1 -type f -name '*.jsonl.gz' -print | sort)
[[ ${#ROLLOUTS[@]} -eq 22 ]] || { echo "Expected 22 compressed rollout shards, found ${#ROLLOUTS[@]}" >&2; exit 1; }

AUDIT=(env PYTHONPATH=. venv/bin/python scripts/audit_native_ppo_rollouts.py)
for path in "${ROLLOUTS[@]}"; do AUDIT+=(--trajectory-path "$path"); done
"${AUDIT[@]}" \
  --expect-games 20000 \
  --expect-opponent-count max_damage=6000 \
  --expect-opponent-count counterplay=6975 \
  --expect-opponent-count public_counterplay=4895 \
  --expect-opponent-count policy=2130 \
  --report "$OUT/preflight/round_0001.json"

TRAIN=(env PYTHONPATH=. venv/bin/python scripts/train_native_rl.py)
for path in "${ROLLOUTS[@]}"; do TRAIN+=(--trajectory-path "$path"); done
"${TRAIN[@]}" \
  --preflight-report "$OUT/preflight/round_0001.json" \
  --checkpoint-dir "$OUT/checkpoints/round_0001" \
  --init-from-checkpoint "$START" \
  --updates 1 --epochs 4 --minibatch-size 512 --lr 5e-5 \
  --training-team trainable --entropy-weight 0.01 --reward-mode terminal \
  --advantage-normalization opponent \
  --opponent-balanced-loss --opponent-balance-power 1.0 \
  --max-damage-loss-weight 1.0 \
  --native-action-features --native-action-feature-variant damage_ko_friendly_fire \
  --device cuda --seed 202611533 --run-id round538-opponent-group-normalization \
  --telemetry-path "$OUT/telemetry.jsonl"
