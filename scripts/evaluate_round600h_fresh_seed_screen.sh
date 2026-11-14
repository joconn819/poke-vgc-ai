#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/training/outputs/phase12_headless"
ROUND600H="${ROUND600H:-$BASE/round600h_round585_competitive_team_pool_20k_20261113/checkpoints/latest.pt}"
ROUND585="${ROUND585:-$BASE/round585_public_pool_20k_20261113/checkpoints/latest.pt}"
ROUND102="${ROUND102:-$BASE/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$BASE/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUT="${OUT:-$BASE/round600h_fresh_randomteam_screen_128x1_20261114}"
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-600800}"

for file in "$ROUND600H" "$ROUND585" "$ROUND102" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required artifact: $file" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

run_cell() {
  local name="$1" opponent="$2" opponent_checkpoint="${3:-}"
  local args=(--pairs "$PAIRS" --repeats 1 --workers "$WORKERS" --seed "$SEED"
    --player policy --opponent "$opponent" --checkpoint "$ROUND600H"
    --balance-team-assignment --episode-timeout 180
    --team-selection-checkpoint "$SELECTOR"
    --player-preview-strategy learned --opponent-preview-strategy learned
    --output-dir "$OUT/$name")
  if [[ -n "$opponent_checkpoint" ]]; then args+=(--opponent-checkpoint "$opponent_checkpoint"); fi
  PYTHONPATH="$ROOT" POKEMON_SHOWDOWN_PATH="$SHOWDOWN" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ROOT/venv/bin/python" "$ROOT/scripts/evaluate_counterplay_paired.py" "${args[@]}"
}

# The pair seeds are new and disjoint from the Round600 rollout seed range.
# Every matchup uses the same generated teams and alternated team roles.
run_cell vs_max_damage max_damage
run_cell vs_public_counterplay public_counterplay
run_cell vs_random random
run_cell vs_round102 policy "$ROUND102"
run_cell vs_round585 policy "$ROUND585"
echo "Round600h fresh-seed screen complete: $OUT/*/paired_summary.json"
