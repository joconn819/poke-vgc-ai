#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/training/outputs/phase12_headless/round430_first5_fresh_league_256x2_20261031_retry1}"
BASE="${BASE:-$ROOT_DIR/training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT_DIR/training/outputs/phase12_headless/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
R102="${R102:-$ROOT_DIR/training/outputs/phase12_headless/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
R476="${R476:-$ROOT_DIR/training/outputs/phase12_headless/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
PAIRS="${PAIRS:-256}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-20680577}"

for path in "$BASE" "$SELECTOR" "$R102" "$R476"; do
  [[ -s "$path" ]] || { echo "Missing required checkpoint: $path" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
if [[ -e "$OUTPUT_ROOT" && "${RESUME:-0}" != "1" ]]; then
  echo "Output exists; set RESUME=1 to continue: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

run_arm() {
  local label="$1" opponent="$2" opponent_checkpoint="$3" arm="$4" seed="$5"
  local out="$OUTPUT_ROOT/$label/$arm"
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$BASE" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats 2 --workers "$WORKERS" --seed "$seed"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu
    --output-dir "$out")
  if [[ -n "$opponent_checkpoint" ]]; then
    command+=(--opponent-checkpoint "$opponent_checkpoint")
  fi
  if [[ "$arm" == "first5" ]]; then
    command+=(--intervention-target evaluated_checkpoint
      --max-damage-action-steps 0 1 2 3 4)
  fi
  if [[ -s "$out/paired_summary.json" ]]; then
    echo "Skipping completed arm: $opponent/$arm" >&2
    return
  fi
  if [[ -d "$out" ]] && find "$out" -mindepth 1 -print -quit | grep -q .; then
    echo "Incomplete arm output exists; inspect before rerunning: $out" >&2
    exit 1
  fi
  "${command[@]}"
}

run_matchup() {
  local label="$1" opponent="$2" opponent_checkpoint="$3" seed="$4"
  echo "Evaluating $label ($opponent) on seed $seed" >&2
  run_arm "$label" "$opponent" "$opponent_checkpoint" baseline "$seed"
  run_arm "$label" "$opponent" "$opponent_checkpoint" first5 "$seed"
  venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$OUTPUT_ROOT/$label/baseline/paired_summary.json" \
    --candidate-summary "$OUTPUT_ROOT/$label/first5/paired_summary.json" \
    --output "$OUTPUT_ROOT/$label/paired_comparison.json" \
    --seed "$seed"
}

run_matchup max_damage max_damage "" "$SEED"
run_matchup counterplay counterplay "" "$((SEED + 1))"
run_matchup public_counterplay public_counterplay "" "$((SEED + 2))"
run_matchup policy policy "$R102" "$((SEED + 3))"
run_matchup policy_round476 policy "$R476" "$((SEED + 4))"

venv/bin/python - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for path in sorted(root.glob("*/paired_comparison.json")):
    opponent = path.parent.name
    rows[opponent] = json.loads(path.read_text())
(root / "league_summary.json").write_text(json.dumps({
    "candidate": "Round430 with first five legal Max Damage decisions",
    "baseline": "Round430 direct masked PPO policy",
    "method": "Fresh team-pair clusters; two side-swapped outcomes; same Round402 selector and teams per arm; paired cluster bootstrap. Intervention applies only to the evaluated checkpoint.",
    "opponents": rows,
}, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: {"delta_pp": round(value["candidate_minus_starting_rate"] * 100, 2), "ci95_pp": [round(x * 100, 2) for x in value["difference_95_percentile_interval"]], "clusters": value["matched_team_pair_clusters"]} for key, value in rows.items()}, indent=2))
PY
