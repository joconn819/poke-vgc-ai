#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
BASE="${BASE:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
CANDIDATE="${CANDIDATE:-$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
R102="${R102:-$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt}"
R485="${R485:-$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round502_round476_fresh_league_128x2_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
REPEATS="${REPEATS:-2}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611502}"

for file in "$BASE" "$CANDIDATE" "$SELECTOR" "$R102" "$R485"; do
  [[ -s "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ "$PAIRS" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAIRS, REPEATS, and WORKERS must be positive integers" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_arm() {
  local matchup="$1" opponent="$2" opponent_checkpoint="$3" model="$4" seed="$5"
  local output="$OUTPUT_ROOT/$matchup/$model"
  local checkpoint="$BASE"
  [[ "$model" != round476 ]] || checkpoint="$CANDIDATE"
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$checkpoint" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats "$REPEATS" --workers "$WORKERS" --seed "$seed"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu
    --output-dir "$output")
  if [[ -n "$opponent_checkpoint" ]]; then
    command+=(--opponent-checkpoint "$opponent_checkpoint")
  fi
  "${command[@]}"
}

run_matchup() {
  local matchup="$1" opponent="$2" opponent_checkpoint="$3" seed="$4"
  echo "Evaluating $matchup on paired team seeds ($seed)" >&2
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round430 "$seed"
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round476 "$seed"
  venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$OUTPUT_ROOT/$matchup/round430/paired_summary.json" \
    --candidate-summary "$OUTPUT_ROOT/$matchup/round476/paired_summary.json" \
    --output "$OUTPUT_ROOT/$matchup/paired_comparison.json" \
    --seed "$seed"
}

# Fresh team seeds are used across the suite. The six opponent classes include
# a learned PPO that Round476 did not train against (Round485).
run_matchup random random "" "$SEED"
run_matchup max_damage max_damage "" "$((SEED + 1))"
run_matchup counterplay counterplay "" "$((SEED + 2))"
run_matchup public_counterplay public_counterplay "" "$((SEED + 3))"
run_matchup historical_ppo policy "$R102" "$((SEED + 4))"
run_matchup heldout_ppo_round485 policy "$R485" "$((SEED + 5))"

venv/bin/python - "$OUTPUT_ROOT" "$BASE" "$CANDIDATE" "$PAIRS" "$REPEATS" "$WORKERS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for path in sorted(root.glob("*/paired_comparison.json")):
    comparison = json.loads(path.read_text())
    starting = json.loads((path.parent / "round430" / "paired_summary.json").read_text())
    candidate = json.loads((path.parent / "round476" / "paired_summary.json").read_text())
    rows[path.parent.name] = {
        "round430_score_pct": round(starting["side_swapped_score_rate"] * 100, 3),
        "round476_score_pct": round(candidate["side_swapped_score_rate"] * 100, 3),
        "round430_games_per_second": round(starting["aggregate_games_per_second"], 3),
        "round476_games_per_second": round(candidate["aggregate_games_per_second"], 3),
        "paired_delta_pp": round(comparison["candidate_minus_starting_rate"] * 100, 3),
        "cluster_ci95_pp": [round(x * 100, 3) for x in comparison["difference_95_percentile_interval"]],
        "matched_team_clusters": comparison["matched_team_pair_clusters"],
        "matched_repeats": comparison["matched_cases"],
        "round430_timeouts_or_errors": starting["timeouts_or_errors"],
        "round476_timeouts_or_errors": candidate["timeouts_or_errors"],
        "round430_action_adjustments": starting["action_adjustments"],
        "round476_action_adjustments": candidate["action_adjustments"],
        "interpretation": "Exploratory paired comparison; checkpoint changes are not promoted by this screen alone.",
    }
if len(rows) != 6:
    raise SystemExit(f"expected six completed opponent classes, found {len(rows)}")
payload = {
    "status": "Complete; exploratory six-class paired league screen",
    "candidate": "Round476 refreshed-population PPO",
    "baseline": "Round430 hard-opponent auxiliary PPO",
    "method": "For each opponent, both checkpoints use the same fresh team seeds, preview selectors, side swaps, and two outcomes per team-pair cluster. Cluster-bootstrap paired intervals compare Round476 with Round430. Showdown battle randomness is generated server-side, so outcome pairing is by team cluster rather than identical random streams.",
    "opponent_classes": ["Random", "Max Damage", "Counterplay", "Public Counterplay", "historical Round102 PPO", "held-out Round485 PPO"],
    "paired_cases_per_checkpoint_per_opponent": int(sys.argv[4]) * int(sys.argv[5]),
    "side_swapped_matches_per_checkpoint_per_opponent_requested": int(sys.argv[4]) * int(sys.argv[5]) * 2,
    "workers": int(sys.argv[6]),
    "round430_checkpoint": sys.argv[2],
    "round476_checkpoint": sys.argv[3],
    "opponents": rows,
    "promotion": "No promotion decision: this is a fresh screening league; repeat any apparent gain on independent teams and seeds before changing the active checkpoint.",
}
(root / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({name: {"round430": row["round430_score_pct"], "round476": row["round476_score_pct"], "delta_pp": row["paired_delta_pp"], "ci95": row["cluster_ci95_pp"]} for name, row in rows.items()}, indent=2))
PY
