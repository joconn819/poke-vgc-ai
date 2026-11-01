#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
BASE="${BASE:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
CANDIDATE="${CANDIDATE:-$ROOT/round530_rectangular_robust_response_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
R473="$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
R476="$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
R524="$ROOT/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
R485="$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round531_round530_fresh_league_128x1_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611531}"

for file in "$BASE" "$CANDIDATE" "$SELECTOR" "$R473" "$R476" "$R485" "$R524"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ "$PAIRS" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAIRS and WORKERS must be positive integers" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_arm() {
  local matchup="$1" opponent="$2" opponent_checkpoint="$3" model="$4" seed="$5"
  local output="$OUTPUT_ROOT/$matchup/$model"
  local checkpoint="$BASE"
  [[ "$model" != round530 ]] || checkpoint="$CANDIDATE"
  local command=(venv/bin/python scripts/evaluate_native_paired_parallel.py
    --checkpoint "$checkpoint" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats 1 --workers "$WORKERS" --seed "$seed"
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
  echo "Evaluating $matchup on fresh paired team seeds ($seed)" >&2
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round430 "$seed"
  run_arm "$matchup" "$opponent" "$opponent_checkpoint" round530 "$seed"
  venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
    --starting-summary "$OUTPUT_ROOT/$matchup/round430/paired_summary.json" \
    --candidate-summary "$OUTPUT_ROOT/$matchup/round530/paired_summary.json" \
    --output "$OUTPUT_ROOT/$matchup/paired_comparison.json" \
    --seed "$seed"
}

# Keep direct heuristics first, then evaluate every prior policy checkpoint in the training pool.
run_matchup random random "" "$SEED"
run_matchup max_damage max_damage "" "$((SEED + 1))"
run_matchup counterplay counterplay "" "$((SEED + 2))"
run_matchup public_counterplay public_counterplay "" "$((SEED + 3))"
run_matchup round473 policy "$R473" "$((SEED + 4))"
run_matchup round476 policy "$R476" "$((SEED + 5))"
run_matchup round485 policy "$R485" "$((SEED + 6))"
run_matchup round524 policy "$R524" "$((SEED + 7))"

venv/bin/python - "$OUTPUT_ROOT" "$BASE" "$CANDIDATE" "$PAIRS" "$WORKERS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for path in sorted(root.glob("*/paired_comparison.json")):
    comparison = json.loads(path.read_text())
    starting = json.loads((path.parent / "round430" / "paired_summary.json").read_text())
    candidate = json.loads((path.parent / "round530" / "paired_summary.json").read_text())
    rows[path.parent.name] = {
        "round430_score_pct": round(starting["side_swapped_score_rate"] * 100, 3),
        "round530_score_pct": round(candidate["side_swapped_score_rate"] * 100, 3),
        "round430_games_per_second": round(starting["aggregate_games_per_second"], 3),
        "round530_games_per_second": round(candidate["aggregate_games_per_second"], 3),
        "paired_delta_pp": round(comparison["candidate_minus_starting_rate"] * 100, 3),
        "cluster_ci95_pp": [round(x * 100, 3) for x in comparison["difference_95_percentile_interval"]],
        "matched_team_clusters": comparison["matched_team_pair_clusters"],
        "matched_cases": comparison["matched_cases"],
        "round430_timeouts_or_errors": starting["timeouts_or_errors"],
        "round530_timeouts_or_errors": candidate["timeouts_or_errors"],
        "round430_action_adjustments": starting["action_adjustments"],
        "round530_action_adjustments": candidate["action_adjustments"],
    }
if len(rows) != 8:
    raise SystemExit(f"expected eight completed opponent classes, found {len(rows)}")
payload = {
    "status": "Complete; fresh eight-class screen",
    "candidate": "Round530 bootstrap-weighted robust-response PPO",
    "baseline": "Round430 hard-opponent auxiliary PPO",
    "method": "128 fresh team-pair clusters per opponent, one outcome repeat per cluster, side-swapped, identical seeds/teams/previews/selectors for both checkpoints. Paired confidence intervals bootstrap team-pair clusters. Server randomness is not identical between candidate and baseline arms.",
    "opponent_classes": ["Random", "Max Damage", "Counterplay", "Public Counterplay", "Round473 PPO", "Round476 PPO", "Round485 PPO", "Round524 PPO"],
    "paired_cases_per_checkpoint_per_opponent": int(sys.argv[4]),
    "side_swapped_matches_per_checkpoint_per_opponent_requested": int(sys.argv[4]) * 2,
    "workers": int(sys.argv[5]),
    "round430_checkpoint": sys.argv[2],
    "round530_checkpoint": sys.argv[3],
    "opponents": rows,
    "promotion": "Screening evidence only; do not promote without independent confirmation and held-out-team evaluation.",
    "reproduction": "scripts/evaluate_round530_fresh_league_128x1.sh",
}
(root / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({name: {"round430": row["round430_score_pct"], "round530": row["round530_score_pct"], "delta_pp": row["paired_delta_pp"], "ci95": row["cluster_ci95_pp"]} for name, row in rows.items()}, indent=2))
PY
