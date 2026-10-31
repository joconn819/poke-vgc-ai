#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
BASE="${BASE:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
CANDIDATE="${CANDIDATE:-$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt}"
OPPONENT="${OPPONENT:-$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round503_round476_vs_round485_confirmation_256x2_20261031}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SEED="${SEED:-2026115502}"

for file in "$BASE" "$CANDIDATE" "$OPPONENT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT"

run_arm() {
  local name="$1" checkpoint="$2"
  venv/bin/python scripts/evaluate_native_paired_parallel.py \
    --checkpoint "$checkpoint" --opponent policy --opponent-checkpoint "$OPPONENT" \
    --pairs 256 --outcome-repeats 2 --workers 8 --seed "$SEED" \
    --team-selection-checkpoint "$SELECTOR" \
    --opponent-team-selection-checkpoint "$SELECTOR" \
    --battle-format gen9championsdoubles2026mcclosed --device cpu \
    --output-dir "$OUTPUT_ROOT/$name"
}

run_arm round430 "$BASE"
run_arm round476 "$CANDIDATE"
venv/bin/python scripts/summarize_paired_checkpoint_comparison.py \
  --starting-summary "$OUTPUT_ROOT/round430/paired_summary.json" \
  --candidate-summary "$OUTPUT_ROOT/round476/paired_summary.json" \
  --output "$OUTPUT_ROOT/paired_comparison.json" --seed "$SEED"

venv/bin/python - "$OUTPUT_ROOT" "$BASE" "$CANDIDATE" "$OPPONENT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "round430/paired_summary.json").read_text())
candidate = json.loads((root / "round476/paired_summary.json").read_text())
paired = json.loads((root / "paired_comparison.json").read_text())
report = {
    "status": "Independent fresh-seed confirmation of Round502 held-out PPO transfer signal",
    "baseline": "Round430 hard-opponent auxiliary PPO",
    "candidate": "Round476 refreshed-population PPO",
    "opponent": "Round485 Max Damage exposure PPO; not included in the Round476 training pool",
    "method": "256 fresh team-pair clusters, two outcomes per cluster, side-swapped; common teams, previews, seed, and Round402 selectors; cluster-bootstrap paired interval.",
    "round430_score_pct": round(base["side_swapped_score_rate"] * 100, 3),
    "round476_score_pct": round(candidate["side_swapped_score_rate"] * 100, 3),
    "paired_delta_pp": round(paired["candidate_minus_starting_rate"] * 100, 3),
    "paired_cluster_ci95_pp": [round(v * 100, 3) for v in paired["difference_95_percentile_interval"]],
    "matched_team_clusters": paired["matched_team_pair_clusters"],
    "matched_repeats": paired["matched_cases"],
    "games_per_checkpoint": base["games_completed"],
    "round430_games_per_second": round(base["aggregate_games_per_second"], 3),
    "round476_games_per_second": round(candidate["aggregate_games_per_second"], 3),
    "round430_timeouts_or_errors": base["timeouts_or_errors"],
    "round476_timeouts_or_errors": candidate["timeouts_or_errors"],
    "round430_action_adjustments": base["action_adjustments"],
    "round476_action_adjustments": candidate["action_adjustments"],
    "promotion": "Confirmation applies to one opponent checkpoint only; continue multi-family held-out league evaluation before promotion.",
    "raw_comparison": str(root / "paired_comparison.json"),
}
(root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2))
PY
