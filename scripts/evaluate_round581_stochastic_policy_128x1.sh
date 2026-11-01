#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$ROOT_DIR/training/outputs/phase12_headless"
CHECKPOINT="${CHECKPOINT:-$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/round581_round430_stochastic_actions_128x1_20261101}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-202611581}"

for file in "$CHECKPOINT" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_arm() {
  local opponent="$1" mode="$2" seed="$3"
  local out="$OUTPUT_ROOT/$opponent/$mode"
  local command=("$ROOT_DIR/venv/bin/python" "$ROOT_DIR/scripts/evaluate_native_paired_parallel.py"
    --checkpoint "$CHECKPOINT" --opponent "$opponent" --pairs "$PAIRS"
    --outcome-repeats 1 --workers "$WORKERS" --seed "$seed"
    --team-selection-checkpoint "$SELECTOR"
    --opponent-team-selection-checkpoint "$SELECTOR"
    --battle-format gen9championsdoubles2026mcclosed --device cpu --output-dir "$out")
  [[ "$mode" != stochastic ]] || command+=(--stochastic)
  "${command[@]}"
}

index=0
for opponent in random max_damage counterplay public_counterplay; do
  seed=$((SEED + index))
  echo "Evaluating Round430 $opponent, deterministic vs stochastic (seed $seed)" >&2
  run_arm "$opponent" deterministic "$seed"
  run_arm "$opponent" stochastic "$seed"
  "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/scripts/summarize_paired_checkpoint_comparison.py" \
    --starting-summary "$OUTPUT_ROOT/$opponent/deterministic/paired_summary.json" \
    --candidate-summary "$OUTPUT_ROOT/$opponent/stochastic/paired_summary.json" \
    --output "$OUTPUT_ROOT/$opponent/paired_comparison.json" --seed "$seed"
  index=$((index + 1))
done

"$ROOT_DIR/venv/bin/python" - "$OUTPUT_ROOT" "$CHECKPOINT" "$PAIRS" "$WORKERS" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
opponents = {}
for path in sorted(root.glob("*/paired_comparison.json")):
    comparison = json.loads(path.read_text())
    deterministic = json.loads((path.parent / "deterministic" / "paired_summary.json").read_text())
    stochastic = json.loads((path.parent / "stochastic" / "paired_summary.json").read_text())
    opponents[path.parent.name] = {
        "deterministic_score_pct": round(deterministic["side_swapped_score_rate"] * 100, 3),
        "stochastic_score_pct": round(stochastic["side_swapped_score_rate"] * 100, 3),
        "paired_delta_pp": round(comparison["candidate_minus_starting_rate"] * 100, 3),
        "cluster_ci95_pp": [round(x * 100, 3) for x in comparison["difference_95_percentile_interval"]],
        "clusters": comparison["matched_team_pair_clusters"],
        "deterministic_errors": deterministic["timeouts_or_errors"],
        "stochastic_errors": stochastic["timeouts_or_errors"],
        "stochastic_action_adjustments": stochastic["action_adjustments"],
    }
if len(opponents) != 4:
    raise SystemExit(f"expected four completed opponent classes, found {len(opponents)}")
payload = {
    "status": "Complete; stochastic-action diagnostic",
    "candidate": "Round430 with stochastic masked action sampling",
    "baseline": "Round430 with deterministic argmax actions",
    "method": "128 fresh team-pair clusters per opponent, both seat orders, identical seeds/team selectors; one outcome repeat per cluster. Only in-battle action sampling changes.",
    "opponents": opponents,
    "checkpoint": sys.argv[2],
    "clusters_per_opponent": int(sys.argv[3]),
    "workers": int(sys.argv[4]),
    "seed_start": int(sys.argv[5]),
    "interpretation": "This measures whether sampling from the existing policy distribution improves these held-out matchups. It does not train a new policy or prove Nash-equilibrium play.",
    "reproduction": "scripts/evaluate_round581_stochastic_policy_128x1.sh",
}
(root / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(opponents, indent=2))
PY
