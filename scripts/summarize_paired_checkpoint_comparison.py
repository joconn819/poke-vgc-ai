"""Compare checkpoints on shared completed paired-evaluation team seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[tuple[int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        (int(row["team_seed"]), int(row.get("repeat", 0))): row
        for row in payload["results"] if row.get("complete")
    }
    if len(rows) != payload.get("pairs_completed"):
        raise ValueError(f"incomplete or duplicate team seeds in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument(
        "--allow-preview-difference", action="store_true",
        help="Compare selector changes on identical team pairs even when their chosen previews differ.",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("use at least 100 bootstrap replicates")

    starting = _load(args.starting_summary)
    candidate = _load(args.candidate_summary)
    shared = sorted(starting.keys() & candidate.keys())
    if not shared:
        parser.error("summary files have no shared completed team/preview seeds")
    for case_key in shared:
        a, b = starting[case_key], candidate[case_key]
        compare_fields = ["team_a", "team_b"]
        if not args.allow_preview_difference:
            compare_fields.extend(("preview_a", "preview_b"))
        for key in compare_fields:
            if a[key] != b[key]:
                parser.error(f"team or preview mismatch for case {case_key}: {key}")

    cluster_ids = sorted({team_seed for team_seed, _ in shared})
    starting_cluster_scores = []
    candidate_cluster_scores = []
    for team_seed in cluster_ids:
        repeats = [key for key in shared if key[0] == team_seed]
        starting_cluster_scores.append(np.mean([
            starting[key]["checkpoint_points"] / 2 for key in repeats
        ]))
        candidate_cluster_scores.append(np.mean([
            candidate[key]["checkpoint_points"] / 2 for key in repeats
        ]))
    start_points = np.asarray(starting_cluster_scores)
    candidate_points = np.asarray(candidate_cluster_scores)
    difference = candidate_points - start_points
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray([
        rng.choice(difference, size=len(difference), replace=True).mean()
        for _ in range(args.bootstrap_replicates)
    ])
    result = {
        "starting_summary": str(args.starting_summary),
        "candidate_summary": str(args.candidate_summary),
        "matched_cases": len(shared),
        "matched_team_pair_clusters": len(cluster_ids),
        "outcome_repeats_per_cluster": int(np.median([
            sum(1 for key in shared if key[0] == cluster)
            for cluster in cluster_ids
        ])),
        "starting_pairs_excluded": len(starting) - len(shared),
        "candidate_pairs_excluded": len(candidate) - len(shared),
        "starting_score_rate": float(start_points.mean()),
        "candidate_score_rate": float(candidate_points.mean()),
        "candidate_minus_starting_rate": float(difference.mean()),
        "difference_95_percentile_interval": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "pairs_candidate_better": int((difference > 0).sum()),
        "pairs_tied": int((difference == 0).sum()),
        "pairs_candidate_worse": int((difference < 0).sum()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "method": (
            "match fixed teams and repeat index between checkpoints; preview choices may differ"
            if args.allow_preview_difference else
            "match fixed teams, previews, and repeat index between checkpoints"
        ) + ". Showdown battle RNG is generated server-side and is not controlled by the evaluator seed, so outcomes are paired by team-pair cluster rather than identical battle randomness. Average repeated case scores within each team-pair cluster and bootstrap paired cluster-level differences",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
