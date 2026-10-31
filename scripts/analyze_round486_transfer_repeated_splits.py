#!/usr/bin/env python3
"""Aggregate out-of-fold action-value selector results across random splits."""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RESPONSES = ("max_damage", "policy", "public_counterplay")


def _ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202610486)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.report_glob))
    if len(paths) < 2:
        raise SystemExit(f"Expected repeated split reports; found {len(paths)}")
    per_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
    run_metrics = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        run_metrics.append({
            "report": str(path),
            "seed": int(report.get("seed", metrics.get("seed", 0))) if report.get("seed") else None,
            "test_team_clusters": metrics["team_split"]["test"],
            "validation_brier": metrics["validation_action_outcome_brier"],
            "selector_worst_response_delta": metrics["robust_selection"]["selected_minus_policy_worst_response"],
            "selector_worst_response_ci95": metrics["robust_selection"]["selected_minus_policy_worst_response_95_ci"],
            "selected_action_changed_states": metrics["robust_selection"]["action_changed_from_policy_states"],
        })
        for row in report["selected_actions"]:
            per_team[int(row["team_cluster"])].append({
                "changed": row["selected_action"] != row["policy_action"],
                "responses": {
                    name: float(row["observed_by_response"][name]["selected_minus_policy"])
                    for name in RESPONSES
                },
                "worst": float(row["observed_worst_response"]["selected_minus_policy"]),
            })

    team_means = []
    all_records = [row for rows in per_team.values() for row in rows]
    for rows in per_team.values():
        team_means.append({
            "changed": statistics.mean(float(row["changed"]) for row in rows),
            **{
                name: statistics.mean(row["responses"][name] for row in rows)
                for name in RESPONSES
            },
            "worst": statistics.mean(row["worst"] for row in rows),
        })

    summary = {}
    for metric in (*RESPONSES, "worst"):
        values = [row[metric] for row in team_means]
        summary[metric] = {
            "mean_delta_pp": round(statistics.mean(values) * 100, 3),
            "team_cluster_bootstrap_ci95_pp": [
                round(value * 100, 3)
                for value in _ci(values, seed=args.seed + len(summary), samples=args.bootstrap_samples)
            ],
        }
    report = {
        "schema_version": 1,
        "experiment": "Round486 Round484 action-value selector repeated cluster splits",
        "method": (
            "Each report trains on disjoint team clusters and the first four chance repeats, "
            "then selects on unseen team clusters and scores on their last four repeats. "
            "Results are first averaged within each held-out team cluster across the splits in "
            "which it appeared, then bootstrapped over unique team clusters."
        ),
        "runs": run_metrics,
        "unique_heldout_team_clusters": len(team_means),
        "heldout_selection_records": len(all_records),
        "selected_nonbaseline_action_records": sum(row["changed"] for row in all_records),
        "out_of_fold_paired_deltas": summary,
        "interpretation_limit": (
            "A diagnostic on 128 opening states and the existing candidate set, not a full-match "
            "agent evaluation. Repeated random splits overlap, so records are clustered by team. "
            "The selector's opponent robustness is limited to three sampled response classes."
        ),
        "bootstrap_samples": args.bootstrap_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(paths),
                      "unique_heldout_team_clusters": len(team_means),
                      "out_of_fold_paired_deltas": summary}, indent=2))


if __name__ == "__main__":
    main()
