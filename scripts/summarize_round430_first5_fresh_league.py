#!/usr/bin/env python3
"""Summarize the paired Round430 first-five intervention league."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_summary(path: Path) -> tuple[dict, dict[tuple[int, int], dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        (int(row["team_seed"]), int(row.get("repeat", 0))): row
        for row in data.get("results", ()) if row.get("complete")
    }
    if len(rows) != data.get("pairs_completed"):
        raise ValueError(f"duplicate or incomplete rows in {path}")
    return data, rows


def paired_clusters(baseline: dict, candidate: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    shared = sorted(baseline.keys() & candidate.keys())
    if not shared:
        raise ValueError("baseline and candidate have no complete matched outcomes")
    cluster_scores: dict[int, list[tuple[float, float]]] = {}
    for key in shared:
        left, right = baseline[key], candidate[key]
        for field in ("team_a", "team_b", "preview_a", "preview_b"):
            if left.get(field) != right.get(field):
                raise ValueError(f"paired mismatch for {key}: {field}")
        cluster_scores.setdefault(key[0], []).append((
            float(left["checkpoint_points"]) / 2.0,
            float(right["checkpoint_points"]) / 2.0,
        ))
    ordered = [cluster_scores[k] for k in sorted(cluster_scores)]
    base = np.asarray([np.mean([pair[0] for pair in values]) for values in ordered])
    cand = np.asarray([np.mean([pair[1] for pair in values]) for values in ordered])
    return cand - base, base, cand, len(shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20680577)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        parser.error("use at least 1,000 bootstrap replicates")

    groups = {}
    for label in ("max_damage", "counterplay", "public_counterplay", "policy", "policy_round476"):
        directory = args.input_dir / label
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        base_payload, base = load_summary(directory / "baseline" / "paired_summary.json")
        cand_payload, cand = load_summary(directory / "first5" / "paired_summary.json")
        shared = sorted(base.keys() & cand.keys())
        delta, base_cluster_scores, candidate_cluster_scores, shared_cases = paired_clusters(base, cand)
        if len(delta) != len({key[0] for key in shared}):
            raise ValueError(f"cluster accounting mismatch in {label}")
        candidate_mean = float(candidate_cluster_scores.mean())
        baseline_mean = float(base_cluster_scores.mean())
        groups[label] = {
            "baseline_score_percent": 100 * baseline_mean,
            "first5_score_percent": 100 * candidate_mean,
            "paired_delta_pp": 100 * float(delta.mean()),
            "clusters": int(delta.size),
            "games_per_arm": shared_cases * 2,
            "baseline_games_per_second": base_payload.get("aggregate_games_per_second"),
            "first5_games_per_second": cand_payload.get("aggregate_games_per_second"),
            "action_corrections_baseline": base_payload.get("action_adjustments"),
            "action_corrections_first5": cand_payload.get("action_adjustments"),
            "paired_cluster_deltas": delta,
        }

    rng = np.random.default_rng(args.seed)
    boot_by_group: dict[str, np.ndarray] = {}
    for label, row in groups.items():
        values = row.pop("paired_cluster_deltas")
        draws = rng.integers(0, len(values), size=(args.bootstrap_replicates, len(values)))
        boot_by_group[label] = values[draws].mean(axis=1)
        row["paired_ci95_pp"] = [
            100 * float(v) for v in np.quantile(boot_by_group[label], [0.025, 0.975])
        ]
    equal_boot = np.mean(np.stack(list(boot_by_group.values()), axis=1), axis=1)
    point_deltas = {label: row["paired_delta_pp"] for label, row in groups.items()}
    result = {
        "status": "Fresh paired league complete · intervention not robust across all opponents",
        "candidate": "Round430 with the first five legal Max Damage actions",
        "baseline": "Round430 masked PPO policy",
        "method": "Five opponent classes, distinct fresh seeds, 256 team-pair clusters and two side-swapped outcomes per cluster. Both arms used Round402 preview selection. Paired bootstrap resampled complete shared team clusters within each opponent; the across-opponent estimate equally weights the five classes.",
        "opponents": groups,
        "equal_weight_delta_pp": float(np.mean(list(point_deltas.values()))),
        "equal_weight_ci95_pp": [100 * float(v) for v in np.quantile(equal_boot, [0.025, 0.975])],
        "worst_opponent": min(point_deltas, key=point_deltas.get),
        "worst_opponent_delta_pp": min(point_deltas.values()),
        "interpretation": "Four opponents favor the intervention, but legacy Counterplay has a significant regression. The opponent-specific lower-tail loss disqualifies this fixed override as a robust default; use these data to motivate policy learning over diverse responses, not an opponent-specific switch.",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
