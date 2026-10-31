#!/usr/bin/env python3
"""Merge Round488 phase-specific counterfactual collections with unique clusters."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: merge_round488_action_value_data.py OUTPUT INPUT_STEP1 INPUT_STEP3 [INPUT_STEP5 ...]")
    output = Path(sys.argv[1])
    inputs = [Path(value) for value in sys.argv[2:]]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    expected = {"checkpoint", "candidate_policy_checkpoint", "opponent_checkpoint",
                "team_selection_checkpoint", "format", "paired_response_classes",
                "hidden_worlds_requested", "hidden_world_prior"}
    for key in expected:
        if len({json.dumps(data.get(key), sort_keys=True) for data in payloads}) != 1:
            raise ValueError(f"collection metadata differs across phases for {key}")

    rows = []
    source_collections: list[dict[str, Any]] = []
    cluster_offset = 0
    total_matches = 0
    total_elapsed = 0.0
    total_missing = 0
    total_rejected = 0
    phase_counts: dict[str, int] = {}
    for path, data in zip(inputs, payloads):
        clusters = int(data["clusters_requested"])
        for row in data.get("rows", []):
            row = dict(row)
            row["team_cluster"] = int(row["team_cluster"]) + cluster_offset
            row["cluster"] = int(row["cluster"]) + cluster_offset
            rows.append(row)
        step = int(data["decision_step"])
        phase_counts[str(step)] = int(data["states_captured"])
        source_collections.append({
            "path": str(path),
            "decision_step": step,
            "clusters_requested": clusters,
            "states_captured": int(data["states_captured"]),
            "matches_executed": int(data["matches_executed"]),
            "elapsed_seconds": float(data["elapsed_seconds"]),
            "matches_per_second": float(data["matches_per_second"]),
            "missing_states": int(data.get("missing_states", 0)),
            "rejected_candidates": int(data.get("rejected_candidates", 0)),
        })
        total_matches += int(data["matches_executed"])
        total_elapsed += float(data["elapsed_seconds"])
        total_missing += int(data.get("missing_states", 0))
        total_rejected += int(data.get("rejected_candidates", 0))
        cluster_offset += clusters

    merged = {
        "schema_version": 1,
        "output": str(output),
        "experiment": "Multi-phase public-view action-value dataset",
        "checkpoint": payloads[0]["checkpoint"],
        "candidate_policy_checkpoint": payloads[0]["candidate_policy_checkpoint"],
        "opponent_checkpoint": payloads[0]["opponent_checkpoint"],
        "team_selection_checkpoint": payloads[0]["team_selection_checkpoint"],
        "format": payloads[0]["format"],
        "decision_steps": sorted(map(int, phase_counts)),
        "phase_state_counts": phase_counts,
        "clusters_requested": cluster_offset,
        "states_captured": sum(phase_counts.values()),
        "chance_repeats": payloads[0]["chance_repeats"],
        "hidden_worlds_requested": payloads[0].get("hidden_worlds_requested", 0),
        "hidden_worlds_prepared": sum(
            int(data.get("hidden_worlds_prepared", 0)) for data in payloads
        ),
        "hidden_world_preparation_seconds": sum(
            float(data.get("hidden_world_preparation_seconds", 0.0)) for data in payloads
        ),
        "hidden_world_prior": payloads[0].get(
            "hidden_world_prior", "actual_captured_battle_world"
        ),
        "candidate_mode": payloads[0]["candidate_mode"],
        "paired_response_classes": payloads[0]["paired_response_classes"],
        "matches_executed": total_matches,
        "elapsed_seconds_sum": round(total_elapsed, 3),
        "matches_per_second_aggregate": round(total_matches / max(total_elapsed, 1e-9), 3),
        "missing_states": total_missing,
        "rejected_candidates": total_rejected,
        "source_collections": source_collections,
        "rows": rows,
        "team_clusters": sorted({int(row["team_cluster"]) for row in rows}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: merged[key] for key in (
        "output", "decision_steps", "phase_state_counts", "clusters_requested",
        "states_captured", "matches_executed", "elapsed_seconds_sum",
        "matches_per_second_aggregate", "missing_states", "rejected_candidates",
    )} | {"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
