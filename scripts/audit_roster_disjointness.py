#!/usr/bin/env python3
"""Audit exact team roster and shared-core overlap between training and evaluation.

Training rollout corpora store the six species used in each full game. Paired
benchmark records store the two rosters and are deduplicated across outcome
repeats. A species overlap threshold is a transparent diagnostic, not a
universal definition of a competitive team family.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import orjson
from pathlib import Path
import re
from typing import Any


def species_roster(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    species: list[str] = []
    for member in value:
        name = (
            member.get("species") or member.get("name")
            if isinstance(member, dict) else member
        )
        if not name:
            return None
        species.append(str(name).strip().lower())
    if len(species) != 6 or len(set(species)) != 6:
        return None
    return tuple(sorted(species))


def digest(roster: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(roster).encode()).hexdigest()[:16]


def read_training_rosters(root: Path) -> tuple[set[tuple[str, ...]], dict[str, int]]:
    rosters: set[tuple[str, ...]] = set()
    counters = Counter()
    compressed_rollouts = sorted(root.rglob("*.jsonl.gz"))
    if compressed_rollouts:
        # Training collectors keep rejected and recovery attempts beside the
        # authoritative compressed shards; only the latter form the PPO corpus.
        paths = compressed_rollouts
    else:
        # Uncompressed corpora are used by older PPO runs. Omit telemetry,
        # rejected and recovery files, which are not optimizer input data.
        paths = sorted(
            path for path in root.rglob("*.jsonl")
            if ".telemetry." not in path.name
            and not path.name.startswith(("rejected_", "recovery_"))
        )
    for path in paths:
        counters["rollout_files"] += 1
        open_binary = gzip.open if path.name.endswith(".gz") else Path.open
        with open_binary(path, "rb") as stream:
            for line in stream:
                counters["rollout_rows"] += 1
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    counters["malformed_rollout_rows"] += 1
                    continue
                teams = row.get("teams")
                if not isinstance(teams, dict):
                    continue
                for side in ("player1", "player2"):
                    roster = species_roster(teams.get(side))
                    if roster is not None:
                        counters["training_roster_instances"] += 1
                        rosters.add(roster)
    if not rosters:
        raise ValueError(f"no six-species training rosters found under {root}")
    return rosters, dict(counters)


def read_evaluation(path: Path) -> tuple[dict[str, tuple[tuple[str, ...], tuple[str, ...]]], dict[str, int]]:
    files = [path] if path.is_file() else sorted(path.rglob("paired_cases.jsonl"))
    clusters: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    counters = Counter(evaluation_files=len(files))
    if not files and path.is_dir():
        team_files = sorted(path.rglob("teams/pair_*_[AB].json"))
        if team_files:
            counters["evaluation_files"] = len(team_files)
            partial: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
            pattern = re.compile(r"pair_(\d+)_([AB])\.json$")
            for team_file in team_files:
                match = pattern.fullmatch(team_file.name)
                if match is None:
                    continue
                try:
                    roster = species_roster(json.loads(team_file.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    roster = None
                if roster is None:
                    counters["invalid_evaluation_rows"] += 1
                    continue
                key = f"{team_file.parent.parent.name}:{match.group(1)}"
                partial[key][match.group(2)] = roster
            for key, pair in partial.items():
                if set(pair) == {"A", "B"}:
                    clusters[key] = (pair["A"], pair["B"])
                else:
                    counters["incomplete_team_file_pairs"] += 1
            counters["evaluation_rows"] = len(team_files)
            counters["team_pair_clusters"] = len(clusters)
            counters["evaluation_roster_instances"] = len(clusters) * 2
            if clusters:
                return clusters, dict(counters)
        raise ValueError(f"no paired_cases.jsonl or paired team files found under {path}")
    for file_path in files:
        with file_path.open(encoding="utf-8") as stream:
            for line in stream:
                counters["evaluation_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counters["malformed_evaluation_rows"] += 1
                    continue
                if row.get("complete") is False:
                    continue
                team_a = species_roster(row.get("team_a"))
                team_b = species_roster(row.get("team_b"))
                if team_a is None or team_b is None:
                    counters["invalid_evaluation_rows"] += 1
                    continue
                key = str(row.get("team_seed", row.get("pair", row.get("pair_index", ""))))
                if not key:
                    key = f"{digest(team_a)}:{digest(team_b)}"
                rosters = (team_a, team_b)
                previous = clusters.get(key)
                if previous is not None and previous != rosters:
                    raise ValueError(f"team rosters changed across repeats for cluster {key}")
                clusters[key] = rosters
    if not clusters:
        raise ValueError(f"no complete team-pair clusters found under {path}")
    counters["team_pair_clusters"] = len(clusters)
    counters["evaluation_roster_instances"] = len(clusters) * 2
    return clusters, dict(counters)


def audit_eval(
    train_rosters: set[tuple[str, ...]],
    clusters: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    counters: dict[str, int],
    threshold: int,
) -> dict[str, Any]:
    train_ids = {digest(roster): roster for roster in train_rosters}
    train_by_species: dict[str, set[str]] = defaultdict(set)
    for roster_id, roster in train_ids.items():
        for species in roster:
            train_by_species[species].add(roster_id)

    eval_rosters: set[tuple[str, ...]] = set()
    overlap_hist = Counter()
    high_overlap_rosters: list[dict[str, Any]] = []
    pair_high_overlap = 0
    for pair_id, (team_a, team_b) in clusters.items():
        pair_exceeds = False
        for label, roster in (("team_a", team_a), ("team_b", team_b)):
            eval_rosters.add(roster)
            candidates: Counter[str] = Counter()
            for species in roster:
                candidates.update(train_by_species.get(species, ()))
            maximum = max(candidates.values(), default=0)
            overlap_hist[str(maximum)] += 1
            if maximum >= threshold:
                pair_exceeds = True
                best = [
                    {
                        "training_roster_hash": roster_id,
                        "shared_species": sorted(set(roster) & set(train_ids[roster_id])),
                    }
                    for roster_id, count in candidates.items()
                    if count == maximum
                ][:5]
                high_overlap_rosters.append({
                    "evaluation_cluster": pair_id,
                    "team": label,
                    "evaluation_roster_hash": digest(roster),
                    "max_shared_species": maximum,
                    "best_training_matches": best,
                })
        pair_high_overlap += int(pair_exceeds)

    exact_overlap = eval_rosters & train_rosters
    return {
        **counters,
        "unique_evaluation_rosters": len(eval_rosters),
        "unique_exact_roster_overlap": len(exact_overlap),
        "exact_roster_overlap_hashes": sorted(digest(roster) for roster in exact_overlap),
        "maximum_shared_species_distribution": dict(sorted(
            overlap_hist.items(), key=lambda item: int(item[0])
        )),
        "shared_core_threshold_species": threshold,
        "evaluation_rosters_at_or_above_shared_core_threshold": sum(
            int(count) for shared, count in overlap_hist.items()
            if int(shared) >= threshold
        ),
        "team_pair_clusters_with_any_roster_at_or_above_threshold": pair_high_overlap,
        "high_overlap_examples": high_overlap_rosters[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-rollout-dir", type=Path, action="append", required=True,
        help="repeat for every checkpoint-training corpus whose team rosters matter",
    )
    parser.add_argument("--evaluation-dir", type=Path, action="append", required=True)
    parser.add_argument("--shared-core-threshold", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roster-manifest-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.shared_core_threshold <= 6:
        parser.error("shared-core threshold must be between 1 and 6")
    train_rosters: set[tuple[str, ...]] = set()
    training_counts = {}
    for train_dir in args.training_rollout_dir:
        roster_set, counts = read_training_rosters(train_dir)
        train_rosters.update(roster_set)
        training_counts[str(train_dir)] = {
            **counts,
            "unique_species_rosters": len(roster_set),
        }
    if args.roster_manifest_output:
        args.roster_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.roster_manifest_output.write_text(json.dumps({
            "sources": [str(path) for path in args.training_rollout_dir],
            "unique_species_rosters": len(train_rosters),
            "rosters": [
                {"hash": digest(roster), "species": list(roster)}
                for roster in sorted(train_rosters)
            ],
        }, separators=(",", ":")) + "\n", encoding="utf-8")
    results = {}
    for eval_dir in args.evaluation_dir:
        clusters, eval_counts = read_evaluation(eval_dir)
        key = str(eval_dir)
        results[key] = audit_eval(
            train_rosters, clusters, eval_counts, args.shared_core_threshold
        )
    report = {
        "method": "Exact species roster match plus max shared-species count to any training roster. Evaluation repeats and duplicate arm files are deduplicated by team_seed.",
        "training_sources": [str(path) for path in args.training_rollout_dir],
        "training_counts_by_source": training_counts,
        "unique_training_species_rosters": len(train_rosters),
        "shared_core_threshold_note": "A 4/6 overlap is a sensitivity diagnostic, not a definitive competitive-team family definition.",
        "evaluations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_counts_by_source": training_counts,
        "unique_training_species_rosters": len(train_rosters),
        "evaluations": {
            key: {
                field: value
                for field, value in result.items()
                if field not in {"high_overlap_examples", "exact_roster_overlap_hashes"}
            }
            for key, result in results.items()
        },
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
