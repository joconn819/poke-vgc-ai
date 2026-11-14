#!/usr/bin/env python3
"""Create deterministic legal development/final team manifests disjoint by roster core.

Each accepted six-Pokemon roster shares fewer than N species with every training,
previously evaluated, and already allocated roster. The threshold is a documented
operational family rule, not a claim that species overlap captures every strategy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from scripts.audit_roster_disjointness import read_evaluation
from src.env.headless_showdown import HeadlessShowdownWorker


def roster_signature(team: list[dict[str, Any]]) -> tuple[str, ...]:
    species = tuple(sorted(
        str(member.get("species") or member.get("name", "")).strip().lower()
        for member in team
    ))
    if len(species) != 6 or len(set(species)) != 6 or not all(species):
        raise ValueError("Showdown generator returned an invalid six-species team")
    return species


def roster_hash(roster: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(roster).encode()).hexdigest()[:16]


class CoreIndex:
    def __init__(self, rosters: list[tuple[str, ...]]):
        self.by_species: dict[str, set[str]] = defaultdict(set)
        self.rosters: dict[str, tuple[str, ...]] = {}
        for roster in rosters:
            self.add(roster)

    def add(self, roster: tuple[str, ...]) -> None:
        key = roster_hash(roster)
        if key in self.rosters:
            return
        self.rosters[key] = roster
        for species in roster:
            self.by_species[species].add(key)

    def max_shared(self, roster: tuple[str, ...]) -> int:
        counts: Counter[str] = Counter()
        for species in roster:
            counts.update(self.by_species.get(species, ()))
        return max(counts.values(), default=0)

    def extend(self, rosters: list[tuple[str, ...]]) -> None:
        for roster in rosters:
            self.add(roster)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-roster-manifest", type=Path, required=True)
    parser.add_argument("--exclude-evaluation-dir", type=Path, action="append", default=[])
    parser.add_argument("--dev-pairs", type=int, default=256)
    parser.add_argument("--final-pairs", type=int, default=1024)
    parser.add_argument("--seed-start", type=int, default=310593001)
    parser.add_argument("--shared-core-threshold", type=int, default=4)
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--showdown-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dev_pairs < 0 or args.final_pairs <= 0:
        parser.error("dev-pairs must be nonnegative and final-pairs must be positive")
    if not 1 <= args.shared_core_threshold <= 6:
        parser.error("shared-core threshold must be between 1 and 6")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    training_manifest = json.loads(args.training_roster_manifest.read_text(encoding="utf-8"))
    training_rosters = [
        tuple(row["species"]) for row in training_manifest.get("rosters", [])
    ]
    if not training_rosters:
        parser.error("training roster manifest contains no rosters")
    family_index = CoreIndex(training_rosters)
    excluded_eval_counts = {}
    for eval_dir in args.exclude_evaluation_dir:
        clusters, counts = read_evaluation(eval_dir)
        eval_rosters = [roster for pair in clusters.values() for roster in pair]
        family_index.extend(eval_rosters)
        excluded_eval_counts[str(eval_dir)] = {
            "team_pair_clusters": len(clusters),
            "unique_evaluation_rosters": len(set(eval_rosters)),
        }

    showdown_path = args.showdown_path.resolve()
    revision = subprocess.run(
        ["git", "-C", str(showdown_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or None
    splits: dict[str, list[dict[str, Any]]] = {"development": [], "final": []}
    rejections: Counter[str] = Counter()
    next_seed = args.seed_start
    started = time.perf_counter()
    worker = HeadlessShowdownWorker(showdown_path)
    try:
        for split_name, desired in (("development", args.dev_pairs), ("final", args.final_pairs)):
            while len(splits[split_name]) < desired:
                seed_a = next_seed
                seed_b = next_seed + 1
                next_seed += 2
                team_a = worker.generate_team(seed_a, args.battle_format)
                team_b = worker.generate_team(seed_b, args.battle_format)
                roster_a = roster_signature(team_a)
                roster_b = roster_signature(team_b)
                overlap_a = family_index.max_shared(roster_a)
                overlap_b = family_index.max_shared(roster_b)
                mutual_overlap = len(set(roster_a) & set(roster_b))
                if overlap_a >= args.shared_core_threshold or overlap_b >= args.shared_core_threshold:
                    rejections["training_or_reserved_core_overlap"] += 1
                    continue
                if mutual_overlap >= args.shared_core_threshold:
                    rejections["opposing_roster_core_overlap"] += 1
                    continue
                pair_id = len(splits[split_name])
                splits[split_name].append({
                    "pair_id": pair_id,
                    "team_seed_a": seed_a,
                    "team_seed_b": seed_b,
                    "team_a": team_a,
                    "team_b": team_b,
                    "roster_hash_a": roster_hash(roster_a),
                    "roster_hash_b": roster_hash(roster_b),
                    "max_training_or_reserved_shared_species_a": overlap_a,
                    "max_training_or_reserved_shared_species_b": overlap_b,
                    "mutual_species_overlap": mutual_overlap,
                })
                # Reserve each allocated family before generating the next pair.
                family_index.extend([roster_a, roster_b])
    finally:
        worker.close()

    source_hash = hashlib.sha256(args.training_roster_manifest.read_bytes()).hexdigest()
    output = {
        "schema": "roster-disjoint-team-splits-v1",
        "format": args.battle_format,
        "showdown_path": str(showdown_path),
        "showdown_revision": revision,
        "seed_start": args.seed_start,
        "shared_core_threshold": args.shared_core_threshold,
        "family_rule": f"Each accepted six-species roster shares fewer than {args.shared_core_threshold} species with all training, prior evaluation, and previously allocated rosters.",
        "training_roster_manifest": str(args.training_roster_manifest),
        "training_roster_manifest_sha256": source_hash,
        "excluded_prior_evaluations": excluded_eval_counts,
        "rejected_candidate_pairs": dict(sorted(rejections.items())),
        "generation_seconds": round(time.perf_counter() - started, 3),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "development_pairs": len(splits["development"]),
        "final_pairs": len(splits["final"]),
        "rejected_candidate_pairs": dict(sorted(rejections.items())),
        "generation_seconds": output["generation_seconds"],
        "showdown_revision": revision,
    }, indent=2))


if __name__ == "__main__":
    main()
