#!/usr/bin/env python3
"""Recheck roster-core separation and provenance for generated held-out team splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.audit_roster_disjointness import read_evaluation
from scripts.create_roster_disjoint_team_splits import CoreIndex, roster_hash, roster_signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-roster-manifest", type=Path, required=True)
    parser.add_argument("--exclude-evaluation-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    training_bytes = args.training_roster_manifest.read_bytes()
    expected_sha = hashlib.sha256(training_bytes).hexdigest()
    if expected_sha != manifest.get("training_roster_manifest_sha256"):
        raise SystemExit("training roster manifest hash mismatch")
    training = json.loads(training_bytes)
    index = CoreIndex([tuple(row["species"]) for row in training.get("rosters", [])])
    for eval_dir in args.exclude_evaluation_dir:
        clusters, _ = read_evaluation(eval_dir)
        index.extend([roster for pair in clusters.values() for roster in pair])

    threshold = int(manifest["shared_core_threshold"])
    split_results = {}
    all_hashes = set()
    for split_name, pairs in manifest.get("splits", {}).items():
        checked_rosters = 0
        for expected_pair_id, pair in enumerate(pairs):
            if pair.get("pair_id") != expected_pair_id:
                raise SystemExit(f"non-contiguous pair ids in {split_name}")
            roster_a = roster_signature(pair["team_a"])
            roster_b = roster_signature(pair["team_b"])
            observed_a = index.max_shared(roster_a)
            observed_b = index.max_shared(roster_b)
            mutual = len(set(roster_a) & set(roster_b))
            if observed_a >= threshold or observed_b >= threshold or mutual >= threshold:
                raise SystemExit(
                    f"core overlap in {split_name} pair {expected_pair_id}: "
                    f"train/reserved={observed_a}/{observed_b}, mutual={mutual}"
                )
            if pair.get("roster_hash_a") != roster_hash(roster_a):
                raise SystemExit(f"roster A hash mismatch in {split_name} pair {expected_pair_id}")
            if pair.get("roster_hash_b") != roster_hash(roster_b):
                raise SystemExit(f"roster B hash mismatch in {split_name} pair {expected_pair_id}")
            if pair["roster_hash_a"] in all_hashes or pair["roster_hash_b"] in all_hashes:
                raise SystemExit("duplicate roster crossed or repeated within the reserved split")
            if pair["max_training_or_reserved_shared_species_a"] != observed_a:
                raise SystemExit(f"stored max overlap A mismatch in {split_name} pair {expected_pair_id}")
            if pair["max_training_or_reserved_shared_species_b"] != observed_b:
                raise SystemExit(f"stored max overlap B mismatch in {split_name} pair {expected_pair_id}")
            all_hashes.update((pair["roster_hash_a"], pair["roster_hash_b"]))
            index.extend([roster_a, roster_b])
            checked_rosters += 2
        split_results[split_name] = {
            "pair_clusters": len(pairs),
            "rosters_checked": checked_rosters,
        }

    revision = subprocess.run(
        ["git", "-C", manifest["showdown_path"], "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if revision != manifest.get("showdown_revision"):
        raise SystemExit("Showdown revision differs from manifest provenance")
    print(json.dumps({
        "status": "verified",
        "showdown_revision": revision,
        "family_threshold": threshold,
        "training_species_rosters": training.get("unique_species_rosters"),
        "split_results": split_results,
    }, indent=2))


if __name__ == "__main__":
    main()
