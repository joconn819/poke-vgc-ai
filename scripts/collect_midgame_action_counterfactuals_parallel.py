#!/usr/bin/env python3
"""Run independent, team-disjoint counterfactual collection shards in parallel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_midgame_action_counterfactuals.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument(
        "--player-strategy", choices=("policy", "public_counterplay"),
        default="policy",
    )
    parser.add_argument(
        "--preview-strategy", choices=("random", "damage"), default="random",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026102802)
    parser.add_argument("--states-per-cluster", type=int, default=2)
    parser.add_argument("--chance-repeats", type=int, default=4)
    parser.add_argument("--decision-step", type=int, default=3)
    parser.add_argument(
        "--candidate-mode", choices=("broad", "policy_vs_max_damage"),
        default="broad",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-public-counterplay-response", action="store_true",
        help="pair each action against Max Damage, Public Counterplay, and the frozen PPO",
    )
    parser.add_argument("--showdown-path", type=Path, required=True)
    args = parser.parse_args()
    if args.clusters < 1 or args.workers < 1 or args.workers > args.clusters:
        parser.error("workers and clusters must be positive, with workers <= clusters")
    if args.player_strategy == "policy" and args.checkpoint is None:
        parser.error("--checkpoint is required when --player-strategy=policy")
    for path in (args.checkpoint, args.opponent_checkpoint,
                 args.team_selection_checkpoint):
        if path is not None and not path.is_file():
            parser.error(f"checkpoint not found: {path}")
    if not (args.showdown_path / "dist" / "sim").is_dir():
        parser.error(f"Showdown simulator not found under {args.showdown_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "data.json"
    if final_path.exists():
        parser.error(f"refusing to overwrite {final_path}")
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir()

    quotient, remainder = divmod(args.clusters, args.workers)
    configs = []
    offset = 0
    for index in range(args.workers):
        count = quotient + int(index < remainder)
        output = shard_dir / f"shard_{index:02d}.json"
        log = shard_dir / f"shard_{index:02d}.log"
        command = [
            sys.executable, str(COLLECTOR),
            "--output", str(output), "--clusters", str(count),
            "--cluster-offset", str(offset),
            "--states-per-cluster", str(args.states_per_cluster),
            "--chance-repeats", str(args.chance_repeats),
            "--decision-step", str(args.decision_step),
            "--candidate-mode", args.candidate_mode,
            "--seed", str(args.seed), "--device", args.device,
            "--showdown-path", str(args.showdown_path),
            "--player-strategy", args.player_strategy,
            "--preview-strategy", args.preview_strategy,
        ]
        if args.checkpoint is not None:
            command.extend(["--checkpoint", str(args.checkpoint)])
        if args.opponent_checkpoint is not None:
            command.extend(["--opponent-checkpoint", str(args.opponent_checkpoint)])
            command.append("--paired-opponent-responses")
            if args.include_public_counterplay_response:
                command.append("--include-public-counterplay-response")
        elif args.include_public_counterplay_response:
            parser.error("--include-public-counterplay-response requires --opponent-checkpoint")
        if args.team_selection_checkpoint is not None:
            command.extend([
                "--team-selection-checkpoint", str(args.team_selection_checkpoint)
            ])
        configs.append({"index": index, "clusters": count, "offset": offset,
                        "output": output, "log": log, "command": command})
        offset += count

    started = time.perf_counter()
    active = []
    for item in configs:
        stream = item["log"].open("wb")
        process = subprocess.Popen(
            item["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT),
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1"},
        )
        item["process"] = process
        item["stream"] = stream
        item["log_position"] = 0
        active.append(item)

    try:
        while active:
            for item in list(active):
                item["stream"].flush()
                with item["log"].open("rb") as log_stream:
                    log_stream.seek(item["log_position"])
                    chunk = log_stream.read()
                    item["log_position"] += len(chunk)
                if chunk:
                    for line in chunk.decode("utf-8", errors="replace").splitlines():
                        if '"clusters_completed"' in line:
                            print(f"[shard {item['index']}] {line}", flush=True)
                status = item["process"].poll()
                if status is not None:
                    item["stream"].close()
                    if status:
                        tail = item["log"].read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-80:]
                        print("\n".join(tail), file=sys.stderr)
                        raise RuntimeError(
                            f"collection shard {item['index']} exited with {status}; "
                            f"see {item['log']}"
                        )
                    active.remove(item)
            if active:
                time.sleep(2)
    finally:
        for item in active:
            item["process"].terminate()
            item["stream"].close()

    shard_payloads = [json.loads(item["output"].read_text()) for item in configs]
    rows = [row for payload in shard_payloads for row in payload["rows"]]
    cluster_ids = {int(row["team_cluster"]) for row in rows}
    if len(cluster_ids) != args.clusters:
        raise RuntimeError(
            f"expected {args.clusters} unique team clusters, observed {len(cluster_ids)}"
        )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "opponent_checkpoint": str(args.opponent_checkpoint),
        "team_selection_checkpoint": str(args.team_selection_checkpoint),
        "format": shard_payloads[0]["format"],
        "seed": args.seed,
        "decision_step": args.decision_step,
        "clusters_requested": args.clusters,
        "states_per_cluster": args.states_per_cluster,
        "chance_repeats": args.chance_repeats,
        "candidate_mode": args.candidate_mode,
        "paired_response_classes": (
            ["max_damage", "public_counterplay", "policy"]
            if args.include_public_counterplay_response else
            (["max_damage", "policy"] if args.opponent_checkpoint is not None else [])
        ),
        "player_strategy": args.player_strategy,
        "preview_strategy": args.preview_strategy,
        "workers": args.workers,
        "states_captured": sum(p["states_captured"] for p in shard_payloads),
        "matches_executed": sum(p["matches_executed"] for p in shard_payloads),
        "elapsed_seconds": elapsed,
        "matches_per_second": sum(p["matches_executed"] for p in shard_payloads)
        / max(elapsed, 1e-9),
        "missing_states": sum(p["missing_states"] for p in shard_payloads),
        "rejected_candidates": sum(p["rejected_candidates"] for p in shard_payloads),
        "team_clusters": sorted(cluster_ids),
        "worker_shards": [
            {key: value for key, value in payload.items() if key != "rows"}
            for payload in shard_payloads
        ],
        "rows": rows,
    }
    final_path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "worker_shards", "team_clusters"}}, indent=2))


if __name__ == "__main__":
    main()
