"""Run deterministic paired local-Showdown evaluation shards concurrently.

Each worker receives a disjoint team-seed range and writes its own raw match
records. The final paired summary combines team-pair clusters without changing
the side-swapped scoring unit used by the serial evaluator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _worker_command(args: argparse.Namespace, worker: int, start_pair: int, pairs: int) -> list[str]:
    worker_dir = args.output_dir / f"worker_{worker:02d}"
    # Keep both teams and per-battle evaluator seeds identical to a serial run.
    # Team-pair RNG advances twice per pair, while battle seeds advance once per
    # evaluated outcome, so these offsets must be tracked independently.
    seed = args.seed + start_pair * args.outcome_repeats
    pair_seed_offset = start_pair * (2 - args.outcome_repeats)
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_native_paired.py")),
        "--checkpoint", str(args.checkpoint),
        "--opponent", args.opponent,
        "--pairs", str(pairs),
        "--outcome-repeats", str(args.outcome_repeats),
        "--seed", str(seed),
        "--team-seed-offset", str(pair_seed_offset),
        "--preview-seed-offset", str(pair_seed_offset),
        "--output-dir", str(worker_dir),
        "--battle-format", args.battle_format,
        "--episode-timeout", str(args.episode_timeout),
        "--device", args.device,
    ]
    if args.stochastic:
        command.append("--stochastic")
    if args.public_counterplay_prior_weight > 0:
        command.extend((
            "--public-counterplay-prior-weight",
            str(args.public_counterplay_prior_weight),
        ))
    if args.public_counterplay_prior_min_margin > 0:
        command.extend((
            "--public-counterplay-prior-min-margin",
            str(args.public_counterplay_prior_min_margin),
        ))
    if args.public_counterplay_incoming_weight is not None:
        command.extend((
            "--public-counterplay-incoming-weight",
            str(args.public_counterplay_incoming_weight),
        ))
    if args.public_counterplay_ko_bonus is not None:
        command.extend((
            "--public-counterplay-ko-bonus",
            str(args.public_counterplay_ko_bonus),
        ))
    optional_paths = (
        ("--opponent-checkpoint", args.opponent_checkpoint),
        ("--team-selection-checkpoint", args.team_selection_checkpoint),
        ("--opponent-team-selection-checkpoint", args.opponent_team_selection_checkpoint),
    )
    for flag, path in optional_paths:
        if path is not None:
            command.extend((flag, str(path)))
    if args.max_damage_score_gate:
        command.append("--max-damage-score-gate")
    if args.intervention_target != "both_policy_sides":
        command.extend(("--intervention-target", args.intervention_target))
    if args.first_turn_max_damage:
        command.append("--first-turn-max-damage")
    if args.max_damage_action_steps is not None:
        command.extend(("--max-damage-action-steps", *map(str, args.max_damage_action_steps)))
    if args.trace_decisions:
        command.append("--trace-decisions")
    return command


def _run_worker(
    args: argparse.Namespace,
    worker: int,
    start_pair: int,
    pairs: int,
) -> dict[str, Any]:
    worker_dir = args.output_dir / f"worker_{worker:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    command = _worker_command(args, worker, start_pair, pairs)
    log_path = worker_dir / "evaluation.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            # Each evaluator is a separate Python/Torch process.  Letting
            # every worker create a machine-sized BLAS/OpenMP thread pool
            # oversubscribes CPUs badly (on this host it spawned ~160 threads
            # per worker and reduced throughput by nearly 10x).
            env={
                **os.environ,
                # Directly executing a script places its `scripts/` directory
                # on sys.path, but this worker imports project packages from
                # the repository root (for example `scripts.evaluate_native`).
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(Path(__file__).resolve().parents[1]),
                            os.environ.get("PYTHONPATH", ""),
                        ),
                    )
                ),
                "OMP_NUM_THREADS": os.environ.get("POKEMON_EVAL_TORCH_THREADS", "1"),
                "MKL_NUM_THREADS": os.environ.get("POKEMON_EVAL_TORCH_THREADS", "1"),
                "OPENBLAS_NUM_THREADS": os.environ.get("POKEMON_EVAL_TORCH_THREADS", "1"),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"evaluation worker {worker} failed with exit code {process.returncode}; "
            f"see {log_path}\n{tail}"
        )
    summary_path = worker_dir / "paired_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"evaluation worker {worker} produced no {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "worker": worker,
        "start_pair": start_pair,
        "pairs_requested": pairs,
        "elapsed_seconds": elapsed,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "summary": summary,
    }


def run_parallel(args: argparse.Namespace) -> dict[str, Any]:
    if args.pairs < 1 or args.workers < 1:
        raise ValueError("pairs and workers must be positive")
    if args.opponent == "policy" and args.opponent_checkpoint is None:
        raise ValueError("--opponent-checkpoint is required for policy opponents")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.opponent_checkpoint is not None:
        args.opponent_checkpoint = args.opponent_checkpoint.expanduser().resolve()
    if args.team_selection_checkpoint is not None:
        args.team_selection_checkpoint = args.team_selection_checkpoint.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a fresh path: {args.output_dir}"
        )

    worker_count = min(args.workers, args.pairs)
    quotient, remainder = divmod(args.pairs, worker_count)
    jobs: list[tuple[int, int, int]] = []
    offset = 0
    for worker in range(worker_count):
        count = quotient + (worker < remainder)
        jobs.append((worker, offset, count))
        offset += count

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_worker, args, *job) for job in jobs]
        # Await every shard so completed workers' summaries remain available
        # even if another worker fails.
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    records: list[dict[str, Any]] = []
    shard_summaries = []
    seen_clusters: set[tuple[int, int]] = set()
    for shard in results:
        summary = shard["summary"]
        shard_summaries.append({key: value for key, value in shard.items() if key != "summary"})
        for row in summary.get("results", []):
            key = (int(row["team_seed"]), int(row.get("repeat", 0)))
            if key in seen_clusters:
                raise RuntimeError(f"overlapping team-pair cluster in shards: {key}")
            seen_clusters.add(key)
            records.append(row)

    records.sort(key=lambda row: (int(row["team_seed"]), int(row.get("repeat", 0))))
    for index, row in enumerate(records):
        row["pair"] = index
        row["case_index"] = index
    complete = [row for row in records if row.get("complete")]
    wins = sum(int(shard["summary"].get("wins", 0)) for shard in results)
    losses = sum(int(shard["summary"].get("losses", 0)) for shard in results)
    draws = sum(int(shard["summary"].get("draws", 0)) for shard in results)
    games_completed = sum(
        int(row.get("leg1_summary", {}).get("games", 0))
        + int(row.get("leg2_summary", {}).get("games", 0))
        for row in complete
    )
    if wins + losses + draws != games_completed:
        raise RuntimeError(
            "merged W/L/D totals do not match the completed game count: "
            f"{wins}+{losses}+{draws}!={games_completed}"
        )
    checkpoints_points = [float(row.get("checkpoint_points", 0.0)) for row in complete]
    paired = {
        "checkpoint": str(args.checkpoint),
        "opponent": args.opponent,
        "deterministic": not args.stochastic,
        "public_counterplay_prior_weight": args.public_counterplay_prior_weight,
        "public_counterplay_prior_min_margin": args.public_counterplay_prior_min_margin,
        "public_counterplay_incoming_weight": args.public_counterplay_incoming_weight,
        "public_counterplay_ko_bonus": args.public_counterplay_ko_bonus,
        "pairs_requested": args.pairs * args.outcome_repeats,
        "pairs_completed": len(complete),
        "games_completed": games_completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "side_swapped_score_rate": (
            (wins + 0.5 * draws) / max(1, games_completed)
        ),
        "pair_sweeps": sum(points == 2.0 for points in checkpoints_points),
        "pair_splits": sum(points == 1.0 for points in checkpoints_points),
        "pair_losses": sum(points == 0.0 for points in checkpoints_points),
        "timeouts_or_errors": len(records) - len(complete),
        "action_adjustments": sum(int(row.get("action_adjustments", 0)) for row in records),
        "max_damage_score_gate": bool(args.max_damage_score_gate),
        "intervention_target": args.intervention_target,
        "max_damage_score_gate_overrides": sum(
            int(row.get("max_damage_score_gate_overrides", 0)) for row in records
        ),
        "first_turn_max_damage": bool(args.first_turn_max_damage),
        "max_damage_action_steps": args.max_damage_action_steps,
        "first_turn_max_damage_overrides": sum(
            int(row.get("first_turn_max_damage_overrides", 0)) for row in records
        ),
        "max_damage_action_overrides": sum(
            int(row.get("max_damage_action_overrides", 0)) for row in records
        ),
        "outcome_repeats_per_pair": args.outcome_repeats,
        "team_pair_clusters_completed": len({int(row["team_seed"]) for row in complete}),
        "team_selection_checkpoint": (
            str(args.team_selection_checkpoint) if args.team_selection_checkpoint else None
        ),
        "opponent_team_selection_checkpoint": (
            str(args.opponent_team_selection_checkpoint)
            if args.opponent_team_selection_checkpoint else None
        ),
        "parallel_workers": worker_count,
        "parallel_wall_seconds": wall_seconds,
        "aggregate_games_per_second": (
            games_completed / wall_seconds if wall_seconds > 0 else None
        ),
        "worker_shards": shard_summaries,
        "results": records,
    }
    summary_path = args.output_dir / "paired_summary.json"
    summary_path.write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in paired.items() if key != "results"}, sort_keys=True))
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        choices=("random", "max_damage", "counterplay", "public_counterplay", "policy"),
        default="max_damage",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--outcome-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261050)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--episode-timeout", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stochastic", action="store_true",
        help="sample masked policy actions instead of taking argmax; applies to each policy player",
    )
    parser.add_argument("--max-damage-score-gate", action="store_true")
    parser.add_argument(
        "--intervention-target",
        choices=("both_policy_sides", "evaluated_checkpoint"),
        default="both_policy_sides",
        help="apply interventions to both policies or only to the evaluated checkpoint",
    )
    parser.add_argument("--first-turn-max-damage", action="store_true")
    parser.add_argument("--max-damage-action-steps", type=int, nargs="+")
    parser.add_argument("--public-counterplay-prior-weight", type=float, default=0.0)
    parser.add_argument("--public-counterplay-prior-min-margin", type=float, default=0.0)
    parser.add_argument("--public-counterplay-incoming-weight", type=float)
    parser.add_argument("--public-counterplay-ko-bonus", type=float)
    parser.add_argument(
        "--trace-decisions", action="store_true",
        help="Record preview selections and per-turn readable action traces.",
    )
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument("--opponent-team-selection-checkpoint", type=Path)
    run_parallel(parser.parse_args())


if __name__ == "__main__":
    main()
