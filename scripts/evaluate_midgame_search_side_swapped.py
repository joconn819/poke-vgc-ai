#!/usr/bin/env python3
"""Paired headless evaluation of closed-sheet midgame search in both seats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_headless_showdown_self_play import (
    _load_policy, _load_team_selector, run_game,
)
from src.env.headless_showdown import HeadlessShowdownWorker
from src.models.state_value import load_state_value_checkpoint
from src.search.local_showdown import LocalShowdownSearchPool


def bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, (20000, values.size))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent-checkpoint", type=Path,
        help="optional frozen policy opponent; otherwise use Max Damage",
    )
    parser.add_argument(
        "--opponent-strategy", choices=("max_damage", "counterplay", "public_counterplay"),
        default="max_damage",
        help="built-in opponent policy used when --opponent-checkpoint is absent",
    )
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--team-selection-checkpoint", type=Path,
        help="pin the deployed team-preview selector on the evaluated policy side",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--showdown-path", type=Path, default=Path(
        "/home/justin/p/poke_learning/pokemon-showdown"))
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--seed", type=int, default=2026103701)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--hidden-worlds", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--max-response-candidates", type=int,
        help="opponent response columns; defaults to --max-candidates",
    )
    parser.add_argument(
        "--max-damage-response-candidates", type=int, default=1,
        help="response rows reserved for highest raw-damage joint actions",
    )
    parser.add_argument(
        "--response-selection-strategy",
        choices=("semantic", "slot_coverage", "damage_ranked"),
        default="semantic", help="opponent response shortlist rule",
    )
    parser.add_argument("--chance-samples", type=int, default=1)
    parser.add_argument("--max-chance-samples", type=int, default=None)
    parser.add_argument(
        "--search-selection", choices=("equilibrium", "maximin"),
        default="equilibrium",
    )
    parser.add_argument("--decision-timeout", type=float, default=5.0)
    parser.add_argument(
        "--frontier-calibration", type=float, nargs=2,
        metavar=("SLOPE", "INTERCEPT"),
        help="optional logistic calibration applied to learned search frontier values",
    )
    parser.add_argument(
        "--response-coverage-diagnostic", action="store_true",
        help="record whether each benchmark opponent action appeared in the search response set",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    response_candidates = (
        args.max_candidates if args.max_response_candidates is None
        else args.max_response_candidates
    )
    if args.clusters < 1 or args.search_workers < 1 or args.hidden_worlds < 1:
        parser.error("clusters, search workers, and hidden worlds must be positive")
    if (args.max_candidates < 4 or response_candidates < args.max_candidates
            or args.max_damage_response_candidates < 1
            or args.max_damage_response_candidates > response_candidates
            or args.chance_samples < 1 or args.decision_timeout <= 0):
        parser.error("search needs four candidates, one chance sample, and a positive timeout")
    if args.frontier_calibration is not None and (
        not np.isfinite(args.frontier_calibration).all()
        or args.frontier_calibration[0] <= 0
    ):
        parser.error("frontier calibration needs a positive finite slope and finite intercept")
    max_chance_samples = (
        args.chance_samples
        if args.max_chance_samples is None else args.max_chance_samples
    )
    if max_chance_samples < args.chance_samples:
        parser.error("maximum chance samples must be at least the initial chance-sample count")
    if max_chance_samples > args.chance_samples and args.search_selection != "maximin":
        parser.error("adaptive chance sampling currently requires --search-selection maximin")
    if (
        not args.checkpoint.is_file()
        or not args.value_checkpoint.is_file()
        or (args.team_selection_checkpoint is not None
            and not args.team_selection_checkpoint.is_file())
        or (args.opponent_checkpoint is not None and not args.opponent_checkpoint.is_file())
    ):
        parser.error("policy, state-value, and optional selector/opponent checkpoints must exist")
    if args.opponent_checkpoint is not None and args.opponent_strategy != "max_damage":
        parser.error("choose either a frozen opponent checkpoint or a built-in opponent strategy")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = _load_policy(args.checkpoint, args.device)
    opponent_model = (
        _load_policy(args.opponent_checkpoint, args.device)
        if args.opponent_checkpoint is not None else None
    )
    opponent_strategy = "policy" if opponent_model is not None else args.opponent_strategy
    evaluator = load_state_value_checkpoint(args.value_checkpoint, map_location=args.device)
    team_selection_model, team_selection_config = _load_team_selector(
        args.team_selection_checkpoint
    )
    rows: list[dict] = []
    team_rosters: list[dict] = []
    matches = 0
    started = time.perf_counter()
    with HeadlessShowdownWorker(args.showdown_path) as worker, LocalShowdownSearchPool(
        format_id=args.format,
        showdown_path=args.showdown_path,
        workers=args.search_workers,
        timeout_seconds=args.decision_timeout,
    ) as search_pool:
        for cluster in range(args.clusters):
            cluster_seed = args.seed + cluster * 10007
            teams = (
                worker.generate_team(cluster_seed, args.format),
                worker.generate_team(cluster_seed + 1, args.format),
            )
            team_rosters.append({
                "cluster": cluster,
                "team_seeds": [cluster_seed, cluster_seed + 1],
                # These species are public at team preview. Do not persist the
                # generated move/item/ability details in the evaluation report.
                "public_species": [
                    [str(member.get("species") or member.get("name") or "") for member in team]
                    for team in teams
                ],
            })
            for bot_side in (0, 1):
                for use_search in (False, True):
                    result = run_game(
                        game=0,
                        seed=cluster_seed + (100_003 if bot_side else 0),
                        format_id=args.format,
                        showdown_path=args.showdown_path,
                        output_path=args.output.with_suffix(".runner.jsonl"),
                        model=model,
                        opponent_model=opponent_model,
                        player_strategy=(
                            "policy" if bot_side == 0 or opponent_model is not None
                            else args.opponent_strategy
                        ),
                        opponent_strategy=(opponent_strategy if bot_side == 0 else "policy"),
                        trainable_side="player1" if bot_side == 0 else "player2",
                        temperature=1.0,
                        device=args.device,
                        telemetry=None,
                        worker=worker,
                        deterministic_policy=True,
                        teams_override=teams,
                        swap_teams=bool(bot_side),
                        persist_record=False,
                        team_selection_model=team_selection_model,
                        team_selection_config=team_selection_config,
                        state_value_evaluator=evaluator if use_search else None,
                        search_pool=search_pool if use_search else None,
                        midgame_search="learned" if use_search else "none",
                        midgame_search_worlds=args.hidden_worlds,
                        search_max_candidates=args.max_candidates,
                        midgame_search_max_response_candidates=response_candidates,
                        midgame_search_max_damage_response_candidates=(
                            args.max_damage_response_candidates
                        ),
                        midgame_search_response_selection_strategy=(
                            args.response_selection_strategy
                        ),
                        midgame_search_frontier_calibration=(
                            tuple(args.frontier_calibration)
                            if args.frontier_calibration is not None else None
                        ),
                        midgame_search_chance_samples=args.chance_samples,
                        midgame_search_max_chance_samples=max_chance_samples,
                        midgame_search_max_decisions=1,
                        midgame_search_policy_mixture=0.0,
                        search_response_coverage_diagnostic=args.response_coverage_diagnostic,
                        midgame_search_timeout_seconds=args.decision_timeout,
                        search_action_selection=args.search_selection,
                    )
                    matches += 1
                    bot_player = "Headless P1" if bot_side == 0 else "Headless P2"
                    reports = result.get("midgame_search_reports", []) if use_search else []
                    rows.append({
                        "cluster": cluster,
                        "bot_side": bot_side,
                        "use_search": use_search,
                        "bot_win": int(result.get("winner") == bot_player),
                        "completed": bool(result.get("winner")),
                        "steps": int(result.get("steps", 0)),
                        "search_attempts": int(result.get("midgame_search_attempts", 0)),
                        "search_teacher_decisions": int(result.get("midgame_search_teacher_decisions", 0)),
                        "search_action_changes": int(result.get("midgame_search_action_changes", 0)),
                        "search_failures": [
                            report.get("error") for report in reports if report.get("error")
                        ],
                        "search_reports": reports,
                    })
            if (cluster + 1) % 8 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(json.dumps({
                    "clusters_completed": cluster + 1,
                    "matches": matches,
                    "matches_per_second": matches / elapsed,
                    "search_failures": sum(bool(r["search_failures"]) for r in rows),
                }), flush=True)

    runner_output = args.output.with_suffix(".runner.jsonl")
    runner_output.unlink(missing_ok=True)
    summary = {}
    for side in (0, 1):
        group = [row for row in rows if row["bot_side"] == side]
        raw = {row["cluster"]: row["bot_win"] for row in group if not row["use_search"]}
        searched = {row["cluster"]: row["bot_win"] for row in group if row["use_search"]}
        keys = sorted(set(raw) & set(searched))
        delta = np.asarray([searched[key] - raw[key] for key in keys], dtype=np.float64)
        summary["p1" if side == 0 else "p2"] = {
            "clusters": len(keys),
            "raw_win_rate": float(np.mean([raw[key] for key in keys])) if keys else None,
            "search_win_rate": float(np.mean([searched[key] for key in keys])) if keys else None,
            "search_minus_raw": float(delta.mean()) if delta.size else None,
            "paired_cluster_bootstrap_95_ci": bootstrap_interval(delta, args.seed + side + 1),
            "search_attempts": sum(row["search_attempts"] for row in group if row["use_search"]),
            "successful_search_decisions": sum(row["search_teacher_decisions"] for row in group if row["use_search"]),
            "changed_policy_actions": sum(row["search_action_changes"] for row in group if row["use_search"]),
            "games_with_search_failures": sum(bool(row["search_failures"]) for row in group if row["use_search"]),
        }
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "information_boundary": "closed-sheet sampled hidden worlds; each acting policy sees only its player-view observation",
        "format": args.format,
        "policy_checkpoint": str(args.checkpoint),
        "opponent_checkpoint": (
            str(args.opponent_checkpoint) if args.opponent_checkpoint is not None else None
        ),
        "opponent": (
            "frozen_policy" if opponent_model is not None else args.opponent_strategy
        ),
        "state_value_checkpoint": str(args.value_checkpoint),
        "team_selection_checkpoint": (
            str(args.team_selection_checkpoint)
            if args.team_selection_checkpoint is not None else None
        ),
        "seed": args.seed,
        "clusters_requested": args.clusters,
        "search_config": {
            "hidden_worlds": args.hidden_worlds,
            "max_candidates": args.max_candidates,
            "max_response_candidates": response_candidates,
            "max_damage_response_candidates": args.max_damage_response_candidates,
            "response_selection_strategy": args.response_selection_strategy,
            "chance_samples": args.chance_samples,
            "maximum_chance_samples": max_chance_samples,
            "decision_timeout_seconds": args.decision_timeout,
            "search_workers": args.search_workers,
            "decision_limit_per_match": 1,
            "selection": args.search_selection,
            "response_coverage_diagnostic": args.response_coverage_diagnostic,
            "frontier_calibration": args.frontier_calibration,
            "opponent_strategy": args.opponent_strategy,
        },
        "matches": matches,
        "elapsed_seconds": elapsed,
        "matches_per_second": matches / max(elapsed, 1e-9),
        "team_rosters": team_rosters,
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
