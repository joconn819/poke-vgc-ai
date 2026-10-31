"""Run paired local-server evaluations with teams and sides swapped.

Each pair uses one fixed team pair and preview selection twice. The first leg
puts the checkpoint on player 1; the second swaps the teams and preview
choices, putting the same checkpoint on player 2. Randomness inside the local
server is independent between legs, but team/preview and side advantages are
controlled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from scripts.evaluate_native import evaluate
from scripts.run_showdown_self_play import (
    LOCAL_SERVER_CONFIGURATION,
    generate_showdown_champions_team_data,
)
from src.training.team_selection import TeamSelectionPolicy, random_action, team_features


def _bot_result(player1_result: str, *, bot_is_player1: bool) -> str:
    if bot_is_player1:
        return player1_result
    return {"win": "loss", "loss": "win", "draw": "draw"}.get(
        player1_result, player1_result
    )


def _save_team(path: Path, team: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(team, indent=2) + "\n", encoding="utf-8")


def _require_local_server() -> None:
    """Fail before generating teams or starting retry loops if Showdown is down."""
    url = urlparse(LOCAL_SERVER_CONFIGURATION.websocket_url)
    if not url.hostname:
        raise RuntimeError(
            f"invalid local Showdown websocket URL: "
            f"{LOCAL_SERVER_CONFIGURATION.websocket_url!r}"
        )
    port = url.port or (443 if url.scheme == "wss" else 80)
    try:
        with socket.create_connection((url.hostname, port), timeout=2):
            pass
    except OSError as error:
        raise RuntimeError(
            f"local Pokémon Showdown is not accepting connections at "
            f"{url.hostname}:{port} ({error}); start the configured local server "
            "before running paired evaluation"
        ) from error


def run_paired(
    *,
    checkpoint: Path,
    opponent: str,
    opponent_checkpoint: Path | None,
    pairs: int,
    outcome_repeats: int = 1,
    seed: int,
    output_dir: Path,
    battle_format: str = "gen9championsdoubles2026mcclosed",
    episode_timeout: int = 60,
    device: str = "cpu",
    deterministic: bool = True,
    player_view_output_dir: Path | None = None,
    trace_decisions: bool = False,
    team_selection_checkpoint: Path | None = None,
    opponent_team_selection_checkpoint: Path | None = None,
    max_damage_score_gate: bool = False,
    intervention_target: str = "both_policy_sides",
    first_turn_max_damage: bool = False,
    max_damage_action_steps: list[int] | None = None,
    public_counterplay_prior_weight: float = 0.0,
    public_counterplay_prior_min_margin: float = 0.0,
    public_counterplay_incoming_weight: float | None = None,
    public_counterplay_ko_bonus: float | None = None,
    team_seed_offset: int = 0,
    preview_seed_offset: int = 0,
) -> dict[str, Any]:
    if opponent not in {"random", "max_damage", "counterplay", "public_counterplay", "policy"}:
        raise ValueError(
            "paired evaluation supports random, max_damage, counterplay, "
            "public_counterplay, or policy opponents"
        )
    if opponent == "policy" and opponent_checkpoint is None:
        raise ValueError("--opponent-checkpoint is required for paired policy evaluation")
    if intervention_target not in {"both_policy_sides", "evaluated_checkpoint"}:
        raise ValueError("intervention_target must be both_policy_sides or evaluated_checkpoint")
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if outcome_repeats < 1:
        raise ValueError("outcome_repeats must be positive")
    if team_selection_checkpoint is None:
        print(
            "warning: no team-selection checkpoint supplied; preview teams/leads "
            "are selected randomly, so this is a policy-only score",
            file=sys.stderr,
            flush=True,
        )
    _require_local_server()
    output_dir.mkdir(parents=True, exist_ok=True)
    if player_view_output_dir is not None:
        player_view_output_dir.mkdir(parents=True, exist_ok=True)
    selector = None
    selector_config = None
    if team_selection_checkpoint is not None:
        selector_data = torch.load(team_selection_checkpoint, map_location="cpu", weights_only=False)
        selector_config = selector_data["model_config"]
        selector = TeamSelectionPolicy(
            selector_config["feature_dim"], selector_config["hidden_dim"],
            opponent_conditioned=selector_config.get("opponent_conditioned", False),
            action_conditioned=selector_config.get("action_conditioned", False),
        )
        selector.load_state_dict(selector_data["model_state_dict"])
        selector.eval()
    opponent_selector = None
    opponent_selector_config = None
    if opponent_team_selection_checkpoint is not None:
        selector_data = torch.load(
            opponent_team_selection_checkpoint, map_location="cpu", weights_only=False
        )
        opponent_selector_config = selector_data["model_config"]
        opponent_selector = TeamSelectionPolicy(
            opponent_selector_config["feature_dim"],
            opponent_selector_config["hidden_dim"],
            opponent_conditioned=opponent_selector_config.get("opponent_conditioned", False),
            action_conditioned=opponent_selector_config.get("action_conditioned", False),
        )
        opponent_selector.load_state_dict(selector_data["model_state_dict"])
        opponent_selector.eval()
    results = []
    for case_index in range(pairs * outcome_repeats):
        pair_index, repeat_index = divmod(case_index, outcome_repeats)
        team_seed = seed + team_seed_offset + pair_index * 2
        team_a = generate_showdown_champions_team_data(team_seed, battle_format)
        team_b = generate_showdown_champions_team_data(team_seed + 1, battle_format)
        retries = 0
        while team_a == team_b and retries < 10:
            retries += 1
            team_b = generate_showdown_champions_team_data(
                team_seed + 1 + retries * 100_000, battle_format
            )
        if team_a == team_b:
            raise RuntimeError(f"could not generate distinct teams for pair {pair_index}")

        team_a_path = output_dir / "teams" / f"pair_{pair_index:03d}_A.json"
        team_b_path = output_dir / "teams" / f"pair_{pair_index:03d}_B.json"
        _save_team(team_a_path, team_a)
        _save_team(team_b_path, team_b)
        if selector is None:
            preview_a = random_action(np.random.default_rng(seed + pair_index * 2))
        else:
            own_features = team_features(team_a, selector_config["feature_dim"]).unsqueeze(0)
            opponent_features = team_features(
                team_b, selector_config["feature_dim"], revealed_only=True
            ).unsqueeze(0)
            with torch.inference_mode():
                scores = selector(
                    own_features,
                    opponent_features=opponent_features
                    if selector_config.get("opponent_conditioned", False) else None,
                )
            preview_a = int(scores.argmax(dim=-1).item())
        if opponent_selector is None:
            preview_b = random_action(
                np.random.default_rng(seed + preview_seed_offset + pair_index * 2 + 1)
            )
        else:
            own_features = team_features(
                team_b, opponent_selector_config["feature_dim"]
            ).unsqueeze(0)
            opponent_features = team_features(
                team_a, opponent_selector_config["feature_dim"], revealed_only=True
            ).unsqueeze(0)
            with torch.inference_mode():
                scores = opponent_selector(
                    own_features,
                    opponent_features=opponent_features
                    if opponent_selector_config.get("opponent_conditioned", False) else None,
                )
            preview_b = int(scores.argmax(dim=-1).item())

        case_suffix = (
            f"pair_{pair_index:03d}_repeat_{repeat_index:02d}"
            if outcome_repeats > 1 else f"pair_{pair_index:03d}"
        )
        leg1_path = output_dir / f"{case_suffix}_checkpoint_p1.jsonl"
        leg2_path = output_dir / f"{case_suffix}_checkpoint_p2.jsonl"
        player_view_leg1 = (
            player_view_output_dir / f"{case_suffix}_p1.jsonl"
            if player_view_output_dir is not None else None
        )
        player_view_leg2 = (
            player_view_output_dir / f"{case_suffix}_p2.jsonl"
            if player_view_output_dir is not None else None
        )
        leg1_seed = seed + case_index
        leg2_seed = seed + 100_000 + case_index
        leg1: dict[str, Any] | None = None
        leg2: dict[str, Any] | None = None
        try:
            leg1 = evaluate(
                checkpoint=checkpoint,
                games=1,
                output=leg1_path,
                seed=leg1_seed,
                team_seed=team_seed,
                battle_format=battle_format,
                episode_timeout=episode_timeout,
                device=device,
                deterministic=deterministic,
                opponent=opponent,
                opponent_checkpoint=opponent_checkpoint,
                player="policy",
                team1_json=team_a_path,
                team2_json=team_b_path,
                preview_action_player1=preview_a,
                preview_action_player2=preview_b,
                player_view_output=player_view_leg1,
                trace_decisions=trace_decisions,
                max_damage_score_gate=max_damage_score_gate,
                intervention_target_player=(
                    0 if intervention_target == "evaluated_checkpoint" else None
                ),
                first_turn_max_damage=first_turn_max_damage,
                max_damage_action_steps=max_damage_action_steps,
                public_counterplay_prior_weight=public_counterplay_prior_weight,
                public_counterplay_prior_min_margin=public_counterplay_prior_min_margin,
                public_counterplay_incoming_weight=public_counterplay_incoming_weight,
                public_counterplay_ko_bonus=public_counterplay_ko_bonus,
            )
            # Give the reversed leg a distinct environment seed/account name while
            # preserving the same teams and the same preview action per roster.
            if opponent == "policy":
                # Reverse the two checkpoints as well as the teams so the
                # candidate remains the evaluated agent on player 2.
                leg2 = evaluate(
                    checkpoint=opponent_checkpoint,
                    games=1,
                    output=leg2_path,
                    seed=leg2_seed,
                    team_seed=team_seed,
                    battle_format=battle_format,
                    episode_timeout=episode_timeout,
                    device=device,
                    deterministic=deterministic,
                    opponent="policy",
                    opponent_checkpoint=checkpoint,
                    player="policy",
                    team1_json=team_b_path,
                    team2_json=team_a_path,
                    preview_action_player1=preview_b,
                    preview_action_player2=preview_a,
                    player_view_output=player_view_leg2,
                    trace_decisions=trace_decisions,
                    max_damage_score_gate=max_damage_score_gate,
                    intervention_target_player=(
                        1 if intervention_target == "evaluated_checkpoint" else None
                    ),
                    first_turn_max_damage=first_turn_max_damage,
                    max_damage_action_steps=max_damage_action_steps,
                    public_counterplay_prior_weight=public_counterplay_prior_weight,
                    public_counterplay_prior_min_margin=public_counterplay_prior_min_margin,
                    public_counterplay_incoming_weight=public_counterplay_incoming_weight,
                    public_counterplay_ko_bonus=public_counterplay_ko_bonus,
                )
            else:
                leg2 = evaluate(
                    checkpoint=None,
                    games=1,
                    output=leg2_path,
                    seed=leg2_seed,
                    team_seed=team_seed,
                    battle_format=battle_format,
                    episode_timeout=episode_timeout,
                    device=device,
                    deterministic=deterministic,
                    opponent="policy",
                    opponent_checkpoint=checkpoint,
                    player=opponent,
                    team1_json=team_b_path,
                    team2_json=team_a_path,
                    preview_action_player1=preview_b,
                    preview_action_player2=preview_a,
                    player_view_output=player_view_leg2,
                    trace_decisions=trace_decisions,
                    max_damage_score_gate=max_damage_score_gate,
                    intervention_target_player=(
                        1 if intervention_target == "evaluated_checkpoint" else None
                    ),
                    first_turn_max_damage=first_turn_max_damage,
                    max_damage_action_steps=max_damage_action_steps,
                    public_counterplay_prior_weight=public_counterplay_prior_weight,
                    public_counterplay_prior_min_margin=public_counterplay_prior_min_margin,
                    public_counterplay_incoming_weight=public_counterplay_incoming_weight,
                    public_counterplay_ko_bonus=public_counterplay_ko_bonus,
                )
            if not leg1_path.exists() or not leg1_path.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"first evaluation leg produced no result: {leg1_path}")
            if not leg2_path.exists() or not leg2_path.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"second evaluation leg produced no result: {leg2_path}")
            row1 = json.loads(leg1_path.read_text(encoding="utf-8").splitlines()[0])
            row2 = json.loads(leg2_path.read_text(encoding="utf-8").splitlines()[0])
        except Exception as error:
            results.append({
                "pair": pair_index,
                "repeat": repeat_index,
                "case_index": case_index,
                "team_seed": team_seed,
                "evaluation_seed_player1": leg1_seed,
                "evaluation_seed_player2": leg2_seed,
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
                "leg1_record": str(leg1_path),
                "leg2_record": str(leg2_path),
                "leg1_summary": leg1,
                "leg2_summary": leg2,
            })
            print(json.dumps(results[-1], sort_keys=True), flush=True)
            continue
        bot1 = _bot_result(row1["result"], bot_is_player1=True)
        bot2 = _bot_result(row2["result"], bot_is_player1=False)
        points = sum(1.0 if result == "win" else 0.5 if result == "draw" else 0.0
                     for result in (bot1, bot2))
        results.append({
            "pair": pair_index,
            "repeat": repeat_index,
            "case_index": case_index,
            "evaluation_seed_player1": leg1_seed,
            "evaluation_seed_player2": leg2_seed,
            "team_seed": team_seed,
            "team_a": [p.get("speciesId", p.get("species", "")) for p in team_a],
            "team_b": [p.get("speciesId", p.get("species", "")) for p in team_b],
            "preview_a": preview_a,
            "preview_b": preview_b,
            "checkpoint_player1_result": bot1,
            "checkpoint_player2_result": bot2,
            "checkpoint_points": points,
            "complete": row1["result"] in {"win", "loss", "draw"}
            and row2["result"] in {"win", "loss", "draw"},
            "action_adjustments": row1.get("action_adjustments", 0)
            + row2.get("action_adjustments", 0),
            "max_damage_score_gate_overrides": row1.get("max_damage_score_gate_overrides", 0)
            + row2.get("max_damage_score_gate_overrides", 0),
            "first_turn_max_damage_overrides": row1.get("first_turn_max_damage_overrides", 0)
            + row2.get("first_turn_max_damage_overrides", 0),
            "max_damage_action_overrides": row1.get("max_damage_action_overrides", 0)
            + row2.get("max_damage_action_overrides", 0),
            "leg1_record": str(leg1_path),
            "leg2_record": str(leg2_path),
            "player_view_leg1": str(player_view_leg1) if player_view_leg1 else None,
            "player_view_leg2": str(player_view_leg2) if player_view_leg2 else None,
            "leg1_summary": leg1,
            "leg2_summary": leg2,
        })
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    complete_pairs = [row for row in results if row["complete"]]
    leg_results = [
        row[key]
        for row in complete_pairs
        for key in ("checkpoint_player1_result", "checkpoint_player2_result")
    ]
    wins = leg_results.count("win")
    losses = leg_results.count("loss")
    draws = leg_results.count("draw")
    summary = {
        "checkpoint": str(checkpoint),
        "team_selection_checkpoint": str(team_selection_checkpoint) if team_selection_checkpoint else None,
        "opponent_team_selection_checkpoint": (
            str(opponent_team_selection_checkpoint)
            if opponent_team_selection_checkpoint else None
        ),
        "opponent": opponent,
        "deterministic": deterministic,
        "public_counterplay_prior_weight": public_counterplay_prior_weight,
        "public_counterplay_prior_min_margin": public_counterplay_prior_min_margin,
        "public_counterplay_incoming_weight": public_counterplay_incoming_weight,
        "public_counterplay_ko_bonus": public_counterplay_ko_bonus,
            "max_damage_score_gate": max_damage_score_gate,
            "intervention_target": intervention_target,
        "first_turn_max_damage": first_turn_max_damage,
        "max_damage_action_steps": max_damage_action_steps,
        "pairs_requested": pairs,
        "outcome_repeats_per_pair": outcome_repeats,
        "cases_requested": pairs * outcome_repeats,
        "cases_completed": len(complete_pairs),
        "team_pair_clusters_completed": len({row["pair"] for row in complete_pairs}),
        "pairs_completed": len(complete_pairs),
        "games_completed": len(leg_results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "max_damage_score_gate_overrides": sum(
            row.get("max_damage_score_gate_overrides", 0) for row in complete_pairs
        ),
        "first_turn_max_damage_overrides": sum(
            row.get("first_turn_max_damage_overrides", 0) for row in complete_pairs
        ),
        "max_damage_action_overrides": sum(
            row.get("max_damage_action_overrides", 0) for row in complete_pairs
        ),
        "side_swapped_score_rate": (
            (wins + 0.5 * draws) / len(leg_results) if leg_results else None
        ),
        "pair_sweeps": sum(row["checkpoint_points"] == 2 for row in complete_pairs),
        "pair_splits": sum(row["checkpoint_points"] == 1 for row in complete_pairs),
        "pair_losses": sum(row["checkpoint_points"] == 0 for row in complete_pairs),
        "timeouts_or_errors": pairs * outcome_repeats - len(complete_pairs),
        "action_adjustments": sum(row.get("action_adjustments", 0) for row in results),
        "results": results,
    }
    summary_path = output_dir / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        choices=("random", "max_damage", "counterplay", "public_counterplay", "policy"),
        default="max_damage",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument(
        "--outcome-repeats", type=int, default=1,
        help="Independent battle-seed repeats for each fixed team pair; each repeat includes both side-swapped legs.",
    )
    parser.add_argument("--seed", type=int, default=20261050)
    parser.add_argument(
        "--team-seed-offset", type=int, default=0,
        help="global team-pair seed offset used when running disjoint parallel shards",
    )
    parser.add_argument(
        "--preview-seed-offset", type=int, default=0,
        help="global preview RNG offset used when running disjoint parallel shards",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--episode-timeout", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stochastic", action="store_true",
        help="sample masked policy actions instead of taking argmax; applies to each policy player",
    )
    parser.add_argument("--max-damage-score-gate", action="store_true",
                        help="apply the immediate-damage gate to a policy side")
    parser.add_argument(
        "--intervention-target",
        choices=("both_policy_sides", "evaluated_checkpoint"),
        default="both_policy_sides",
        help=("apply an intervention to both policies for historical symmetric "
              "comparisons, or only to the evaluated checkpoint"),
    )
    parser.add_argument("--first-turn-max-damage", action="store_true",
                        help="use Max Damage for the evaluated policy's first action in both legs")
    parser.add_argument("--max-damage-action-steps", type=int, nargs="+",
                        help="evaluated-policy decision indices to replace with Max Damage")
    parser.add_argument("--public-counterplay-prior-weight", type=float, default=0.0)
    parser.add_argument("--public-counterplay-prior-min-margin", type=float, default=0.0)
    parser.add_argument("--public-counterplay-incoming-weight", type=float)
    parser.add_argument("--public-counterplay-ko-bonus", type=float)
    parser.add_argument(
        "--player-view-output-dir", type=Path,
        help="Optional directory for perspective-safe per-state evaluation records used to train the win-probability model.",
    )
    parser.add_argument(
        "--trace-decisions", action="store_true",
        help="Record preview selections and per-turn readable action traces in paired match records.",
    )
    parser.add_argument("--team-selection-checkpoint", type=Path,
                        help="Use this selector for the evaluated checkpoint's team preview.")
    parser.add_argument("--opponent-team-selection-checkpoint", type=Path,
                        help="Use this selector for the opponent side's team preview, including a fixed heuristic opponent.")
    args = vars(parser.parse_args())
    args["deterministic"] = not args.pop("stochastic")
    run_paired(**args)


if __name__ == "__main__":
    main()
