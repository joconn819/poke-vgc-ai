"""Collect perspective-safe self-play trajectories on the local Showdown engine.

This runner skips the websocket server during rollout collection while keeping
Showdown's Battle API as the authoritative simulator and the existing
PolicyAgent/observation/action interfaces as the acting and data boundaries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from scripts.run_showdown_self_play import (
    choose_masked_actions,
)
from src.env.headless_showdown import HeadlessShowdownProcess, HeadlessShowdownWorker
from src.env.showdown_env import ShowdownDoublesEnv
from src.features.public_mechanics import encode_public_mechanics
from src.models.policy import TransformerPolicy
from src.models.state_value import StateValueEvaluator, load_state_value_checkpoint
from src.baselines.native_damage import (
    choose_counterplay_action,
    choose_max_damage_action,
    choose_public_counterplay_action,
    score_max_damage_action,
)
from src.training.observability import TrainingTelemetry
from src.training.native_ppo import NATIVE_ACTION_FEATURE_ENCODER_VERSION
from src.training.perspective_data import project_episode_to_player_views
from src.training.self_play import PolicyAgent
from src.search.local_showdown import LocalShowdownSearchPool, choose_opening_search_action
from src.search.local_showdown import choose_sanitized_state_search_action
from src.belief.closed_sheet import (
    ClosedSheetBeliefTracker,
    possible_selected_rosters,
    sample_showdown_hidden_teams,
)
from src.training.team_selection import (
    TeamSelectionPolicy,
    decode_action,
    preview_command,
    random_action,
    team_features,
)


def _load_policy(
    path: Path | None,
    device: str,
    *,
    compile_policy: bool = False,
    compile_mode: str = "reduce-overhead",
) -> TransformerPolicy | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TransformerPolicy.from_checkpoint_state(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    if compile_policy:
        model = torch.compile(model, mode=compile_mode)
    return model


def _load_team_selector(path: Path | None) -> tuple[TeamSelectionPolicy | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = TeamSelectionPolicy(
        int(config["feature_dim"]),
        int(config["hidden_dim"]),
        opponent_conditioned=bool(config.get("opponent_conditioned", False)),
        action_conditioned=bool(config.get("action_conditioned", False)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def _public_species(team: Any) -> list[dict[str, str]]:
    members = team.values() if isinstance(team, dict) else team or ()
    result = []
    for pokemon in members:
        species = getattr(pokemon, "species", None)
        if species and not any(row["species"] == species for row in result):
            result.append({"species": str(species)})
    return result[:6]


def _public_species_for_battle(battle: Any) -> list[dict[str, str]]:
    """Use the closed-sheet preview roster in the same stable order as the view."""
    preview = getattr(battle, "teampreview_opponent_team", None)
    return _public_species(preview or getattr(battle, "opponent_team", ()))


def _public_preview_may_have_illusion(species: list[str]) -> bool:
    pokedex = GenData.from_gen(9).pokedex
    for name in species:
        entry = pokedex.get(to_id_str(name), {})
        if any(
            to_id_str(ability) == "illusion"
            for ability in entry.get("abilities", {}).values()
        ):
            return True
    return False


def _safe_policy_for(
    model: TransformerPolicy,
    player_team: list[dict[str, Any]],
    battle: Any,
    *,
    device: str,
    temperature: float,
) -> PolicyAgent:
    # The engine's hidden opponent team is not passed through this boundary.
    return PolicyAgent(
        model,
        player_team,
        _public_species_for_battle(battle),
        deterministic=False,
        temperature=temperature,
        device=device,
    )


def _opening_search(
    match: HeadlessShowdownProcess,
    *,
    player_index: int,
    own_team: list[dict[str, Any]],
    own_preview_action: int,
    public_opponent_species: list[str],
    seed: int,
    format_id: str,
    showdown_path: Path,
    mode: str,
    hidden_worlds: int,
    max_candidates: int,
    max_response_candidates: int,
    chance_samples: int,
    candidate_damage_samples: int,
    action_selection_method: str,
    workers: int,
    return_response_candidates: bool,
    evaluator: StateValueEvaluator | None,
    search_pool: LocalShowdownSearchPool | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode not in {"material", "learned"}:
        raise ValueError(f"unsupported opening search mode: {mode}")
    selection = decode_action(int(own_preview_action))
    own_preview = list(selection.leads + selection.back)
    action, report = choose_opening_search_action(
        match.battles[player_index],
        own_team,
        own_preview,
        opponent_preview_species=public_opponent_species,
        format_id=format_id,
        showdown_path=showdown_path,
        seed=seed,
        hidden_worlds=hidden_worlds,
        max_candidates=max_candidates,
        max_response_candidates=max_response_candidates,
        chance_samples=chance_samples,
        candidate_damage_samples=candidate_damage_samples,
        action_selection_method=action_selection_method,
        return_response_candidates=return_response_candidates,
        workers=workers,
        evaluator=evaluator if mode == "learned" else None,
        search_pool=search_pool,
    )
    summary = {
        key: report[key]
        for key in (
            "selected_action", "selected_action_index", "robust_scores",
            "matrix_game_method", "matrix_game_shape", "matrix_game_value",
            "matrix_game_exploitability",
            "mixed_row_strategy", "mixed_column_strategy",
            "frontier", "frontier_value", "world_count", "chance_samples",
            "simulated_turn_advances", "elapsed_ms",
            "snapshot_candidate_generation_ms",
            "p1_candidate_generation_ms", "p2_candidate_generation_ms",
            "estimated_damage_scoring_ms",
            "candidate_consistency_filter_ms", "rollout_loop_elapsed_ms",
            "rollout_json_parse_ms", "rollout_deserialization_ms",
            "turn_advance_ms", "material_value_ms", "battle_cleanup_ms",
            "transitions_per_second", "search_subprocess_wall_ms",
            "matrix_selection_ms",
            "value_inference_ms", "value_inference_device", "search_workers",
            "frontier_application_ms", "hidden_hypothesis_setup_ms",
            "action_translation_ms", "search_adapter_wall_ms",
        )
        if key in report
    }
    summary["candidate_commands"] = [
        candidate["command"] for candidate in report.get("own_candidates", ())
    ]
    summary["own_candidates"] = [
        {
            key: candidate.get(key)
            for key in ("command", "kinds", "actions", "score", "maxDamageScore")
            if key in candidate
        }
        for candidate in report.get("own_candidates", ())
    ]
    if return_response_candidates:
        summary["response_candidates_by_world"] = [
            {
                "world_index": world.get("world_index"),
                "actions": [candidate.get("actions") for candidate in world.get(
                    "opponent_response_candidates", ()
                )],
            }
            for world in report.get("world_reports", ())
        ]
    return action, summary


def _midgame_search(
    match: HeadlessShowdownProcess,
    *,
    acting_side: int,
    preferred_action_indices: np.ndarray,
    own_team: list[dict[str, Any]],
    own_preview_action: int,
    public_opponent_species: list[str],
    seed: int,
    format_id: str,
    showdown_path: Path,
    hidden_worlds: int,
    max_candidates: int,
    max_response_candidates: int | None,
    max_damage_response_candidates: int,
    response_selection_strategy: str,
    frontier_calibration: tuple[float, float] | None,
    chance_samples: int,
    max_chance_samples: int | None,
    action_selection_method: str,
    evaluator: StateValueEvaluator,
    search_pool: LocalShowdownSearchPool,
    timeout_seconds: float,
    return_response_candidates: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build public-compatible hidden worlds and search the acting player's turn."""
    if acting_side not in {0, 1}:
        raise ValueError("acting_side must be P1 (0) or P2 (1)")
    battle = match.battles[acting_side]
    if battle.turn <= 1:
        raise ValueError("midgame search starts after the opening turn")
    if any(battle.force_switch):
        raise ValueError("midgame matrix search skips forced-switch requests")
    species = [str(row) for row in public_opponent_species]
    if len(species) != 6:
        raise ValueError("midgame search needs the six public preview species")
    if _public_preview_may_have_illusion(species):
        raise ValueError(
            "midgame search skips public previews containing a possible Illusion user"
        )
    preferred_order = ShowdownDoublesEnv.action_to_order(
        preferred_action_indices, battle, strict=True
    )
    preferred_actions = []
    for slot, single_order in enumerate((
        preferred_order.first_order,
        preferred_order.second_order,
    )):
        chosen = single_order.order
        if hasattr(chosen, "base_species"):
            semantic = {
                "type": "switch",
                "species": to_id_str(str(chosen.species)),
            }
        elif hasattr(chosen, "id"):
            semantic = {
                "type": "move",
                "move_id": str(chosen.id),
                "target": int(single_order.move_target),
                "mega": bool(single_order.mega),
                "terastallize": bool(single_order.terastallize),
            }
            live_request = getattr(battle, "_last_request", {}) or {}
            active_requests = live_request.get("active", [])
            active_request = active_requests[slot] if slot < len(active_requests) else {}
            moves = active_request.get("moves", [])
            if len(moves) == 1 and moves[0].get("pp") is None:
                semantic["forced_targetless"] = True
        elif str(single_order) == "/choose pass":
            semantic = {"type": "pass"}
        else:
            raise ValueError(f"cannot represent the policy proposal for search: {single_order}")
        preferred_actions.append(semantic)
    belief = ClosedSheetBeliefTracker(species)
    belief.update_from_battle(battle)
    leads = belief.opening_leads()
    if len(leads) != 2:
        raise ValueError("midgame search could not recover both public opening leads")
    selection_hypotheses = possible_selected_rosters(
        species, leads, belief.appeared_species()
    )
    preview_index = {to_id_str(name): index for index, name in enumerate(species)}
    selection_indexes = [
        [preview_index[to_id_str(name)] for name in selection]
        for selection in selection_hypotheses
    ]
    if not selection_indexes:
        raise ValueError("no public-compatible opponent team selection remains")

    own_selection = decode_action(int(own_preview_action))
    own_preview = list(own_selection.leads + own_selection.back)
    try:
        hypotheses = sample_showdown_hidden_teams(
            species,
            belief.public_facts(),
            format_id=format_id,
            count=hidden_worlds,
            seed=seed,
            showdown_path=showdown_path,
            sampler=search_pool.hidden_sampler,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        raise RuntimeError(f"hidden-team sampling failed: {error}") from error
    selection_rng = np.random.default_rng(seed + 31337)
    worlds = [
        {
            "opponent_team": hypothesis,
            "opponent_preview": selection_indexes[int(selection_rng.integers(len(selection_indexes)))],
        }
        for hypothesis in hypotheses
    ]
    prepared = match.prepare_search_worlds({
        "acting_side": acting_side,
        "seed": int(seed),
        "own_preview": own_preview,
        "opponent_preview_species": species,
        "revealed_facts": belief.public_facts(),
        "hypotheses": worlds,
    })
    action, report = choose_sanitized_state_search_action(
        battle,
        prepared["snapshots"],
        prepared["worlds"],
        format_id=format_id,
        showdown_path=showdown_path,
        seed=seed,
        max_candidates=max_candidates,
        max_response_candidates=max_response_candidates,
        max_damage_response_candidates=max_damage_response_candidates,
        response_selection_strategy=response_selection_strategy,
        frontier_calibration=frontier_calibration,
        chance_samples=chance_samples,
        max_chance_samples=max_chance_samples,
        action_selection_method=action_selection_method,
        evaluator=evaluator,
        search_pool=search_pool,
        preferred_own_action=preferred_actions,
        return_response_candidates=return_response_candidates,
        timeout_seconds=timeout_seconds,
    )
    report["preferred_own_action"] = preferred_actions
    report["preferred_action_candidate_indexes"] = [
        index
        for index, candidate in enumerate(report.get("own_candidates", ()))
        if candidate.get("actions") == preferred_actions
    ]
    report["hidden_hypothesis_setup_ms"] = float(prepared.get("setup_ms", 0.0))
    report["selection_hypothesis_count"] = len(selection_indexes)
    return action, report


def _semantic_native_action_pair(
    actions: list[int] | np.ndarray, battle: Any
) -> list[dict[str, Any]]:
    """Decode a native pair for offline diagnostics; never used in search choice."""
    order = ShowdownDoublesEnv.action_to_order(
        np.asarray(actions, dtype=np.int64), battle, strict=True
    )
    semantic_actions = []
    request = getattr(battle, "_last_request", {}) or {}
    active_requests = request.get("active", [])
    for slot, single_order in enumerate((order.first_order, order.second_order)):
        chosen = single_order.order
        if hasattr(chosen, "id"):
            semantic = {
                "type": "move",
                "move_id": str(chosen.id),
                "target": int(getattr(single_order, "move_target", 0)),
                "mega": bool(getattr(single_order, "mega", False)),
                "terastallize": bool(getattr(single_order, "terastallize", False)),
            }
            active = active_requests[slot] if slot < len(active_requests) else {}
            move_slots = active.get("moves", [])
            if len(move_slots) == 1 and move_slots[0].get("pp") is None:
                semantic["forced_targetless"] = True
        elif hasattr(chosen, "species"):
            semantic = {
                "type": "switch",
                "species": to_id_str(str(chosen.species)),
            }
        elif str(single_order) == "/choose pass":
            semantic = {"type": "pass"}
        else:
            raise ValueError(f"cannot semantically decode native action {single_order}")
        semantic_actions.append(semantic)
    return semantic_actions


def _legal_native_action_descriptors(
    mask: list[int] | np.ndarray, battle: Any
) -> list[list[dict[str, Any]]]:
    """Describe current legal individual choices for offline diagnostics.

    This is emitted only when a counterfactual state is explicitly captured.
    It contains the acting player's current legal actions, never simulated
    opponent sets or actions.
    """
    from poke_env.environment.doubles_env import DoublesEnv

    values = np.asarray(mask, dtype=np.int8).reshape(-1)
    width = values.size // 2
    pokedex_moves = GenData.from_gen(9).moves
    protect_moves = {
        "protect", "detect", "spikyshield", "banefulbunker", "kingsshield",
        "silktrap", "burningbulwark", "obstruct", "maxguard", "wideguard",
        "quickguard", "endure",
    }
    redirection_moves = {"followme", "ragepowder", "allyswitch"}
    descriptors: list[list[dict[str, Any]]] = [[], []]
    for slot in range(2):
        for action_index in np.flatnonzero(values[slot * width:(slot + 1) * width]):
            try:
                order = DoublesEnv._action_to_order_individual(
                    np.int64(action_index), battle, True, slot
                )
            except (IndexError, ValueError):
                continue
            chosen = getattr(order, "order", None)
            if hasattr(chosen, "id"):
                move_id = to_id_str(str(chosen.id))
                dex_move = pokedex_moves.get(move_id, {})
                categories: list[str] = []
                if move_id in protect_moves:
                    categories.extend(("protect", "defensive"))
                if move_id in redirection_moves:
                    categories.extend(("redirection", "defensive"))
                if dex_move.get("boosts"):
                    categories.extend(("setup", "defensive"))
                target = str(dex_move.get("target", ""))
                if target in {"allAdjacent", "allAdjacentFoes", "all"}:
                    categories.append("spread")
                move_category = str(dex_move.get("category", ""))
                if move_category == "Status" and "setup" not in categories:
                    categories.append("status")
                elif move_category in {"Physical", "Special"}:
                    categories.append("attack")
                descriptors[slot].append({
                    "action_index": int(action_index),
                    "kind": "move",
                    "move_id": move_id,
                    "target": int(getattr(order, "move_target", 0)),
                    "mechanic": (
                        "tera" if bool(getattr(order, "terastallize", False))
                        else "mega" if bool(getattr(order, "mega", False))
                        else "base"
                    ),
                    "categories": sorted(set(categories or ["move"])),
                })
            elif hasattr(chosen, "species"):
                descriptors[slot].append({
                    "action_index": int(action_index),
                    "kind": "switch",
                    "species": to_id_str(str(chosen.species)),
                    "categories": ["defensive", "switch"],
                })
            else:
                descriptors[slot].append({
                    "action_index": int(action_index),
                    "kind": "pass",
                    "categories": ["defensive", "pass"],
                })
    return descriptors


def run_game(
    *,
    game: int,
    seed: int,
    format_id: str,
    showdown_path: Path,
    output_path: Path,
    model: TransformerPolicy | None,
    opponent_model: TransformerPolicy | None,
    player_strategy: str,
    opponent_strategy: str,
    temperature: float,
    device: str,
    telemetry: TrainingTelemetry | None,
    worker: HeadlessShowdownWorker,
    opening_search: str = "none",
    midgame_search: str = "none",
    state_value_evaluator: StateValueEvaluator | None = None,
    search_hidden_worlds: int = 2,
    search_max_candidates: int = 4,
    search_max_response_candidates: int | None = None,
    search_chance_samples: int = 4,
    search_candidate_damage_samples: int = 4,
    search_action_selection: str = "equilibrium",
    search_workers: int = 4,
    midgame_search_worlds: int = 2,
    midgame_search_max_response_candidates: int | None = None,
    midgame_search_max_damage_response_candidates: int = 1,
    midgame_search_response_selection_strategy: str = "semantic",
    midgame_search_frontier_calibration: tuple[float, float] | None = None,
    midgame_search_chance_samples: int = 1,
    midgame_search_max_chance_samples: int | None = None,
    midgame_search_timeout_seconds: float = 3.0,
    midgame_search_max_decisions: int = 0,
    midgame_search_policy_mixture: float = 0.0,
    search_response_coverage_diagnostic: bool = False,
    search_pool: LocalShowdownSearchPool | None = None,
    episode_offset: int = 0,
    record_episode_offset: int = 0,
    deterministic_policy: bool = False,
    first_turn_max_damage: bool = False,
    max_damage_action_steps: set[int] | None = None,
    opponent_max_damage_action_steps: set[int] | None = None,
    max_damage_action_slots: set[int] | None = None,
    first_turn_forced_action: list[int] | None = None,
    forced_opponent_action: list[int] | None = None,
    first_turn_forced_joint_actions: tuple[list[int], list[int]] | None = None,
    forced_action_step: int = 0,
    max_damage_action_step: int = 0,
    capture_action_step: int | None = None,
    capture_battle_snapshot: bool = False,
    capture_policy_action_log_prob: bool = False,
    capture_legal_joint_action_log_probs: bool = False,
    reseed_random_at_decision: tuple[int, int] | None = None,
    counterfactual_action_selector: Any | None = None,
    counterfactual_selector_decision: int = 5,
    teams_override: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
    preview_actions_override: tuple[int, int] | None = None,
    initial_battle_snapshot: dict[str, Any] | None = None,
    initial_parser_snapshot: dict[str, Any] | None = None,
    initial_decision_counts: tuple[int, int] = (0, 0),
    persist_record: bool = True,
    max_damage_score_gate: bool = False,
    max_damage_gate_on_policy: bool = False,
    gate_policy_opponent: bool = False,
    counterplay_teacher: bool = False,
    counterplay_incoming_weight: float = 1.0,
    counterplay_ko_bonus: float = 0.7,
    counterplay_setup_bonus: float = 0.0,
    counterplay_redirection: bool = False,
    swap_teams: bool = False,
    trainable_side: str = "player1",
    team_selection_model: TeamSelectionPolicy | None = None,
    team_selection_config: dict[str, Any] | None = None,
    team_selection_temperature: float = 0.0,
    opponent_team_selection_model: TeamSelectionPolicy | None = None,
    opponent_team_selection_config: dict[str, Any] | None = None,
    opponent_team_selection_temperature: float | None = None,
) -> dict[str, Any]:
    if first_turn_forced_action is not None and first_turn_forced_joint_actions is not None:
        raise ValueError("use either first_turn_forced_action or first_turn_forced_joint_actions")
    if forced_opponent_action is not None and first_turn_forced_action is None:
        raise ValueError("a forced opponent action requires a forced player1 action")
    if max_damage_gate_on_policy and not max_damage_score_gate:
        raise ValueError("on-policy max-damage gate mode requires the max-damage score gate")
    if gate_policy_opponent and max_damage_gate_on_policy:
        raise ValueError("policy-opponent gate cannot be combined with on-policy gated PPO")
    if not 0.0 <= midgame_search_policy_mixture <= 1.0:
        raise ValueError("midgame search policy mixture must be in [0, 1]")
    if (initial_battle_snapshot is None) != (initial_parser_snapshot is None):
        raise ValueError("restored battles require both simulator and player-view parser snapshots")
    if len(initial_decision_counts) != 2 or any(int(value) < 0 for value in initial_decision_counts):
        raise ValueError("initial decision counts must contain two nonnegative integers")
    if reseed_random_at_decision is not None:
        reseed_step, reseed_seed = reseed_random_at_decision
        if reseed_step < 0 or not 0 <= reseed_seed <= 2**53 - 1:
            raise ValueError("battle RNG reseed needs a nonnegative decision index and safe integer seed")
    if trainable_side not in {"player1", "player2"}:
        raise ValueError("trainable_side must be player1 or player2")
    if not math.isfinite(team_selection_temperature) or team_selection_temperature < 0:
        raise ValueError("team_selection_temperature must be finite and nonnegative")
    if opponent_team_selection_temperature is None:
        opponent_team_selection_temperature = team_selection_temperature
    if (not math.isfinite(opponent_team_selection_temperature)
            or opponent_team_selection_temperature < 0):
        raise ValueError("opponent team-selection temperature must be finite and nonnegative")
    # Make policy sampling reproducible per episode even when independent
    # rollout jobs run concurrently in separate processes.
    episode_index = episode_offset + game
    record_episode = record_episode_offset + game
    torch.manual_seed(seed + episode_index)
    started_at = time.perf_counter()
    if teams_override is None:
        team1 = worker.generate_team(seed + 2 * episode_index, format_id)
        team2 = worker.generate_team(seed + 2 * episode_index + 1, format_id)
    else:
        team1, team2 = deepcopy(teams_override)
    if swap_teams:
        team1, team2 = team2, team1
    rngs = [np.random.default_rng(seed + episode_index * 2), np.random.default_rng(seed + episode_index * 2 + 1)]
    preview_actions = list(preview_actions_override) if preview_actions_override is not None else [
        random_action(rngs[0]), random_action(rngs[1])
    ]
    team_selection_applied = False
    team_selection_side: int | None = None
    opponent_team_selection_applied = False
    opponent_team_selection_side: int | None = None
    preview_log_probs: dict[str, float] = {}
    def choose_preview(model: TeamSelectionPolicy, config: dict[str, Any] | None,
                       selected_side: int, selection_temperature: float) -> int:
        if config is None:
            raise ValueError("team-selection model needs its saved model configuration")
        own_team, opposing_team = (
            (team1, team2) if selected_side == 0 else (team2, team1)
        )
        feature_dim = int(config["feature_dim"])
        own_features = team_features(own_team, feature_dim).unsqueeze(0)
        # Opponent team sheets are closed: only the six preview species are
        # visible to a selector when scoring its own preview choice.
        opponent_features = team_features(
            opposing_team, feature_dim, revealed_only=True
        ).unsqueeze(0)
        with torch.inference_mode():
            scores = model(
                own_features,
                opponent_features=opponent_features
                if model.opponent_conditioned else None,
            )
            preview_logits = scores.squeeze(0)
            if selection_temperature > 0:
                log_probs = torch.log_softmax(preview_logits / selection_temperature, dim=-1)
                probabilities = log_probs.exp().cpu().numpy()
                selected_action = int(rngs[selected_side].choice(len(probabilities), p=probabilities))
                preview_log_probs["player1" if selected_side == 0 else "player2"] = float(
                    log_probs[selected_action].item()
                )
            else:
                selected_action = int(preview_logits.argmax(dim=-1).item())
        return selected_action

    if preview_actions_override is None:
        selected_side = 0 if trainable_side == "player1" else 1
        if team_selection_model is not None:
            selected_strategy = player_strategy if selected_side == 0 else opponent_strategy
            if selected_strategy not in {"policy", "public_counterplay"}:
                raise ValueError(
                    "team selection can only be applied to a policy or public-counterplay side"
                )
            selected_action = choose_preview(
                team_selection_model, team_selection_config, selected_side,
                team_selection_temperature,
            )
            team_selection_applied = True
            team_selection_side = selected_side
            preview_actions[selected_side] = selected_action

        other_side = 1 - selected_side
        other_strategy = player_strategy if other_side == 0 else opponent_strategy
        if opponent_team_selection_model is not None and other_strategy == "policy":
            selected_action = choose_preview(
                opponent_team_selection_model, opponent_team_selection_config, other_side,
                opponent_team_selection_temperature,
            )
            opponent_team_selection_applied = True
            opponent_team_selection_side = other_side
            preview_actions[other_side] = selected_action
    agent_names = ["Headless P1", "Headless P2"]
    agent_to_team = {agent_names[0]: "player1", agent_names[1]: "player2"}
    if trainable_side == "player2" and opponent_strategy not in {"policy", "public_counterplay"}:
        raise ValueError("player2 evaluation requires the selected policy in the P2 seat")
    policies: dict[str, PolicyAgent] = {}
    if player_strategy == "policy":
        if model is None:
            raise ValueError("--policy-checkpoint is required for player_strategy=policy")
        policies[agent_names[0]] = PolicyAgent(
            (opponent_model or model) if trainable_side == "player2" else model,
            team1, [], deterministic=deterministic_policy,
            temperature=temperature, device=device
        )
    if opponent_strategy == "policy":
        selected_model = model if trainable_side == "player2" else (opponent_model or model)
        if selected_model is None:
            raise ValueError("a policy checkpoint is required for opponent_strategy=policy")
        policies[agent_names[1]] = PolicyAgent(
            selected_model, team2, [], deterministic=deterministic_policy,
            temperature=temperature, device=device
        )

    rng = np.random.default_rng(seed + episode_index * 1009)
    trajectory: list[dict[str, Any]] = []
    first_turn_action_data: dict[str, Any] | None = None
    first_turn_joint_action_data: dict[str, dict[str, Any]] | None = None
    counterfactual_action_data: dict[str, Any] | None = None
    counterfactual_action_data_by_agent: dict[str, dict[str, Any]] = {}
    counterfactual_joint_action_data: dict[str, dict[str, Any]] | None = None
    first_turn_action_source = "policy"
    joint_steps = action_adjustments = 0
    player1_decisions = int(initial_decision_counts[0])
    player2_decisions = int(initial_decision_counts[1])
    baseline_fallbacks = 0
    baseline_fallback_reasons: list[str] = []
    max_damage_score_gate_overrides = 0
    policy_opponent_score_gate_overrides = 0
    first_turn_max_damage_override_actions = 0
    max_damage_action_fallbacks = 0
    opponent_max_damage_override_actions = 0
    counterplay_teacher_actions = 0
    counterfactual_selector_report: dict[str, Any] | None = None
    action_seconds = engine_roundtrip_seconds = observation_seconds = 0.0
    policy_inference_seconds = baseline_action_seconds = action_proposal_seconds = 0.0
    preview_seconds = 0.0
    search_report: dict[str, Any] | None = None
    search_failure_reason: str | None = None
    midgame_search_reports: list[dict[str, Any]] = []
    midgame_search_teacher_decisions = 0
    midgame_search_attempts = 0
    midgame_search_action_changes = 0
    midgame_search_policy_branch_decisions = 0
    wall_started = time.perf_counter()
    with HeadlessShowdownProcess(
        team1,
        team2,
        format_id=format_id,
        seed=seed + episode_index,
        showdown_path=showdown_path,
        worker=worker,
        snapshot=initial_battle_snapshot,
        parser_snapshot=initial_parser_snapshot,
    ) as match:
        if initial_battle_snapshot is None:
            preview_started = time.perf_counter()
            state = match.advance(
                preview_command(preview_actions[0]).removeprefix("/"),
                preview_command(preview_actions[1]).removeprefix("/"),
            )
            preview_seconds = time.perf_counter() - preview_started
        else:
            if preview_actions_override is None:
                raise ValueError("restored battle requires the original preview actions")
            state = match.state
        while not match.ended:
            observation_started = time.perf_counter()
            obs1, mask1, obs2, mask2 = match.observations()
            observation_seconds += time.perf_counter() - observation_started
            observations = {agent_names[0]: obs1, agent_names[1]: obs2}
            masks = {agent_names[0]: mask1, agent_names[1]: mask2}
            mechanics_by_player = match.public_mechanics()
            public_mechanics = {
                agent_names[0]: mechanics_by_player[0],
                agent_names[1]: mechanics_by_player[1],
            }
            opponent_species_by_agent = {
                agent_name: [
                    member["species"]
                    for member in _public_species_for_battle(match.battles[index])
                ]
                for index, agent_name in enumerate(agent_names)
            }
            for index, agent_name in enumerate(agent_names):
                policy = policies.get(agent_name)
                if policy is not None:
                    policy.opponent_team = [
                        {"species": species}
                        for species in opponent_species_by_agent[agent_name]
                    ]
                if policy is not None and getattr(policy.model, "public_context_dim", 0):
                    policy.public_context_features = encode_public_mechanics(
                        public_mechanics[agent_name],
                        observations[agent_name],
                        policy.player_team,
                        policy.opponent_team,
                        format_id,
                    )
            recorded_masks = {agent: mask.copy() for agent, mask in masks.items()}
            requests = state.get("requests", [None, None])
            player1_decision_index = (
                player1_decisions
                if requests[0] is not None and not requests[0].get("wait")
                else None
            )
            player2_decision_index = (
                player2_decisions
                if requests[1] is not None and not requests[1].get("wait")
                else None
            )
            if player1_decision_index is not None:
                player1_decisions += 1
            if player2_decision_index is not None:
                player2_decisions += 1
            action_started = time.perf_counter()
            action_features_by_agent: dict[str, list[list[list[float]]]] = {}
            teacher_actions_this_step: set[str] = set()
            teacher_joint_distributions_this_step: dict[str, dict[str, list]] = {}
            search_action_behavior_log_probs_this_step: dict[str, float] = {}
            max_damage_gate_teacher_actions: set[str] = set()
            policy_proposed_actions_this_step: dict[str, list[int]] = {}
            forced_action_fallback: list[int] | None = None
            forced_opponent_action_fallback: list[int] | None = None
            forced_action_accepted = True
            forced_action_rejection_error: str | None = None
            if all(request and request.get("wait") for request in requests):
                # This is not expected for a live action state; ask Showdown to
                # resolve its defaults instead of fabricating a policy action.
                actions = {agent_names[0]: np.array([0, 0]), agent_names[1]: np.array([0, 0])}
                choices = ("", "")
                outputs: dict[str, tuple[np.ndarray, float, float]] = {}
            else:
                payloads = {
                    agent: {
                        "observation": observations[agent],
                        "action_mask": masks[agent],
                        "wait": bool(requests[index] and requests[index].get("wait")),
                        "active_request_summary": [
                            {
                                "forceSwitch": bool(row.get("forceSwitch")),
                                "moves_count": len(row.get("moves", ())),
                                "trapped": bool(row.get("trapped")),
                                "maybeTrapped": bool(row.get("maybeTrapped")),
                                "active_present": (
                                    match.battles[index].active_pokemon[i] is not None
                                    if i < len(match.battles[index].active_pokemon) else False
                                ),
                                "valid_order_count": (
                                    len(match.battles[index].valid_orders[i])
                                    if i < len(match.battles[index].valid_orders) else 0
                                ),
                            }
                            for i, row in enumerate(((requests[index] or {}).get("active") or []))
                        ],
                    }
                    for index, agent in enumerate(agent_names)
                }
                proposal_started = time.perf_counter()
                actions = choose_masked_actions(payloads, rng)
                action_proposal_seconds += time.perf_counter() - proposal_started
                outputs = {}
                pair_predictions: dict[str, tuple[np.ndarray, float, float]] = {}
                forced_targetless_search_positions: list[int] = []
                first_agent, second_agent = agent_names
                first_policy, second_policy = policies.get(first_agent), policies.get(second_agent)
                can_batch_pair = (
                    first_policy is not None
                    and second_policy is not None
                    and first_policy.model is second_policy.model
                    and requests[0] is not None
                    and requests[1] is not None
                    and not requests[0].get("wait")
                    and not requests[1].get("wait")
                    and not (opening_search != "none" and joint_steps == 0)
                    and not (
                        first_turn_max_damage and max_damage_action_steps is not None
                    )
                    and opponent_max_damage_action_steps is None
                )
                if can_batch_pair:
                    first_policy.opponent_team = [
                        {"species": species}
                        for species in opponent_species_by_agent[agent_names[0]]
                    ]
                    second_policy.opponent_team = [
                        {"species": species}
                        for species in opponent_species_by_agent[agent_names[1]]
                    ]
                    inference_started = time.perf_counter()
                    first_prediction, second_prediction = PolicyAgent.predict_native_pair_with_value(
                        first_policy,
                        second_policy,
                        observations[first_agent],
                        masks[first_agent],
                        observations[second_agent],
                        masks[second_agent],
                        use_critic=True,
                    )
                    policy_inference_seconds += time.perf_counter() - inference_started
                    pair_predictions = {
                        first_agent: first_prediction,
                        second_agent: second_prediction,
                    }
                for index, agent in enumerate(agent_names):
                    policy = policies.get(agent)
                    request = requests[index]
                    if request is None or request.get("wait"):
                        continue
                    if policy is None:
                        if (
                            index == 0
                            and first_turn_forced_action is not None
                            and player1_decision_index == forced_action_step
                        ):
                            forced = np.asarray(first_turn_forced_action, dtype=np.int64)
                            legal = np.asarray(masks[agent], dtype=np.int8)
                            width = legal.size // 2
                            if forced.shape != (2,) or legal.size % 2:
                                raise ValueError("forced action must be a pair of native action indices")
                            if any(
                                action < 0 or action >= width
                                or legal[slot * width + action] != 1
                                for slot, action in enumerate(forced.tolist())
                            ):
                                raise ValueError(
                                    f"forced action {forced.tolist()} is not legal under the current mask"
                                )
                            try:
                                match.choices_from_native_actions([
                                    forced, actions[agent_names[1]],
                                ])
                            except ValueError as error:
                                forced_action_accepted = False
                                forced_action_rejection_error = str(error)
                                native_action = choose_public_counterplay_action(
                                    masks[agent], match.battles[index], match.public_history,
                                    incoming_weight=counterplay_incoming_weight,
                                    ko_bonus=counterplay_ko_bonus,
                                    setup_bonus=counterplay_setup_bonus,
                                    redirect_actions=counterplay_redirection,
                                    order_validator=ShowdownDoublesEnv.action_to_order,
                                )
                            else:
                                native_action = forced
                            actions[agent] = np.asarray(native_action, dtype=np.int64)
                            first_turn_action_source = "counterfactual"
                            teacher_actions_this_step.add(agent)
                        continue
                    if agent in pair_predictions:
                        native_action, log_prob, value = pair_predictions[agent]
                    elif (
                        index == 0
                        and first_turn_max_damage
                        and player1_decision_index in (
                            max_damage_action_steps
                            if max_damage_action_steps is not None
                            else {max_damage_action_step}
                        )
                    ):
                        policy.opponent_team = [
                            {"species": species}
                            for species in opponent_species_by_agent[agent]
                        ]
                        inference_started = time.perf_counter()
                        policy_action, log_prob, value = policy.predict_native_with_value(
                            observations[agent], masks[agent], use_critic=True
                        )
                        policy_inference_seconds += time.perf_counter() - inference_started
                        max_damage_pair = choose_max_damage_action(
                            masks[agent], match.battles[0],
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        native_action = np.asarray(policy_action, dtype=np.int64)
                        slots_to_override = (
                            {0, 1} if max_damage_action_slots is None
                            else max_damage_action_slots
                        )
                        for action_slot in slots_to_override:
                            if action_slot not in {0, 1}:
                                raise ValueError("max_damage_action_slots must contain only 0 and 1")
                            native_action[action_slot] = max_damage_pair[action_slot]
                        try:
                            match.choices_from_native_actions([
                                native_action,
                                actions[agent_names[1]],
                            ])
                        except ValueError:
                            # A one-slot hybrid can create an incompatible
                            # joint order, such as both active Pokémon
                            # switching to the same teammate. Keep the policy
                            # pair when the proposed hybrid is invalid.
                            native_action = np.asarray(policy_action, dtype=np.int64)
                            max_damage_action_fallbacks += 1
                            try:
                                match.choices_from_native_actions([
                                    native_action,
                                    actions[agent_names[1]],
                                ])
                            except ValueError:
                                # Forced-switch masks can occasionally admit
                                # two copies of one switch target. Find the
                                # first strictly valid pair as a safe fallback.
                                legal_mask = np.asarray(masks[agent], dtype=np.int8)
                                width = legal_mask.size // 2
                                legal0 = np.flatnonzero(legal_mask[:width]).tolist()
                                legal1 = np.flatnonzero(legal_mask[width:]).tolist()
                                valid_fallback = None
                                for first in legal0:
                                    for second in legal1:
                                        if first == second and 1 <= first <= 6:
                                            continue
                                        candidate = np.asarray([first, second], dtype=np.int64)
                                        try:
                                            match.choices_from_native_actions([
                                                candidate,
                                                actions[agent_names[1]],
                                            ])
                                        except ValueError:
                                            continue
                                        valid_fallback = candidate
                                        break
                                    if valid_fallback is not None:
                                        break
                                if valid_fallback is None:
                                    raise RuntimeError(
                                        "no strictly valid action pair found after Max Damage intervention rejection"
                                    )
                                native_action = valid_fallback
                        first_turn_action_source = "max_damage"
                        first_turn_max_damage_override_actions += int(bool(slots_to_override))
                        # Retain the intervention as a teacher-labeled policy
                        # transition. `log_prob` remains the parent's sampled
                        # proposal probability, so training must treat this as
                        # off-policy teacher data rather than a PPO sample.
                        teacher_actions_this_step.add(agent)
                    elif (
                        index == 0
                        and first_turn_forced_action is not None
                        and player1_decision_index == forced_action_step
                    ):
                        forced = np.asarray(first_turn_forced_action, dtype=np.int64)
                        legal = np.asarray(masks[agent])
                        if forced.shape != (2,) or legal.ndim != 1 or legal.size % 2:
                            raise ValueError("forced action must be a legal pair of native action indices")
                        slot_width = legal.size // 2
                        if any(
                            action < 0 or action >= slot_width or legal[slot * slot_width + action] != 1
                            for slot, action in enumerate(forced.tolist())
                        ):
                            raise ValueError(f"forced action {forced.tolist()} is not legal under the current mask")
                        if policy is not None:
                            policy.opponent_team = [
                                {"species": species}
                                for species in opponent_species_by_agent[agent]
                            ]
                            inference_started = time.perf_counter()
                            policy_action, log_prob, value = policy.predict_native_with_value(
                                observations[agent], masks[agent], use_critic=True
                            )
                            policy_inference_seconds += time.perf_counter() - inference_started
                            forced_action_fallback = np.asarray(
                                policy_action, dtype=np.int64
                            ).tolist()
                        else:
                            # Counterfactual replays can intervene on a
                            # rule-based baseline too; retain its original
                            # legal action as the fallback/logged proposal.
                            forced_action_fallback = np.asarray(
                                native_action, dtype=np.int64
                            ).tolist()
                        native_action = forced.tolist()
                        first_turn_action_source = "counterfactual"
                        teacher_actions_this_step.add(agent)
                    elif (
                        index == (0 if trainable_side == "player1" else 1)
                        and opening_search != "none"
                        and joint_steps == 0
                    ):
                        own_team = team1 if index == 0 else team2
                        own_preview_action = preview_actions[index]
                        public_species = opponent_species_by_agent[agent]
                        try:
                            actions[agent], search_report = _opening_search(
                                match,
                                player_index=index,
                                own_team=own_team,
                                own_preview_action=own_preview_action,
                                public_opponent_species=public_species,
                                seed=seed + episode_index * 1009 + 1,
                                format_id=format_id,
                                showdown_path=showdown_path,
                                mode=opening_search,
                                hidden_worlds=search_hidden_worlds,
                                max_candidates=search_max_candidates,
                                max_response_candidates=(
                                    search_max_candidates
                                    if search_max_response_candidates is None
                                    else search_max_response_candidates
                                ),
                                chance_samples=search_chance_samples,
                                candidate_damage_samples=search_candidate_damage_samples,
                                action_selection_method=search_action_selection,
                                workers=search_workers,
                                return_response_candidates=search_response_coverage_diagnostic,
                                evaluator=state_value_evaluator,
                                search_pool=search_pool,
                            )
                            continue
                        except Exception as error:
                            # Search is optional at runtime. If belief sampling
                            # or a worker fails, use the normal legal policy
                            # action and keep the failure visible in the record.
                            search_failure_reason = f"{type(error).__name__}: {error}"
                            search_report = {
                                "error": search_failure_reason,
                                "fallback": "policy",
                            }
                            policy.opponent_team = [
                                {"species": species}
                                for species in opponent_species_by_agent[agent]
                            ]
                            inference_started = time.perf_counter()
                            native_action, log_prob, value = policy.predict_native_with_value(
                                observations[agent], masks[agent], use_critic=True
                            )
                            policy_inference_seconds += time.perf_counter() - inference_started
                    else:
                        mirror = match.battles[index]
                        # Refresh the species-only opponent view from the parsed
                        # public log before every policy call.
                        policy.opponent_team = [
                            {"species": species}
                            for species in opponent_species_by_agent[agent]
                        ]
                        inference_started = time.perf_counter()
                        native_action, log_prob, value = policy.predict_native_with_value(
                            observations[agent], masks[agent], use_critic=True
                        )
                        policy_inference_seconds += time.perf_counter() - inference_started
                    if (
                        index == (0 if trainable_side == "player1" else 1)
                        and midgame_search == "learned"
                        and joint_steps > 0
                        and match.state.get("request_state") == "move"
                        and (
                            midgame_search_max_decisions == 0
                            or midgame_search_attempts < midgame_search_max_decisions
                        )
                    ):
                        if state_value_evaluator is None or search_pool is None:
                            raise RuntimeError("learned midgame search needs its evaluator and search pool")
                        midgame_search_attempts += 1
                        policy_action_before_search = np.asarray(
                            native_action, dtype=np.int64
                        ).copy()
                        try:
                            searched_action, midgame_report = _midgame_search(
                                match,
                                acting_side=index,
                                preferred_action_indices=policy_action_before_search,
                                own_team=team1 if index == 0 else team2,
                                own_preview_action=preview_actions[index],
                                public_opponent_species=opponent_species_by_agent[agent],
                                seed=seed + episode_index * 1009 + joint_steps * 17 + 7,
                                format_id=format_id,
                                showdown_path=showdown_path,
                                hidden_worlds=midgame_search_worlds,
                                max_candidates=search_max_candidates,
                                max_response_candidates=(
                                    search_max_candidates
                                    if midgame_search_max_response_candidates is None
                                    else midgame_search_max_response_candidates
                                ),
                                max_damage_response_candidates=(
                                    midgame_search_max_damage_response_candidates
                                ),
                                response_selection_strategy=(
                                    midgame_search_response_selection_strategy
                                ),
                                frontier_calibration=midgame_search_frontier_calibration,
                                chance_samples=midgame_search_chance_samples,
                                max_chance_samples=midgame_search_max_chance_samples,
                                action_selection_method=search_action_selection,
                                evaluator=state_value_evaluator,
                                search_pool=search_pool,
                                timeout_seconds=midgame_search_timeout_seconds,
                                return_response_candidates=search_response_coverage_diagnostic,
                            )
                            # The search adapter translates semantic Showdown
                            # orders back to native action indices. Check that
                            # translation against the exact policy mask before
                            # recording it as a teacher decision. Rarely, a
                            # target or transformed-move edge case can make an
                            # otherwise accepted Showdown order disagree with
                            # the native mask; in that case keep the policy
                            # action and omit the bad teacher label.
                            search_mask = np.asarray(masks[agent], dtype=np.int8)
                            search_action = np.asarray(searched_action, dtype=np.int64)
                            if (
                                search_mask.ndim != 1
                                or search_mask.size % 2
                                or search_action.shape != (2,)
                            ):
                                raise ValueError("search returned an invalid native action or mask")
                            slot_width = search_mask.size // 2
                            illegal_search_slots = [
                                {
                                    "slot": slot,
                                    "action": int(action),
                                    "legal_actions": np.flatnonzero(
                                        search_mask[slot * slot_width:(slot + 1) * slot_width]
                                    ).tolist(),
                                }
                                for slot, action in enumerate(search_action)
                                if (
                                    int(action) < 0
                                    or int(action) >= slot_width
                                    or not search_mask[slot * slot_width + int(action)]
                                )
                            ]
                            if illegal_search_slots:
                                live_battle = match.battles[index]
                                live_request = getattr(live_battle, "_last_request", {}) or {}
                                live_active_requests = live_request.get("active", [])
                                live_diagnostics = []
                                for detail in illegal_search_slots:
                                    slot = detail["slot"]
                                    active_request = (
                                        live_active_requests[slot]
                                        if slot < len(live_active_requests) else {}
                                    )
                                    active_pokemon = (
                                        live_battle.active_pokemon[slot]
                                        if slot < len(live_battle.active_pokemon) else None
                                    )
                                    live_diagnostics.append({
                                        **detail,
                                        "request_moves": active_request.get("moves", []),
                                        "native_known_moves": [
                                            move.id for move in list(active_pokemon.moves.values())[:4]
                                        ] if active_pokemon is not None else [],
                                        "valid_orders": [
                                            str(order) for order in live_battle.valid_orders[slot]
                                        ],
                                    })
                                raise ValueError(
                                    "search action is not legal under the live native action mask: "
                                    f"selected={search_action.tolist()}, slots={live_diagnostics}, "
                                    f"matrix_choice={midgame_report.get('selected_action')}"
                                )
                            search_action_changed_policy = not np.array_equal(
                                search_action, policy_action_before_search
                            )
                            # Keep the baseline proposal in the same semantic
                            # action vocabulary as the searched candidates.
                            # Native indices alone are not enough to diagnose
                            # candidate pruning when active move ordering differs.
                            try:
                                policy_order = ShowdownDoublesEnv.action_to_order(
                                    policy_action_before_search,
                                    match.battles[index],
                                    strict=True,
                                )
                                policy_command = str(policy_order)
                            except Exception as error:
                                policy_command = None
                                policy_command_error = f"{type(error).__name__}: {error}"
                            else:
                                policy_command_error = None
                            candidate_diagnostics = [
                                {
                                    "command": candidate.get("command"),
                                    "action_indices": candidate.get("action_indices"),
                                    "actions": candidate.get("actions"),
                                }
                                for candidate in midgame_report.get("own_candidates", ())
                            ]
                            mixed_strategy = np.asarray(
                                midgame_report.get("mixed_row_strategy", ()),
                                dtype=np.float64,
                            )
                            if mixed_strategy.shape != (len(candidate_diagnostics),):
                                raise ValueError(
                                    "search equilibrium strategy does not align with candidates"
                                )
                            if (
                                not np.isfinite(mixed_strategy).all()
                                or (mixed_strategy < 0).any()
                                or float(mixed_strategy.sum()) <= 0
                            ):
                                raise ValueError("search equilibrium strategy is invalid")
                            mixed_strategy /= mixed_strategy.sum()
                            joint_actions = []
                            joint_probabilities = []
                            for candidate, probability in zip(
                                candidate_diagnostics, mixed_strategy, strict=True
                            ):
                                if probability <= 0:
                                    continue
                                action_indices = candidate.get("action_indices")
                                if (
                                    not isinstance(action_indices, (list, tuple))
                                    or len(action_indices) != 2
                                ):
                                    raise ValueError(
                                        "search candidate has no native joint action indices"
                                    )
                                joint_actions.append(
                                    [int(action_indices[0]), int(action_indices[1])]
                                )
                                joint_probabilities.append(float(probability))
                            if not joint_actions:
                                raise ValueError("search equilibrium strategy has empty support")
                            selected_candidate_index = int(
                                midgame_report.get("selected_action_index", -1)
                            )
                            if not 0 <= selected_candidate_index < len(mixed_strategy):
                                raise ValueError(
                                    "search selected candidate is outside its equilibrium strategy"
                                )
                            selected_probability = float(
                                mixed_strategy[selected_candidate_index]
                            )
                            if selected_probability <= 0:
                                raise ValueError(
                                    "search selected a zero-probability equilibrium action"
                                )
                            teacher_joint_distributions_this_step[agent] = {
                                "actions": joint_actions,
                                "probabilities": joint_probabilities,
                            }
                            policy_candidate_indexes = [
                                candidate_index
                                for candidate_index, candidate in enumerate(candidate_diagnostics)
                                if candidate.get("actions")
                                == midgame_report.get("preferred_own_action")
                            ]
                            selected_semantic_action = candidate_diagnostics[
                                selected_candidate_index
                            ].get("actions")
                            equilibrium_probability_of_search_action = float(sum(
                                mixed_strategy[candidate_index]
                                for candidate_index, candidate in enumerate(candidate_diagnostics)
                                if candidate.get("actions") == selected_semantic_action
                            ))
                            search_probability = 1.0 - midgame_search_policy_mixture
                            parent_probability = midgame_search_policy_mixture
                            use_parent_policy = (
                                parent_probability > 0.0
                                and rng.random() < parent_probability
                            )
                            parent_policy_action_probability = math.exp(float(log_prob))
                            if use_parent_policy:
                                native_action = policy_action_before_search
                                equilibrium_probability_of_executed_action = float(sum(
                                    mixed_strategy[candidate_index]
                                    for candidate_index in policy_candidate_indexes
                                ))
                                behavior_probability = (
                                    parent_probability * parent_policy_action_probability
                                    + search_probability
                                    * equilibrium_probability_of_executed_action
                                )
                                midgame_search_policy_branch_decisions += 1
                            elif parent_probability == 0.0:
                                native_action = searched_action
                                parent_policy_action_probability = 0.0
                                equilibrium_probability_of_executed_action = (
                                    equilibrium_probability_of_search_action
                                )
                                behavior_probability = (
                                    search_probability
                                    * equilibrium_probability_of_executed_action
                                )
                            else:
                                native_action = searched_action
                                parent_policy_action_probability = math.exp(
                                    policy.log_prob_native_action(
                                        observations[agent], masks[agent], searched_action
                                    )
                                )
                                equilibrium_probability_of_executed_action = (
                                    equilibrium_probability_of_search_action
                                )
                                behavior_probability = (
                                    search_probability
                                    * equilibrium_probability_of_executed_action
                                    + parent_probability
                                    * parent_policy_action_probability
                                )
                            if (
                                not math.isfinite(behavior_probability)
                                or behavior_probability <= 0.0
                                or behavior_probability > 1.0 + 1e-6
                            ):
                                raise ValueError("mixed search behavior probability is invalid")
                            forced_targetless_search_positions = list(
                                midgame_report.get("forced_targetless_positions", [])
                            )
                            teacher_actions_this_step.add(agent)
                            search_action_behavior_log_probs_this_step[agent] = math.log(
                                behavior_probability
                            )
                            midgame_search_teacher_decisions += 1
                            midgame_search_action_changes += int(
                                search_action_changed_policy
                            )
                            summary_keys = (
                                "selected_action", "matrix_game_method", "matrix_game_shape",
                                "matrix_game_value", "matrix_game_exploitability",
                                "mixed_row_strategy", "mixed_column_strategy", "frontier_value",
                                "frontier_probability_calibration",
                                "world_count", "chance_samples", "simulated_turn_advances",
                                "minimum_chance_samples", "maximum_chance_samples",
                                "adaptive_stop_reason",
                                "elapsed_ms", "transitions_per_second", "value_inference_ms",
                                "search_subprocess_wall_ms", "search_adapter_wall_ms",
                                "selection_hypothesis_count", "hidden_hypothesis_setup_ms",
                                "forced_targetless_positions",
                            )
                            midgame_search_reports.append({
                                "turn": int(match.state.get("turn", 0)),
                                "joint_step": int(joint_steps),
                                "search_request_state": match.state.get("request_state"),
                                "opponent_move_request_at_search": bool(
                                    requests[1 - index] is not None
                                    and not requests[1 - index].get("wait")
                                    and not requests[1 - index].get("forceSwitch")
                                    and requests[1 - index].get("active") is not None
                                ),
                                "policy_action_indices": policy_action_before_search.tolist(),
                                "policy_command": policy_command,
                                "policy_command_error": policy_command_error,
                                "policy_semantic_actions": midgame_report.get("preferred_own_action"),
                                "policy_candidate_indexes": policy_candidate_indexes,
                                "candidate_diagnostics": candidate_diagnostics,
                                "robust_scores": midgame_report.get("robust_scores"),
                                "selected_native_action_indices": search_action.tolist(),
                                "executed_native_action_indices": native_action.tolist(),
                                "changed_policy_action": search_action_changed_policy,
                                **({
                                    "response_candidate_action_indices_by_world": [
                                        {
                                            "world_index": world.get("world_index"),
                                            "actions": [
                                                candidate.get("actions")
                                                for candidate in world.get(
                                                    "opponent_response_candidates", ()
                                                )
                                            ],
                                            "action_indices": [
                                                candidate.get("action_indices")
                                                for candidate in world.get(
                                                    "opponent_response_candidates", ()
                                                )
                                            ],
                                            "request_diagnostic": world.get(
                                                "opponent_response_request_diagnostic"
                                            ),
                                        }
                                        for world in midgame_report.get("world_reports", ())
                                    ],
                                } if search_response_coverage_diagnostic else {}),
                                "behavior_used_parent_policy": use_parent_policy,
                                "parent_policy_mixture": parent_probability,
                                "equilibrium_probability_of_executed_action": (
                                    equilibrium_probability_of_executed_action
                                ),
                                "parent_policy_probability_of_executed_action": (
                                    parent_policy_action_probability
                                ),
                                "behavior_probability_of_executed_action": (
                                    behavior_probability
                                ),
                                **{key: midgame_report[key] for key in summary_keys if key in midgame_report},
                            })
                        except Exception as error:
                            live_battle = match.battles[index]
                            midgame_search_reports.append({
                                "turn": int(match.state.get("turn", 0)),
                                "policy_action_indices": policy_action_before_search.tolist(),
                                "acting_side": index,
                                "live_active_species": [
                                    str(pokemon.species) if pokemon is not None else None
                                    for pokemon in live_battle.active_pokemon
                                ],
                                "selected_native_action_indices": None,
                                "changed_policy_action": False,
                                "error": f"{type(error).__name__}: {error}",
                                "fallback": "policy",
                            })
                    is_policy_against_max_damage = (
                        (index == 0 and opponent_strategy == "max_damage")
                        or (index == 1 and player_strategy == "max_damage")
                    )
                    is_gated_policy_opponent = (
                        gate_policy_opponent
                        and (
                            (index == 1 and trainable_side == "player1"
                             and opponent_strategy == "policy")
                            or (index == 0 and trainable_side == "player2"
                                and player_strategy == "policy")
                        )
                    )
                    if (
                        ((max_damage_score_gate and is_policy_against_max_damage)
                         or is_gated_policy_opponent)
                        and not (
                            first_turn_forced_action is not None
                            and player1_decision_index == forced_action_step
                        )
                        and not forced_targetless_search_positions
                    ):
                        battle = match.battles[index]
                        if max_damage_gate_on_policy:
                            # Keep the sampled action/log-probability pair for
                            # PPO. Showdown still receives the post-gate action;
                            # the gate is part of the training environment.
                            policy_proposed_actions_this_step[agent] = [
                                int(value) for value in native_action
                            ]
                        candidate_action = choose_max_damage_action(
                            masks[agent], battle,
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        policy_damage = sum(
                            score_max_damage_action(int(value), battle, slot)
                            for slot, value in enumerate(native_action)
                        )
                        reference_damage = sum(
                            score_max_damage_action(int(value), battle, slot)
                            for slot, value in enumerate(candidate_action)
                        )
                        if reference_damage > policy_damage + 1e-6:
                            native_action = candidate_action
                            forced_targetless_search_positions = []
                            max_damage_score_gate_overrides += 1
                            if is_gated_policy_opponent:
                                policy_opponent_score_gate_overrides += 1
                            if is_policy_against_max_damage and not max_damage_gate_on_policy:
                                max_damage_gate_teacher_actions.add(agent)
                    trainable_index = 0 if trainable_side == "player1" else 1
                    teacher_opponent_is_max_damage = (
                        (trainable_index == 0 and opponent_strategy == "max_damage")
                        or (trainable_index == 1 and player_strategy == "max_damage")
                    )
                    if (
                        index == trainable_index
                        and counterplay_teacher
                        and teacher_opponent_is_max_damage
                    ):
                        # This is a closed-sheet teacher: it sees the learner's
                        # own Battle view plus facts reconstructed from the
                        # public protocol log. Never pass the other player's
                        # Battle object or its hidden move set here.
                        native_action = choose_public_counterplay_action(
                            masks[agent],
                            match.battles[index],
                            match.public_history,
                            incoming_weight=counterplay_incoming_weight,
                            ko_bonus=counterplay_ko_bonus,
                            setup_bonus=counterplay_setup_bonus,
                            redirect_actions=counterplay_redirection,
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        teacher_actions_this_step.add(agent)
                        counterplay_teacher_actions += 1
                    if (
                        index == (0 if trainable_side == "player1" else 1)
                        and counterfactual_action_selector is not None
                        and (
                            player1_decision_index if index == 0
                            else player2_decision_index
                        ) == counterfactual_selector_decision
                    ):
                        forced_action_fallback = np.asarray(
                            native_action, dtype=np.int64
                        ).tolist()
                        selected_action, counterfactual_selector_report = (
                            counterfactual_action_selector(
                                np.asarray(observations[agent], dtype=np.float32),
                                np.asarray(masks[agent], dtype=np.int8),
                                np.asarray(native_action, dtype=np.int64),
                                match.battles[index],
                                seed + episode_index * 1009 + index,
                                policy,
                            )
                        )
                        native_action = np.asarray(
                            selected_action, dtype=np.int64
                        ).tolist()
                        if counterfactual_selector_report.get("selected_source"):
                            first_turn_action_source = str(
                                counterfactual_selector_report["selected_source"]
                            )
                    if (
                        index == 1
                        and policy is not None
                        and opponent_max_damage_action_steps is not None
                        and player2_decision_index in opponent_max_damage_action_steps
                    ):
                        opponent_policy_action = np.asarray(
                            native_action, dtype=np.int64
                        ).copy()
                        native_action = choose_max_damage_action(
                            masks[agent], match.battles[index],
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        if not np.array_equal(native_action, opponent_policy_action):
                            opponent_max_damage_override_actions += 2
                        teacher_actions_this_step.add(agent)
                    actions[agent] = native_action
                    outputs[agent] = (native_action, log_prob, value)
                    if policy.last_native_action_features is not None:
                        action_features_by_agent[agent] = (
                            policy.last_native_action_features.tolist()
                        )
                    # The policy samples the second slot conditionally to
                    # prevent duplicate switches and invalid pass/pass. Store
                    # that same conditional mask so PPO can reproduce the
                    # recorded joint-action log probability.
                    half = recorded_masks[agent].size // 2
                    second_slot = recorded_masks[agent][half:]
                    action_for_conditional_mask = policy_proposed_actions_this_step.get(
                        agent, native_action
                    )
                    if action_for_conditional_mask[0] == 0:
                        second_slot[0] = 0
                    elif 1 <= action_for_conditional_mask[0] <= 6:
                        second_slot[int(action_for_conditional_mask[0])] = 0
                    if not second_slot.any():
                        recorded_masks[agent][half:] = masks[agent][half:]
                strategies_by_agent = (player_strategy, opponent_strategy)
                for index, agent in enumerate(agent_names):
                    request = requests[index]
                    strategy = strategies_by_agent[index]
                    if (
                        strategy in {"policy", "random"}
                        or agent in policies
                        or request is None
                        or request.get("wait")
                    ):
                        continue
                    if (
                        strategy == "public_counterplay"
                        and index == (0 if trainable_side == "player1" else 1)
                        and opening_search != "none"
                        and joint_steps == 0
                    ):
                        try:
                            actions[agent], search_report = _opening_search(
                                match,
                                player_index=index,
                                own_team=team1 if index == 0 else team2,
                                own_preview_action=preview_actions[index],
                                public_opponent_species=opponent_species_by_agent[agent],
                                seed=seed + episode_index * 1009 + 1,
                                format_id=format_id,
                                showdown_path=showdown_path,
                                mode=opening_search,
                                hidden_worlds=search_hidden_worlds,
                                max_candidates=search_max_candidates,
                                max_response_candidates=(
                                    search_max_candidates
                                    if search_max_response_candidates is None
                                    else search_max_response_candidates
                                ),
                                chance_samples=search_chance_samples,
                                candidate_damage_samples=search_candidate_damage_samples,
                                action_selection_method=search_action_selection,
                                workers=search_workers,
                                return_response_candidates=search_response_coverage_diagnostic,
                                evaluator=state_value_evaluator,
                                search_pool=search_pool,
                            )
                            continue
                        except Exception as error:
                            search_failure_reason = f"{type(error).__name__}: {error}"
                            search_report = {"error": search_failure_reason, "fallback": "public_counterplay"}
                    baseline_started = time.perf_counter()
                    own_battle = match.battles[index]
                    other_index = 1 - index
                    try:
                        if strategy == "max_damage":
                            actions[agent] = choose_max_damage_action(
                                masks[agent], own_battle,
                                order_validator=ShowdownDoublesEnv.action_to_order,
                            )
                        elif strategy == "counterplay":
                            actions[agent] = choose_counterplay_action(
                                masks[agent], own_battle, match.battles[other_index],
                                actions[agent_names[other_index]],
                            )
                        elif (
                            strategy == "public_counterplay"
                            and not (
                                index == 0
                                and first_turn_forced_action is not None
                                and player1_decision_index == forced_action_step
                            )
                        ):
                            actions[agent] = choose_public_counterplay_action(
                                masks[agent], own_battle, match.public_history,
                                incoming_weight=counterplay_incoming_weight,
                                ko_bonus=counterplay_ko_bonus,
                                setup_bonus=counterplay_setup_bonus,
                                redirect_actions=counterplay_redirection,
                                order_validator=ShowdownDoublesEnv.action_to_order,
                            )
                    except ValueError as exc:
                        baseline_fallbacks += 1
                        baseline_fallback_reasons.append(str(exc))
                        try:
                            actions[agent] = choose_max_damage_action(
                                masks[agent], own_battle,
                                order_validator=ShowdownDoublesEnv.action_to_order,
                            )
                        except ValueError as backup_exc:
                            actions[agent] = choose_masked_actions(
                                {agent: payloads[agent]}, rng
                            )[agent]
                            baseline_fallbacks += 1
                            baseline_fallback_reasons.append(str(backup_exc))
                    baseline_action_seconds += time.perf_counter() - baseline_started
                    if (
                        strategy == "public_counterplay"
                        and index == (0 if trainable_side == "player1" else 1)
                        and midgame_search == "learned"
                        and joint_steps > 0
                        and match.state.get("request_state") == "move"
                        and (
                            midgame_search_max_decisions == 0
                            or midgame_search_attempts < midgame_search_max_decisions
                        )
                    ):
                        if state_value_evaluator is None or search_pool is None:
                            raise RuntimeError(
                                "learned public-counterplay midgame search needs its evaluator and search pool"
                            )
                        midgame_search_attempts += 1
                        parent_action = np.asarray(actions[agent], dtype=np.int64).copy()
                        try:
                            searched_action, midgame_report = _midgame_search(
                                match,
                                acting_side=index,
                                preferred_action_indices=parent_action,
                                own_team=team1 if index == 0 else team2,
                                own_preview_action=preview_actions[index],
                                public_opponent_species=opponent_species_by_agent[agent],
                                seed=seed + episode_index * 1009 + joint_steps * 17 + 7,
                                format_id=format_id,
                                showdown_path=showdown_path,
                                hidden_worlds=midgame_search_worlds,
                                max_candidates=search_max_candidates,
                                max_response_candidates=(
                                    search_max_candidates
                                    if midgame_search_max_response_candidates is None
                                    else midgame_search_max_response_candidates
                                ),
                                max_damage_response_candidates=(
                                    midgame_search_max_damage_response_candidates
                                ),
                                response_selection_strategy=(
                                    midgame_search_response_selection_strategy
                                ),
                                frontier_calibration=midgame_search_frontier_calibration,
                                chance_samples=midgame_search_chance_samples,
                                max_chance_samples=midgame_search_max_chance_samples,
                                action_selection_method=search_action_selection,
                                evaluator=state_value_evaluator,
                                search_pool=search_pool,
                                timeout_seconds=midgame_search_timeout_seconds,
                                return_response_candidates=search_response_coverage_diagnostic,
                            )
                            search_mask = np.asarray(masks[agent], dtype=np.int8)
                            search_action = np.asarray(searched_action, dtype=np.int64)
                            if search_mask.ndim != 1 or search_mask.size % 2 or search_action.shape != (2,):
                                raise ValueError("midgame search returned an invalid native action or mask")
                            slot_width = search_mask.size // 2
                            if any(
                                not 0 <= int(search_action[slot]) < slot_width
                                or search_mask[slot * slot_width + int(search_action[slot])] == 0
                                for slot in range(2)
                            ):
                                raise ValueError("midgame search returned an illegal public-counterplay action")
                            midgame_report["public_counterplay_parent_action"] = parent_action.tolist()
                            midgame_report["public_counterplay_action_changed"] = bool(
                                not np.array_equal(parent_action, search_action)
                            )
                            if midgame_report["public_counterplay_action_changed"]:
                                midgame_search_action_changes += 1
                            actions[agent] = search_action
                            midgame_search_reports.append(midgame_report)
                            midgame_search_teacher_decisions += 1
                        except Exception as error:
                            midgame_search_reports.append({
                                "turn": int(match.state.get("turn", 0)),
                                "error": f"{type(error).__name__}: {error}",
                                "fallback": "public_counterplay",
                            })
                if (
                    search_response_coverage_diagnostic
                    and midgame_search_reports
                ):
                    search_side = 0 if trainable_side == "player1" else 1
                    opponent_side = 1 - search_side
                    opponent_strategy_for_search = strategies_by_agent[opponent_side]
                    opponent_request = requests[opponent_side]
                    if (
                        opponent_strategy_for_search in {"max_damage", "public_counterplay"}
                        and opponent_request is not None
                        and not opponent_request.get("wait")
                    ):
                        actual_pair = np.asarray(
                            actions[agent_names[opponent_side]], dtype=np.int64
                        ).reshape(-1).tolist()
                        actual_semantics = _semantic_native_action_pair(
                            actual_pair, match.battles[opponent_side]
                        )
                        live_opponent_pokemon = (
                            opponent_request.get("side", {}).get("pokemon", [])
                        )
                        live_opponent_request_debug = {
                            "force_switch": opponent_request.get("forceSwitch"),
                            "active": [
                                {
                                    "slot": slot,
                                    "condition": (
                                        live_opponent_pokemon[slot].get("condition")
                                        if slot < len(live_opponent_pokemon) else None
                                    ),
                                    "moves": [
                                        move.get("id")
                                        for move in active_request.get("moves", ())
                                    ],
                                }
                                for slot, active_request in enumerate(
                                    opponent_request.get("active", ())
                                )
                            ],
                        }
                        for report in reversed(midgame_search_reports):
                            if (
                                report.get("turn") != int(match.state.get("turn", 0))
                                or report.get("joint_step") != int(joint_steps)
                                or report.get("search_request_state") != "move"
                                or not report.get("opponent_move_request_at_search")
                                or opponent_request.get("forceSwitch")
                                or opponent_request.get("active") is None
                                or "response_candidate_action_indices_by_world" not in report
                            ):
                                continue
                            worlds = report["response_candidate_action_indices_by_world"]
                            covered_worlds = []
                            covered_slots = [[], []]
                            for world in worlds:
                                candidates = world.get("actions", [])
                                if any(candidate == actual_semantics for candidate in candidates):
                                    covered_worlds.append(world.get("world_index"))
                                for slot in range(2):
                                    if any(
                                        isinstance(candidate, list)
                                        and len(candidate) == 2
                                        and candidate[slot] == actual_semantics[slot]
                                        for candidate in candidates
                                    ):
                                        covered_slots[slot].append(world.get("world_index"))
                            report["observed_opponent_strategy"] = opponent_strategy_for_search
                            report["observed_opponent_action_indices"] = actual_pair
                            report["observed_opponent_action_semantics"] = actual_semantics
                            report["live_opponent_request_diagnostic"] = live_opponent_request_debug
                            coverage = {
                                "covered_world_count": len(covered_worlds),
                                "world_count": len(worlds),
                                "covered_fraction": len(covered_worlds) / max(1, len(worlds)),
                                "covered_world_indices": covered_worlds,
                                "per_slot_covered_world_counts": [len(rows) for rows in covered_slots],
                                "candidate_counts_by_world": [
                                    len(world.get("actions", ())) for world in worlds
                                ],
                            }
                            report["opponent_response_joint_candidate_coverage"] = coverage
                            if opponent_strategy_for_search == "max_damage":
                                report["max_damage_actual_action_indices"] = actual_pair
                                report["max_damage_actual_action_semantics"] = actual_semantics
                                report["max_damage_joint_candidate_coverage"] = coverage
                            break
                if joint_steps == 0 and first_turn_forced_joint_actions is not None:
                    if len(first_turn_forced_joint_actions) != 2:
                        raise ValueError("forced joint action needs one action pair per player")
                    for side, forced_pair in enumerate(first_turn_forced_joint_actions):
                        forced = np.asarray(forced_pair, dtype=np.int64)
                        legal = np.asarray(masks[agent_names[side]])
                        if forced.shape != (2,) or legal.ndim != 1 or legal.size % 2:
                            raise ValueError("each forced joint-action component must be a pair of native actions")
                        width = legal.size // 2
                        if any(
                            action < 0 or action >= width or legal[slot * width + action] != 1
                            for slot, action in enumerate(forced.tolist())
                        ):
                            raise ValueError(f"forced joint action for P{side + 1} is illegal under its current mask")
                        actions[agent_names[side]] = forced
                    first_turn_action_source = "counterfactual_joint"

                if (
                    forced_opponent_action is not None
                    and player1_decision_index == forced_action_step
                    and requests[1] is not None
                    and not requests[1].get("wait")
                ):
                    forced = np.asarray(forced_opponent_action, dtype=np.int64)
                    legal = np.asarray(masks[agent_names[1]])
                    if forced.shape != (2,) or legal.ndim != 1 or legal.size % 2:
                        raise ValueError("forced opponent action must be a pair of native action indices")
                    slot_width = legal.size // 2
                    if any(
                        action < 0 or action >= slot_width or legal[slot * slot_width + action] != 1
                        for slot, action in enumerate(forced.tolist())
                    ):
                        raise ValueError("forced opponent action is illegal under its current mask")
                    forced_opponent_action_fallback = np.asarray(
                        actions[agent_names[1]], dtype=np.int64
                    ).tolist()
                    actions[agent_names[1]] = forced
                    first_turn_action_source = "counterfactual_joint"

                try:
                    choices = match.choices_from_native_actions(
                        [actions[agent_names[0]], actions[agent_names[1]]],
                        forced_targetless_positions={
                            (0 if trainable_side == "player1" else 1): forced_targetless_search_positions
                        } if forced_targetless_search_positions else None,
                    )
                    if counterfactual_selector_report is not None:
                        counterfactual_selector_report["selected_action_accepted"] = True
                except ValueError as rejection_error:
                    if forced_action_fallback is None:
                        raise
                    # Invalid combinations of independently legal slot actions
                    # must not leave the persistent Showdown worker waiting for
                    # a choice. Replay the known-valid policy pair and mark this
                    # candidate as rejected so it is never used as a label.
                    actions[agent_names[0]] = np.asarray(
                        forced_action_fallback, dtype=np.int64
                    )
                    if forced_opponent_action_fallback is not None:
                        actions[agent_names[1]] = np.asarray(
                            forced_opponent_action_fallback, dtype=np.int64
                        )
                    forced_action_accepted = False
                    forced_action_rejection_error = str(rejection_error)
                    if counterfactual_selector_report is not None:
                        counterfactual_selector_report["selected_action_accepted"] = False
                    first_turn_action_source = "counterfactual_rejected"
                    teacher_actions_this_step.discard(agent_names[0])
                    choices = match.choices_from_native_actions(
                        [actions[agent_names[0]], actions[agent_names[1]]]
                    )
                if (
                    joint_steps == 0
                    and requests[0] is not None
                    and not requests[0].get("wait")
                ):
                    first_turn_action_data = {
                        "action": np.asarray(
                            actions[agent_names[0]], dtype=np.int64
                        ).tolist(),
                        "observation": np.asarray(
                            observations[agent_names[0]], dtype=np.float32
                        ).tolist(),
                        "legal_action_mask": np.asarray(
                            masks[agent_names[0]], dtype=np.int8
                        ).tolist(),
                        "recorded_action_mask": np.asarray(
                            recorded_masks[agent_names[0]], dtype=np.int8
                        ).tolist(),
                        "native_action_features": (
                            np.asarray(
                                policies[agent_names[0]].last_native_action_features,
                                dtype=np.float32,
                            ).tolist()
                            if policies.get(agent_names[0]) is not None
                            and policies[agent_names[0]].last_native_action_features is not None
                            else None
                        ),
                        "public_context_features": (
                            np.asarray(
                                policies[agent_names[0]].public_context_features,
                                dtype=np.float32,
                            ).tolist()
                            if policies.get(agent_names[0]) is not None
                            and policies[agent_names[0]].public_context_features is not None
                            else None
                        ),
                        "source": first_turn_action_source,
                    }
                if joint_steps == 0 and all(
                    request is not None and not request.get("wait") for request in requests
                ):
                    first_turn_joint_action_data = {}
                    for side, agent in enumerate(agent_names):
                        own_team = team1 if side == 0 else team2
                        opposing_team = team2 if side == 0 else team1
                        policy = policies.get(agent)
                        context = (
                            policy.public_context_features
                            if policy is not None else encode_public_mechanics(
                                public_mechanics[agent], observations[agent],
                                own_team, opposing_team, format_id,
                            )
                        )
                        max_damage_pair = choose_max_damage_action(
                            masks[agent], match.battles[side],
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        first_turn_joint_action_data[agent] = {
                            "action": np.asarray(actions[agent], dtype=np.int64).tolist(),
                            "observation": np.asarray(observations[agent], dtype=np.float32).tolist(),
                            "legal_action_mask": np.asarray(masks[agent], dtype=np.int8).tolist(),
                            "max_damage_action": np.asarray(max_damage_pair, dtype=np.int64).tolist(),
                            "native_action_features": (
                                np.asarray(policy.last_native_action_features, dtype=np.float32).tolist()
                                if policy is not None and policy.last_native_action_features is not None
                                else None
                            ),
                            "public_context_features": np.asarray(context, dtype=np.float32).tolist(),
                        }
                    search_side = 0 if trainable_side == "player1" else 1
                    response_worlds = (
                        search_report.get("response_candidates_by_world", ())
                        if search_response_coverage_diagnostic and search_report
                        else ()
                    )
                    if response_worlds:
                        opponent_side = 1 - search_side
                        opponent_agent = agent_names[opponent_side]
                        actual_opponent_actions = np.asarray(
                            actions[opponent_agent], dtype=np.int64
                        ).reshape(-1).tolist()
                        actual_opponent_semantics = _semantic_native_action_pair(
                            actual_opponent_actions, match.battles[opponent_side]
                        )
                        covered_worlds = [
                            world.get("world_index")
                            for world in response_worlds
                            if actual_opponent_semantics in world.get("actions", ())
                        ]
                        coverage = {
                            "observed_opponent_strategy": (
                                opponent_strategy if trainable_side == "player1"
                                else player_strategy
                            ),
                            "actual_opponent_action_semantics": actual_opponent_semantics,
                            "covered_world_count": len(covered_worlds),
                            "world_count": len(response_worlds),
                            "covered_fraction": len(covered_worlds) / len(response_worlds),
                            "candidate_counts_by_world": [
                                len(world.get("actions", ())) for world in response_worlds
                            ],
                        }
                        search_report["opponent_response_joint_candidate_coverage"] = coverage
                        first_turn_joint_action_data[
                            agent_names[search_side]
                        ]["opening_search_response_coverage"] = coverage
                if (
                    capture_action_step == player1_decision_index
                    and requests[0] is not None
                    and not requests[0].get("wait")
                ):
                    captured_battle_snapshot = (
                        match.snapshot() if capture_battle_snapshot else None
                    )
                    captured_parser_snapshot = (
                        match.parser_snapshot() if capture_battle_snapshot else None
                    )
                    captured_policy_log_prob = None
                    captured_policy_log_prob_error = None
                    legal_joint_action_log_probs = None
                    if capture_policy_action_log_prob:
                        acting_policy = policies.get(agent_names[0])
                        if acting_policy is not None:
                            try:
                                captured_policy_log_prob = float(
                                    acting_policy.log_prob_native_action(
                                        observations[agent_names[0]],
                                        masks[agent_names[0]],
                                        actions[agent_names[0]],
                                    )
                                )
                            except (ValueError, RuntimeError) as error:
                                captured_policy_log_prob_error = (
                                    f"{type(error).__name__}: {error}"
                                )
                    if capture_legal_joint_action_log_probs:
                        acting_policy = policies.get(agent_names[0])
                        if acting_policy is not None:
                            legal_joint_actions = [
                                [int(a), int(b)]
                                for a in np.flatnonzero(masks[agent_names[0]][:len(masks[agent_names[0]]) // 2])
                                for b in np.flatnonzero(masks[agent_names[0]][len(masks[agent_names[0]]) // 2:])
                            ]
                            try:
                                log_probs = acting_policy.log_prob_native_actions(
                                    observations[agent_names[0]],
                                    masks[agent_names[0]],
                                    legal_joint_actions,
                                )
                                legal_joint_action_log_probs = [
                                    {"action": action, "log_probability": log_prob}
                                    for action, log_prob in zip(legal_joint_actions, log_probs)
                                ]
                            except (ValueError, RuntimeError) as error:
                                captured_policy_log_prob_error = (
                                    f"{type(error).__name__}: {error}"
                                )
                    counterfactual_action_data = {
                        "action": np.asarray(
                            actions[agent_names[0]], dtype=np.int64
                        ).tolist(),
                        "observation": np.asarray(
                            observations[agent_names[0]], dtype=np.float32
                        ).tolist(),
                        "legal_action_mask": np.asarray(
                            masks[agent_names[0]], dtype=np.int8
                        ).tolist(),
                        "recorded_action_mask": np.asarray(
                            recorded_masks[agent_names[0]], dtype=np.int8
                        ).tolist(),
                        "legal_actions_by_slot": _legal_native_action_descriptors(
                            masks[agent_names[0]], match.battles[0]
                        ),
                        "native_action_features": (
                            np.asarray(policies[agent_names[0]].last_native_action_features, dtype=np.float32).tolist()
                            if policies.get(agent_names[0]) is not None
                            and policies[agent_names[0]].last_native_action_features is not None else None
                        ),
                        "public_context_features": (
                            np.asarray(policies[agent_names[0]].public_context_features, dtype=np.float32).tolist()
                            if policies.get(agent_names[0]) is not None
                            and policies[agent_names[0]].public_context_features is not None else None
                        ),
                        "source": first_turn_action_source,
                        "joint_step": joint_steps,
                        "player1_decision_index": player1_decision_index,
                        "player2_decision_index": player2_decision_index,
                        "decision_counts_at_snapshot": [
                            int(player1_decision_index),
                            int(player2_decision_index
                                if player2_decision_index is not None
                                else player2_decisions),
                        ],
                        "battle_snapshot": captured_battle_snapshot,
                        "parser_snapshot": captured_parser_snapshot,
                        "forced_action_accepted": (
                            forced_action_accepted
                            if first_turn_forced_action is not None
                            and player1_decision_index == forced_action_step
                            else None
                        ),
                        "forced_action_rejection_error": (
                            forced_action_rejection_error
                            if first_turn_forced_action is not None
                            and player1_decision_index == forced_action_step
                            else None
                        ),
                        "policy_log_probability_of_selected_action": (
                            captured_policy_log_prob
                        ),
                        "policy_log_probability_error": captured_policy_log_prob_error,
                        "legal_joint_action_log_probabilities": legal_joint_action_log_probs,
                    }
                    counterfactual_action_data_by_agent[agent_names[0]] = (
                        counterfactual_action_data
                    )
                if (
                    capture_action_step == player2_decision_index
                    and requests[1] is not None
                    and not requests[1].get("wait")
                ):
                    counterfactual_action_data_by_agent[agent_names[1]] = {
                        "action": np.asarray(
                            actions[agent_names[1]], dtype=np.int64
                        ).tolist(),
                        "observation": np.asarray(
                            observations[agent_names[1]], dtype=np.float32
                        ).tolist(),
                        "legal_action_mask": np.asarray(
                            masks[agent_names[1]], dtype=np.int8
                        ).tolist(),
                        "recorded_action_mask": np.asarray(
                            recorded_masks[agent_names[1]], dtype=np.int8
                        ).tolist(),
                        "legal_actions_by_slot": _legal_native_action_descriptors(
                            masks[agent_names[1]], match.battles[1]
                        ),
                        "native_action_features": (
                            np.asarray(policies[agent_names[1]].last_native_action_features, dtype=np.float32).tolist()
                            if policies.get(agent_names[1]) is not None
                            and policies[agent_names[1]].last_native_action_features is not None else None
                        ),
                        "public_context_features": (
                            np.asarray(policies[agent_names[1]].public_context_features, dtype=np.float32).tolist()
                            if policies.get(agent_names[1]) is not None
                            and policies[agent_names[1]].public_context_features is not None else None
                        ),
                        "source": "policy",
                        "joint_step": joint_steps,
                        "player2_decision_index": player2_decision_index,
                    }
                if (
                    capture_action_step == player1_decision_index
                    and all(request is not None and not request.get("wait") for request in requests)
                ):
                    counterfactual_joint_action_data = {}
                    for side, agent in enumerate(agent_names):
                        max_damage_pair = choose_max_damage_action(
                            masks[agent], match.battles[side],
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        )
                        counterfactual_joint_action_data[agent] = {
                            "action": np.asarray(actions[agent], dtype=np.int64).tolist(),
                            "observation": np.asarray(observations[agent], dtype=np.float32).tolist(),
                            "legal_action_mask": np.asarray(masks[agent], dtype=np.int8).tolist(),
                            "max_damage_action": np.asarray(max_damage_pair, dtype=np.int64).tolist(),
                        }
                if os.environ.get("POKE_HEADLESS_DEBUG"):
                    debug_payload = {
                        "episode": game,
                        "turn": int(match.state.get("turn", 0)),
                        "actions": [np.asarray(actions[name]).tolist() for name in agent_names],
                        "choices": choices,
                        "request_waits": [
                            request is None or bool(request.get("wait"))
                            for request in requests
                        ],
                        "p1_first_turn_override": bool(
                            first_turn_max_damage and joint_steps == 0
                        ),
                        "p1_max_damage_score_gate": bool(max_damage_score_gate),
                        "trapped": [list(getattr(battle, "trapped", ())) for battle in match.battles],
                        "force_switch": [list(getattr(battle, "force_switch", ())) for battle in match.battles],
                    }
                    if player_strategy == "policy" and requests[0] is not None and not requests[0].get("wait"):
                        try:
                            debug_payload["p1_policy_order"] = str(
                                ShowdownDoublesEnv.action_to_order(
                                    actions[agent_names[0]], match.battles[0], strict=True
                                )
                            )
                            reference_action = choose_max_damage_action(
                                masks[agent_names[0]],
                                match.battles[0],
                                order_validator=ShowdownDoublesEnv.action_to_order,
                            )
                            debug_payload["p1_max_damage_reference_action"] = reference_action.tolist()
                            debug_payload["p1_policy_damage_score"] = float(sum(
                                score_max_damage_action(int(value), match.battles[0], slot)
                                for slot, value in enumerate(actions[agent_names[0]])
                            ))
                            debug_payload["p1_max_damage_reference_score"] = float(sum(
                                score_max_damage_action(int(value), match.battles[0], slot)
                                for slot, value in enumerate(reference_action)
                            ))
                            debug_payload["p1_max_damage_reference_order"] = str(
                                ShowdownDoublesEnv.action_to_order(
                                    reference_action, match.battles[0], strict=True
                                )
                            )
                        except ValueError as error:
                            debug_payload["p1_max_damage_reference_error"] = str(error)
                    print(
                        "headless_action_debug",
                        json.dumps(debug_payload),
                        flush=True,
                    )
                action_adjustments = match.action_adjustments + baseline_fallbacks
            action_seconds += time.perf_counter() - action_started
            step_started = time.perf_counter()
            if (
                reseed_random_at_decision is not None
                and player1_decision_index == reseed_random_at_decision[0]
            ):
                match.reseed_random(reseed_random_at_decision[1])
            state = match.advance(*choices)
            engine_roundtrip_seconds += time.perf_counter() - step_started

            winner = state.get("winner") if state.get("ended") else None
            rewards = {agent: 0.0 for agent in agent_names}
            reward_components = {agent: {"terminal_outcome": 0.0} for agent in agent_names}
            terminated = {agent: bool(state.get("ended")) for agent in agent_names}
            if winner:
                for agent in agent_names:
                    outcome = 1.0 if winner == agent else -1.0
                    rewards[agent] = outcome
                    reward_components[agent]["terminal_outcome"] = outcome
            transition: dict[str, Any] = {
                "turn": int(state.get("turn", 0)),
                "observations": {agent: observations[agent].tolist() for agent in agent_names},
                "action_masks": {agent: recorded_masks[agent].tolist() for agent in agent_names},
                # The policy's action-feature encoder sees the original legal
                # mask. `action_masks` may be narrowed below for PPO replay to
                # preserve the conditional second action after slot one.
                "raw_action_masks": {
                    agent: np.asarray(masks[agent], dtype=np.int8).tolist()
                    for agent in agent_names
                },
                "actions": {agent: np.asarray(actions[agent], dtype=np.int64).tolist() for agent in agent_names},
                "public_mechanics": public_mechanics,
                "opponent_species": opponent_species_by_agent,
                "rewards": rewards,
                "reward_components": reward_components,
                "terminated": terminated,
                "truncated": {agent: False for agent in agent_names},
            }
            if policy_proposed_actions_this_step:
                transition["policy_proposed_actions"] = (
                    policy_proposed_actions_this_step
                )
            if outputs:
                transition["log_probs"] = {agent: output[1] for agent, output in outputs.items()}
                transition["values"] = {agent: output[2] for agent, output in outputs.items()}
            if max_damage_gate_teacher_actions:
                # Legacy gate training treats true overrides as off-policy
                # teachers. On-policy gate training records the proposal
                # separately and keeps it in PPO with its sampled log-prob.
                transition["teacher_actions"] = {
                    agent: np.asarray(actions[agent], dtype=np.int64).tolist()
                    for agent in max_damage_gate_teacher_actions
                    if agent in outputs
                }
            if teacher_actions_this_step:
                transition.setdefault("teacher_actions", {}).update({
                    agent: np.asarray(actions[agent], dtype=np.int64).tolist()
                    for agent in teacher_actions_this_step
                })
            if teacher_joint_distributions_this_step:
                transition["teacher_joint_action_distributions"] = (
                    teacher_joint_distributions_this_step
                )
            if search_action_behavior_log_probs_this_step:
                transition["search_action_behavior_log_probs"] = (
                    search_action_behavior_log_probs_this_step
                )
            if action_features_by_agent:
                transition["native_action_features"] = action_features_by_agent
            trajectory.append(transition)
            joint_steps += 1

    match_wall_seconds = max(time.perf_counter() - wall_started, 1e-9)
    winner = state.get("winner")
    rewards = {agent: (1.0 if winner == agent else -1.0) for agent in agent_names}
    runtime_metrics = {
        "episode_wall_seconds": match_wall_seconds,
        "preview_and_first_step_seconds": preview_seconds,
        "observation_encoding_seconds": observation_seconds,
        "action_selection_seconds": action_seconds,
        "policy_inference_seconds": policy_inference_seconds,
        "baseline_action_seconds": baseline_action_seconds,
        "action_proposal_seconds": action_proposal_seconds,
        "headless_engine_roundtrip_seconds": engine_roundtrip_seconds,
        "joint_turns_per_second": joint_steps / match_wall_seconds,
        "individual_decisions_per_second": 2 * joint_steps / match_wall_seconds,
    }
    match_record = {
        "episode": record_episode,
        "format": format_id,
        "opponent": (
            opponent_strategy if trainable_side == "player1" else player_strategy
        ),
        "player_strategy": player_strategy,
        "trainable_agent": agent_names[0] if trainable_side == "player1" else agent_names[1],
        "policy_deterministic": deterministic_policy,
        "native_action_feature_encoder_version": NATIVE_ACTION_FEATURE_ENCODER_VERSION,
        "native_action_feature_dim": int(getattr(model, "native_action_feature_dim", 0) or 0),
        "native_action_feature_variant": getattr(model, "native_action_feature_variant", None),
        "teams": {"player1": team1, "player2": team2},
        "agent_to_team": agent_to_team,
        "teams_differ": team1 != team2,
        "preview_actions": {"player1": preview_actions[0], "player2": preview_actions[1]},
        "preview_log_probs": preview_log_probs,
        "team_selection_applied": team_selection_applied,
        "team_selection_side": team_selection_side,
        "opponent_team_selection_applied": opponent_team_selection_applied,
        "opponent_team_selection_side": opponent_team_selection_side,
        "first_turn_action_data": (
            {agent_names[0]: first_turn_action_data}
            if first_turn_action_data is not None else {}
        ),
        "first_turn_joint_action_data": first_turn_joint_action_data or {},
        "steps": joint_steps,
        "action_adjustments": action_adjustments,
        "baseline_fallbacks": baseline_fallbacks,
        "max_damage_score_gate_overrides": max_damage_score_gate_overrides,
        "policy_opponent_score_gate_overrides": policy_opponent_score_gate_overrides,
        "gate_policy_opponent": bool(gate_policy_opponent),
        "first_turn_max_damage_override_actions": first_turn_max_damage_override_actions,
        "opponent_max_damage_override_actions": opponent_max_damage_override_actions,
        "max_damage_action_steps": (
            sorted(max_damage_action_steps)
            if max_damage_action_steps is not None else None
        ),
        "opponent_max_damage_action_steps": (
            sorted(opponent_max_damage_action_steps)
            if opponent_max_damage_action_steps is not None else None
        ),
        "max_damage_action_fallbacks": max_damage_action_fallbacks,
        "max_damage_gate_on_policy": bool(max_damage_gate_on_policy),
        "counterplay_teacher": bool(counterplay_teacher),
        "counterplay_teacher_actions": counterplay_teacher_actions,
        "baseline_fallback_reasons": baseline_fallback_reasons,
        "opening_search": opening_search,
        "opening_search_report": search_report,
        "midgame_search": midgame_search,
        "midgame_search_policy_mixture": midgame_search_policy_mixture,
        "midgame_search_reports": midgame_search_reports,
        "midgame_search_attempts": midgame_search_attempts,
        "midgame_search_action_changes": midgame_search_action_changes,
        "midgame_search_policy_branch_decisions": midgame_search_policy_branch_decisions,
        "midgame_search_decisions": sum("error" not in row for row in midgame_search_reports),
        "midgame_search_failures": sum("error" in row for row in midgame_search_reports),
        "midgame_search_teacher_decisions": midgame_search_teacher_decisions,
        "midgame_search_has_behavior_log_prob": midgame_search_teacher_decisions > 0,
        "search_failure_reason": search_failure_reason,
        "rewards": rewards,
        "terminated": {agent: True for agent in agent_names},
        "truncated": {agent: False for agent in agent_names},
        "runtime_metrics": runtime_metrics,
        "trajectory": trajectory,
    }
    # A forced wait is not a decision by that player. Keep PPO trajectories
    # to policy decisions only, and attach the terminal result to each
    # policy's last actual decision so GAE sees a proper terminal transition.
    trajectory_by_agent: dict[str, list[dict[str, Any]]] = {}
    for agent in policies:
        player_steps = [
            deepcopy(step)
            for step in trajectory
            if agent in step.get("log_probs", {})
        ]
        if player_steps:
            for step in player_steps:
                step["terminated"] = {agent: False}
                step["truncated"] = {agent: False}
                step["rewards"] = {agent: 0.0}
                step["reward_components"] = {agent: {"terminal_outcome": 0.0}}
            player_steps[-1]["terminated"] = {agent: True}
            player_steps[-1]["rewards"] = {agent: rewards[agent]}
            player_steps[-1]["reward_components"] = {
                agent: {"terminal_outcome": rewards[agent]}
            }
        trajectory_by_agent[agent] = player_steps
    match_record["trajectory_by_agent"] = trajectory_by_agent
    persist_started = time.perf_counter()
    if persist_record:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            for view in project_episode_to_player_views(match_record):
                stream.write(json.dumps(view, sort_keys=True) + "\n")
            stream.flush()
    runtime_metrics["trajectory_persist_seconds"] = time.perf_counter() - persist_started
    runtime_metrics["pre_match_setup_seconds"] = max(wall_started - started_at, 0.0)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    runtime_metrics["end_to_end_episode_seconds"] = elapsed
    if telemetry:
        telemetry.emit(
            "episode_completed",
            episode=record_episode,
            games=1,
            steps=joint_steps,
            timed_out=False,
            winner=winner,
            player1_reward=rewards[agent_names[0]],
            player2_reward=rewards[agent_names[1]],
            max_damage_score_gate_overrides=max_damage_score_gate_overrides,
            counterplay_teacher_actions=counterplay_teacher_actions,
            midgame_search_decisions=midgame_search_teacher_decisions,
            midgame_search_policy_branch_decisions=midgame_search_policy_branch_decisions,
            midgame_search_policy_mixture=midgame_search_policy_mixture,
            midgame_search_failures=sum("error" in row for row in midgame_search_reports),
            **runtime_metrics,
        )
    return {
        "episode": record_episode,
        "winner": winner,
        "first_turn_action_data": first_turn_action_data,
        "first_turn_joint_action_data": first_turn_joint_action_data,
        "counterfactual_action_data": counterfactual_action_data,
        "counterfactual_action_data_by_agent": counterfactual_action_data_by_agent,
        "counterfactual_joint_action_data": counterfactual_joint_action_data,
        "counterfactual_selector_report": counterfactual_selector_report,
        "preview_actions": preview_actions,
        "steps": joint_steps,
        "policy_deterministic": deterministic_policy,
        "action_adjustments": action_adjustments,
        "baseline_fallbacks": baseline_fallbacks,
        "baseline_fallback_reasons": baseline_fallback_reasons,
        "max_damage_score_gate_overrides": max_damage_score_gate_overrides,
        "policy_opponent_score_gate_overrides": policy_opponent_score_gate_overrides,
        "gate_policy_opponent": bool(gate_policy_opponent),
        "first_turn_max_damage_override_actions": first_turn_max_damage_override_actions,
        "opponent_max_damage_override_actions": opponent_max_damage_override_actions,
        "max_damage_action_steps": (
            sorted(max_damage_action_steps)
            if max_damage_action_steps is not None else None
        ),
        "opponent_max_damage_action_steps": (
            sorted(opponent_max_damage_action_steps)
            if opponent_max_damage_action_steps is not None else None
        ),
        "max_damage_action_fallbacks": max_damage_action_fallbacks,
        "max_damage_gate_on_policy": bool(max_damage_gate_on_policy),
        "team_selection_applied": team_selection_applied,
        "team_selection_side": team_selection_side,
        "opponent_team_selection_applied": opponent_team_selection_applied,
        "opponent_team_selection_side": opponent_team_selection_side,
        "counterplay_teacher_actions": counterplay_teacher_actions,
        "opening_search": opening_search,
        "opening_search_report": search_report,
        "midgame_search": midgame_search,
        "midgame_search_policy_mixture": midgame_search_policy_mixture,
        "midgame_search_policy_branch_decisions": midgame_search_policy_branch_decisions,
        "midgame_search_reports": midgame_search_reports,
        "midgame_search_attempts": midgame_search_attempts,
        "midgame_search_action_changes": midgame_search_action_changes,
        "midgame_search_decisions": sum("error" not in row for row in midgame_search_reports),
        "midgame_search_failures": sum("error" in row for row in midgame_search_reports),
        "midgame_search_teacher_decisions": midgame_search_teacher_decisions,
        "search_failure_reason": search_failure_reason,
        "elapsed_seconds": elapsed,
        **runtime_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--episode-offset", type=int, default=0,
                        help="global episode index offset for reproducibly sharded rollout runs")
    parser.add_argument("--record-episode-offset", type=int, default=0,
                        help="record ID offset for uniquely identifying player-view games")
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--showdown-path", type=Path, default=Path(os.environ.get("POKEMON_SHOWDOWN_PATH", "/home/justin/p/poke_learning/pokemon-showdown")))
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument(
        "--team-selection-checkpoint", type=Path,
        help="choose the trained policy player's preview using a frozen team selector",
    )
    parser.add_argument(
        "--opponent-team-selection-checkpoint", type=Path,
        help="choose the frozen policy opponent's preview using a frozen team selector",
    )
    parser.add_argument(
        "--team-selection-temperature", type=float, default=0.0,
        help="sample preview choices from selector softmax at this temperature; 0 keeps greedy selection",
    )
    parser.add_argument("--opening-search", choices=("none", "material", "learned"), default="none",
                        help="replace the policy's first action for an isolated full-match search comparison")
    parser.add_argument(
        "--midgame-search", choices=("none", "learned"), default="none",
        help="run one-turn simultaneous-action equilibrium search from public-compatible midgame worlds",
    )
    parser.add_argument("--midgame-search-worlds", type=int, default=2)
    parser.add_argument("--midgame-search-chance-samples", type=int, default=1)
    parser.add_argument("--midgame-search-max-chance-samples", type=int, default=None)
    parser.add_argument("--midgame-search-timeout-seconds", type=float, default=3.0)
    parser.add_argument(
        "--midgame-search-max-decisions", type=int, default=0,
        help="maximum learned midgame searches per match; 0 means every eligible turn",
    )
    parser.add_argument(
        "--midgame-search-policy-mixture", type=float, default=0.0,
        help="probability of executing the parent policy action instead of the searched action",
    )
    parser.add_argument(
        "--search-response-coverage-diagnostic", action="store_true",
        help="log whether the opponent's actual response appears in sampled search response candidates; diagnostic only",
    )
    parser.add_argument("--state-value-checkpoint", type=Path,
                        help="closed-sheet value model required by learned search")
    parser.add_argument(
        "--state-value-device", choices=("cpu", "cuda"),
        help="device for the learned frontier model; defaults to --device",
    )
    parser.add_argument("--search-hidden-worlds", type=int, default=2)
    parser.add_argument("--search-max-candidates", type=int, default=4)
    parser.add_argument(
        "--search-max-response-candidates", type=int,
        help="opponent response shortlist size; defaults to --search-max-candidates",
    )
    parser.add_argument(
        "--midgame-search-max-damage-response-candidates", type=int, default=1,
        help="reserve this many response rows for the highest raw-damage joint actions",
    )
    parser.add_argument(
        "--midgame-search-response-selection-strategy",
        choices=("semantic", "slot_coverage"), default="semantic",
        help="opponent response shortlist rule for midgame search",
    )
    parser.add_argument("--search-chance-samples", type=int, default=4,
                        help="simulator outcomes averaged per joint-action pair")
    parser.add_argument(
        "--search-action-selection", choices=("maximin", "equilibrium"),
        default="equilibrium",
        help="pure worst-case action or seeded regret-matched mixed strategy",
    )
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument(
        "--player-strategy",
        choices=("policy", "random", "max_damage", "counterplay", "public_counterplay"),
        default="policy",
    )
    parser.add_argument(
        "--opponent-strategy",
        choices=("policy", "random", "max_damage", "counterplay", "public_counterplay"),
        default="random",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--deterministic-policy", action="store_true",
        help="use greedy policy actions for reproducible paired evaluations",
    )
    parser.add_argument(
        "--first-turn-max-damage", action="store_true",
        help="use max-damage for P1's first joint action, then return to the policy",
    )
    parser.add_argument(
        "--max-damage-action-steps", type=int, nargs="+",
        help="P1 decision indices at which to replace the policy action with Max Damage",
    )
    parser.add_argument(
        "--opponent-max-damage-action-steps", type=int, nargs="+",
        help="P2 policy decision indices at which to replace its action with Max Damage",
    )
    parser.add_argument(
        "--max-damage-score-gate", action="store_true",
        help="replace any policy action with a strictly higher immediate-damage max-damage action",
    )
    parser.add_argument(
        "--gate-policy-opponent", action="store_true",
        help="apply the immediate-damage gate only to the frozen policy opponent",
    )
    parser.add_argument(
        "--max-damage-gate-on-policy", action="store_true",
        help="train PPO on sampled proposals while executing the gated action",
    )
    parser.add_argument(
        "--counterplay-teacher", action="store_true",
        help=("use the public-only counterplay scorer for trainable-side teacher labels "
              "in matches against max-damage"),
    )
    parser.add_argument("--counterplay-incoming-weight", type=float, default=1.0)
    parser.add_argument("--counterplay-ko-bonus", type=float, default=0.7)
    parser.add_argument("--counterplay-setup-bonus", type=float, default=0.0)
    parser.add_argument("--counterplay-redirection", action="store_true")
    parser.add_argument(
        "--swap-teams", action="store_true",
        help="swap generated teams between player sides for paired evaluation",
    )
    parser.add_argument("--trainable-side", choices=("player1", "player2"), default="player1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compile-policy", action="store_true",
        help="compile policy forward passes with torch.compile (experimental; benchmark before training)",
    )
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode used with --compile-policy",
    )
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument("--run-id", default="headless-showdown-self-play")
    args = parser.parse_args()
    if (args.episodes < 1 or args.temperature <= 0
            or not math.isfinite(args.team_selection_temperature)
            or args.team_selection_temperature < 0):
        parser.error("episodes/policy temperature must be positive; selector temperature must be finite and nonnegative")
    if args.team_selection_temperature > 0 and args.team_selection_checkpoint is None:
        parser.error("--team-selection-temperature requires --team-selection-checkpoint")
    if args.max_damage_action_steps is not None and any(
        step < 0 for step in args.max_damage_action_steps
    ):
        parser.error("--max-damage-action-steps must be nonnegative")
    if args.opponent_max_damage_action_steps is not None and any(
        step < 0 for step in args.opponent_max_damage_action_steps
    ):
        parser.error("--opponent-max-damage-action-steps must be nonnegative")
    if not 0.0 <= args.midgame_search_policy_mixture <= 1.0:
        parser.error("--midgame-search-policy-mixture must be in [0, 1]")
    if args.counterplay_teacher and args.max_damage_score_gate:
        parser.error("--counterplay-teacher and --max-damage-score-gate cannot be combined")
    if args.max_damage_gate_on_policy and not args.max_damage_score_gate:
        parser.error("--max-damage-gate-on-policy requires --max-damage-score-gate")
    if args.gate_policy_opponent and args.max_damage_score_gate:
        parser.error("--gate-policy-opponent cannot be combined with --max-damage-score-gate")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.player_strategy == "policy" and args.policy_checkpoint is None:
        parser.error("--policy-checkpoint is required with --player-strategy policy")
    if args.team_selection_checkpoint is not None and not args.team_selection_checkpoint.is_file():
        parser.error(f"team-selection checkpoint does not exist: {args.team_selection_checkpoint}")
    if (args.opponent_team_selection_checkpoint is not None
            and not args.opponent_team_selection_checkpoint.is_file()):
        parser.error(
            f"opponent team-selection checkpoint does not exist: "
            f"{args.opponent_team_selection_checkpoint}"
        )
    if args.opponent_strategy == "policy" and args.policy_checkpoint is None:
        parser.error("policy opponent requires --policy-checkpoint or --opponent-checkpoint")
    learned_search = args.opening_search == "learned" or args.midgame_search == "learned"
    if learned_search and args.state_value_checkpoint is None:
        parser.error("--state-value-checkpoint is required for learned search")
    if not learned_search and args.state_value_checkpoint is not None:
        parser.error("--state-value-checkpoint is only used by learned search")
    if args.midgame_search_policy_mixture > 0 and args.midgame_search != "learned":
        parser.error("--midgame-search-policy-mixture requires learned midgame search")
    if args.state_value_device is not None and not learned_search:
        parser.error("--state-value-device requires learned search")
    if args.state_value_device == "cuda" and not torch.cuda.is_available():
        parser.error("--state-value-device cuda requires CUDA support")
    if args.opening_search != "none":
        search_policy_strategy = (
            args.player_strategy if args.trainable_side == "player1"
            else args.opponent_strategy
        )
        if search_policy_strategy != "policy":
            parser.error(
                "matrix search requires the trainable side to use --policy-checkpoint"
            )
    midgame_search_strategy = (
        args.player_strategy if args.trainable_side == "player1"
        else args.opponent_strategy
    )
    if args.midgame_search != "none" and midgame_search_strategy != "policy":
        parser.error("midgame search requires the selected trainable side to use the policy")
    response_candidates = (
        args.search_max_candidates
        if args.search_max_response_candidates is None
        else args.search_max_response_candidates
    )
    if (args.search_hidden_worlds < 1 or args.search_max_candidates < 4
            or response_candidates < args.search_max_candidates
            or args.midgame_search_max_damage_response_candidates < 1
            or args.midgame_search_max_damage_response_candidates > response_candidates
            or args.search_chance_samples < 1 or args.search_workers < 1
            or args.midgame_search_worlds < 1 or args.midgame_search_chance_samples < 1
            or (args.midgame_search_max_chance_samples is not None
                and args.midgame_search_max_chance_samples < args.midgame_search_chance_samples)
            or args.midgame_search_timeout_seconds <= 0
            or args.midgame_search_max_decisions < 0):
        parser.error("search requires one hidden world, four candidate actions, one chance sample, and one worker")
    if (
        args.midgame_search_max_chance_samples is not None
        and args.midgame_search_max_chance_samples > args.midgame_search_chance_samples
        and args.search_action_selection != "maximin"
    ):
        parser.error("adaptive midgame chance sampling currently requires --search-action-selection maximin")
    model = _load_policy(
        args.policy_checkpoint,
        args.device,
        compile_policy=args.compile_policy,
        compile_mode=args.compile_mode,
    )
    team_selection_model, team_selection_config = _load_team_selector(
        args.team_selection_checkpoint
    )
    opponent_team_selection_model, opponent_team_selection_config = _load_team_selector(
        args.opponent_team_selection_checkpoint
    )
    opponent_model = None
    if args.opponent_checkpoint:
        if (
            args.policy_checkpoint is not None
            and args.policy_checkpoint.resolve() == args.opponent_checkpoint.resolve()
        ):
            opponent_model = model
        else:
            opponent_model = _load_policy(
                args.opponent_checkpoint,
                args.device,
                compile_policy=args.compile_policy,
                compile_mode=args.compile_mode,
            )
    state_value_evaluator = (
        load_state_value_checkpoint(
            args.state_value_checkpoint,
            map_location=args.state_value_device or args.device,
        )
        if learned_search else None
    )
    telemetry = TrainingTelemetry(args.telemetry_path, args.run_id) if args.telemetry_path else None
    if telemetry:
        telemetry.start(
            phase="headless_rollout",
            format=args.format,
            episodes=args.episodes,
            seed=args.seed,
            policy_checkpoint=str(args.policy_checkpoint) if args.policy_checkpoint else None,
            opponent_checkpoint=str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
            team_selection_checkpoint=(
                str(args.team_selection_checkpoint)
                if args.team_selection_checkpoint is not None else None
            ),
            player_strategy=args.player_strategy,
            opponent_strategy=args.opponent_strategy,
            opening_search=args.opening_search,
            midgame_search=args.midgame_search,
            midgame_search_worlds=args.midgame_search_worlds,
            midgame_search_chance_samples=args.midgame_search_chance_samples,
            midgame_search_max_chance_samples=args.midgame_search_max_chance_samples,
            midgame_search_max_decisions=args.midgame_search_max_decisions,
            midgame_search_policy_mixture=args.midgame_search_policy_mixture,
            search_response_coverage_diagnostic=args.search_response_coverage_diagnostic,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    started = time.perf_counter()
    results = []
    with HeadlessShowdownWorker(args.showdown_path) as worker, ExitStack() as stack:
        search_pool = (
            stack.enter_context(LocalShowdownSearchPool(
                format_id=args.format,
                showdown_path=args.showdown_path,
                workers=args.search_workers,
            ))
            if args.opening_search != "none" or args.midgame_search != "none" else None
        )
        for game in range(args.episodes):
            result = run_game(
                game=game,
                seed=args.seed,
                format_id=args.format,
                showdown_path=args.showdown_path,
                output_path=args.output,
                model=model,
                opponent_model=opponent_model,
                player_strategy=args.player_strategy,
                opponent_strategy=args.opponent_strategy,
                temperature=args.temperature,
                device=args.device,
                telemetry=telemetry,
                worker=worker,
                opening_search=args.opening_search,
                midgame_search=args.midgame_search,
                state_value_evaluator=state_value_evaluator,
                search_hidden_worlds=args.search_hidden_worlds,
                search_max_candidates=args.search_max_candidates,
                search_max_response_candidates=response_candidates,
                search_chance_samples=args.search_chance_samples,
                search_action_selection=args.search_action_selection,
                search_workers=args.search_workers,
                midgame_search_worlds=args.midgame_search_worlds,
                midgame_search_max_damage_response_candidates=(
                    args.midgame_search_max_damage_response_candidates
                ),
                midgame_search_chance_samples=args.midgame_search_chance_samples,
                midgame_search_max_chance_samples=args.midgame_search_max_chance_samples,
                midgame_search_timeout_seconds=args.midgame_search_timeout_seconds,
                midgame_search_max_decisions=args.midgame_search_max_decisions,
                midgame_search_policy_mixture=args.midgame_search_policy_mixture,
                search_response_coverage_diagnostic=args.search_response_coverage_diagnostic,
                search_pool=search_pool,
                episode_offset=args.episode_offset,
                trainable_side=args.trainable_side,
                team_selection_model=team_selection_model,
                team_selection_config=team_selection_config,
                team_selection_temperature=args.team_selection_temperature,
                opponent_team_selection_model=opponent_team_selection_model,
                opponent_team_selection_config=opponent_team_selection_config,
                record_episode_offset=args.record_episode_offset,
                deterministic_policy=args.deterministic_policy,
                first_turn_max_damage=(
                    args.first_turn_max_damage or args.max_damage_action_steps is not None
                ),
                max_damage_action_steps=(
                    set(args.max_damage_action_steps)
                    if args.max_damage_action_steps is not None else None
                ),
                opponent_max_damage_action_steps=(
                    set(args.opponent_max_damage_action_steps)
                    if args.opponent_max_damage_action_steps is not None else None
                ),
                max_damage_score_gate=args.max_damage_score_gate,
                max_damage_gate_on_policy=args.max_damage_gate_on_policy,
                gate_policy_opponent=args.gate_policy_opponent,
                counterplay_teacher=args.counterplay_teacher,
                counterplay_incoming_weight=args.counterplay_incoming_weight,
                counterplay_ko_bonus=args.counterplay_ko_bonus,
                counterplay_setup_bonus=args.counterplay_setup_bonus,
                counterplay_redirection=args.counterplay_redirection,
                swap_teams=args.swap_teams,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "episodes": len(results),
        "elapsed_seconds": elapsed,
        "matches_per_second": len(results) / elapsed,
        "max_damage_gate_on_policy": bool(args.max_damage_gate_on_policy),
        "max_damage_score_gate_overrides": sum(
            int(row.get("max_damage_score_gate_overrides", 0)) for row in results
        ),
        "policy_opponent_score_gate_overrides": sum(
            int(row.get("policy_opponent_score_gate_overrides", 0)) for row in results
        ),
        "gate_policy_opponent": bool(args.gate_policy_opponent),
        "counterplay_teacher_actions": sum(
            int(row.get("counterplay_teacher_actions", 0)) for row in results
        ),
        "first_turn_max_damage_teacher_episodes": sum(
            int(
                isinstance(row.get("first_turn_action_data"), dict)
                and row["first_turn_action_data"].get("source") == "max_damage"
            )
            for row in results
        ),
        "team_selection_checkpoint": (
            str(args.team_selection_checkpoint)
            if args.team_selection_checkpoint is not None else None
        ),
        "team_selection_applied_episodes": sum(
            bool(row.get("team_selection_applied", False)) for row in results
        ),
        "mean_steps": float(np.mean([row["steps"] for row in results])),
        "mean_runtime_metrics": {
            key: float(np.mean([row[key] for row in results]))
            for key in (
                "pre_match_setup_seconds", "preview_and_first_step_seconds",
                "observation_encoding_seconds", "action_selection_seconds",
                "policy_inference_seconds", "baseline_action_seconds",
                "action_proposal_seconds", "headless_engine_roundtrip_seconds",
                "trajectory_persist_seconds", "end_to_end_episode_seconds",
            )
        },
        "games": results,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if telemetry:
        telemetry.emit("run_completed", episodes=len(results), elapsed_seconds=elapsed, matches_per_second=len(results) / elapsed)


if __name__ == "__main__":
    main()
