"""Evaluate a native policy checkpoint against a random Showdown player."""

from __future__ import annotations

import argparse
import json
from itertools import count
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Keep imports working when invoked as ``python scripts/evaluate_native.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

# Evaluations frequently run many independent battles in separate processes.
# A per-process machine-sized Torch pool causes severe oversubscription in
# that workload, so default to one intra-op/inter-op worker and allow explicit
# tuning for hosts with a different profile.
_eval_torch_threads = max(1, int(os.environ.get("POKEMON_EVAL_TORCH_THREADS", "1")))
torch.set_num_threads(_eval_torch_threads)
try:
    torch.set_num_interop_threads(_eval_torch_threads)
except RuntimeError:
    # Another imported caller may already have initialized the inter-op pool.
    pass

from src.env.showdown_env import ShowdownDoublesEnv
from src.env.public_battle_history import PublicBattleHistory
from src.features.public_mechanics import encode_public_mechanics
from src.baselines.gimmick_opponents import (
    choose_no_guard_fissure_action, choose_perish_trap_action,
)
from src.models.policy import TransformerPolicy
from src.training.observability import TrainingTelemetry
from src.training.native_ppo import decode_native_action_index
from src.training.perspective_data import project_episode_to_player_views
from src.training.self_play import PolicyAgent
from src.training.team_selection import (
    TeamSelectionPolicy, decode_action, encode_action, preview_command, random_action,
    team_features,
)
from poke_env.ps_client import AccountConfiguration
from scripts.run_showdown_self_play import (
    EpisodeTimeout,
    LOCAL_SERVER_CONFIGURATION,
    archetype_preview_action,
    choose_masked_actions,
    current_observations,
    generate_showdown_champions_team_data,
    pack_showdown_team,
    shuffle_archetype_team_slots,
    sanitize_native_action,
    team_to_showdown_text,
)


def _timeout(_signum: int, _frame: Any) -> None:
    raise EpisodeTimeout("evaluation battle exceeded timeout")


_EVALUATION_ACCOUNT_NONCE = count()


def _trace_action(action: int, battle: Any, position: int) -> dict[str, Any]:
    """Describe a native doubles action using the live battle request."""
    descriptor = decode_native_action_index(int(action))
    result: dict[str, Any] = {
        "index": int(action),
        "kind": descriptor.kind,
        "mechanic": descriptor.mechanic,
        "target_code": descriptor.target_code,
    }
    active = battle.active_pokemon[position]
    result["active"] = getattr(active, "species", None) if active is not None else None
    if descriptor.kind == "switch" and descriptor.switch_index is not None:
        team = list(battle.team.values())
        switch_index = int(action) - 1
        if 0 <= switch_index < len(team):
            result["switch_to"] = getattr(team[switch_index], "species", None)
    elif descriptor.kind == "move" and descriptor.move_index is not None and active is not None:
        known_moves = list(active.moves.values())[:4]
        available_moves = list(battle.available_moves[position])
        available_ids = [move.id for move in available_moves]
        known_ids = [move.id for move in known_moves]
        move_list = (
            available_moves
            if len(available_ids) == 1 and available_ids[0] not in known_ids
            else known_moves
        )
        if descriptor.move_index < len(move_list):
            result["move"] = move_list[descriptor.move_index].id
    return result


from src.baselines.native_damage import (
    choose_counterplay_action as _choose_counterplay_action,
    choose_max_damage_action as _choose_max_damage_action,
    choose_public_counterplay_action as _choose_public_counterplay_action,
    score_max_damage_action as _score_max_damage_action,
)


def evaluate(
    checkpoint: Path | None,
    games: int,
    output: Path,
    seed: int,
    battle_format: str,
    episode_timeout: int,
    device: str,
    deterministic: bool = True,
    opponent: str = "random",
    opponent_checkpoint: Path | None = None,
    player: str = "policy",
    team_selection_checkpoint: Path | None = None,
    opponent_team_selection_checkpoint: Path | None = None,
    team_selection_mode: str = "random",
    team_selection_side: str = "both",
    trace_decisions: bool = False,
    counterplay_incoming_weight: float = 1.0,
    counterplay_support_actions: bool = False,
    counterplay_ko_bonus: float = 0.7,
    counterplay_threat_aggregation: str = "sum",
    counterplay_respect_protect_streaks: bool = True,
    counterplay_cap_overkill: bool = True,
    counterplay_prior_weight: float | None = None,
    public_counterplay_prior_weight: float = 0.0,
    public_counterplay_prior_min_margin: float = 0.0,
    public_counterplay_incoming_weight: float | None = None,
    public_counterplay_ko_bonus: float | None = None,
    public_counterplay_setup_bonus: float = 0.0,
    public_counterplay_switch_penalty: float = 0.05,
    public_counterplay_support_actions: bool = False,
    public_counterplay_redirection: bool = False,
    public_counterplay_joint_outgoing: bool = False,
    public_counterplay_sequential_damage: bool = False,
    public_counterplay_move_objective: str = "damage",
    counterplay_predict_policy_opponent: bool = False,
    team_seed: int | None = None,
    team1_json: Path | None = None,
    team2_json: Path | None = None,
    preview_action_player1: int | None = None,
    preview_action_player2: int | None = None,
    player_view_output: Path | None = None,
    max_damage_score_gate: bool = False,
    max_damage_score_gate_margin: float = 0.0,
    intervention_target_player: int | None = None,
    first_turn_max_damage: bool = False,
    max_damage_action_steps: list[int] | None = None,
    allow_privileged_opponent_action_oracle: bool = False,
) -> dict[str, Any]:
    model = None
    if player == "policy":
        if checkpoint is None:
            raise ValueError("checkpoint is required for a policy player")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = TransformerPolicy.from_checkpoint_state(state)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
        if counterplay_prior_weight is None:
            counterplay_prior_weight = float(state.get("counterplay_prior_weight", 0.0))
    opening_action_steps = set(max_damage_action_steps or [])
    if first_turn_max_damage and max_damage_action_steps is None:
        opening_action_steps.add(0)
    if any(step < 0 for step in opening_action_steps):
        raise ValueError("max-damage action steps must be nonnegative")
    elif counterplay_prior_weight is None:
        counterplay_prior_weight = 0.0
    if not np.isfinite(counterplay_prior_weight) or counterplay_prior_weight < 0:
        raise ValueError("counterplay_prior_weight must be finite and nonnegative")
    if not np.isfinite(public_counterplay_prior_weight) or public_counterplay_prior_weight < 0:
        raise ValueError("public_counterplay_prior_weight must be finite and nonnegative")
    if not np.isfinite(public_counterplay_prior_min_margin) or public_counterplay_prior_min_margin < 0:
        raise ValueError("public_counterplay_prior_min_margin must be finite and nonnegative")
    public_incoming_weight = (
        counterplay_incoming_weight
        if public_counterplay_incoming_weight is None
        else public_counterplay_incoming_weight
    )
    public_ko_bonus = (
        counterplay_ko_bonus
        if public_counterplay_ko_bonus is None
        else public_counterplay_ko_bonus
    )
    if not np.isfinite(public_incoming_weight) or public_incoming_weight < 0:
        raise ValueError("public_counterplay_incoming_weight must be finite and nonnegative")
    if not np.isfinite(public_ko_bonus) or public_ko_bonus < 0:
        raise ValueError("public_counterplay_ko_bonus must be finite and nonnegative")
    if not np.isfinite(public_counterplay_switch_penalty) or public_counterplay_switch_penalty < 0:
        raise ValueError("public_counterplay_switch_penalty must be finite and nonnegative")
    if public_counterplay_move_objective not in {"damage", "hp_fraction"}:
        raise ValueError("public_counterplay_move_objective must be 'damage' or 'hp_fraction'")
    if not np.isfinite(max_damage_score_gate_margin) or max_damage_score_gate_margin < 0:
        raise ValueError("max_damage_score_gate_margin must be finite and nonnegative")
    if intervention_target_player not in (None, 0, 1):
        raise ValueError("intervention_target_player must be None, 0, or 1")
    if not allow_privileged_opponent_action_oracle:
        # Counterplay scores in the legacy path are computed from the other
        # player's mirrored battle object, which contains private team state.
        counterplay_prior_weight = 0.0
        if counterplay_predict_policy_opponent:
            raise ValueError(
                "counterplay_predict_policy_opponent requires the explicitly "
                "privileged oracle flag; it predicts from a private battle view"
            )
    opponent_model = None
    opponent_prior_weight = 0.0
    if opponent == "policy":
        if opponent_checkpoint is None:
            raise ValueError("opponent_checkpoint is required for policy opponents")
        opponent_state = torch.load(opponent_checkpoint, map_location="cpu", weights_only=True)
        opponent_model = TransformerPolicy.from_checkpoint_state(opponent_state)
        opponent_model.load_state_dict(opponent_state["model_state_dict"], strict=False)
        opponent_model.eval()
        opponent_prior_weight = float(opponent_state.get("counterplay_prior_weight", 0.0))
        if not allow_privileged_opponent_action_oracle:
            opponent_prior_weight = 0.0
    def load_selection_policy(path: Path | None) -> TeamSelectionPolicy | None:
        if path is None or team_selection_mode != "policy":
            return None
        selection_state = torch.load(path, map_location="cpu", weights_only=True)
        config = selection_state.get("model_config", {})
        if not {"feature_dim", "hidden_dim"} <= config.keys():
            raise ValueError(
                f"{path} is not a team-selection checkpoint "
                "(model_config must contain feature_dim and hidden_dim)"
            )
        policy = TeamSelectionPolicy(
            feature_dim=int(config["feature_dim"]), hidden_dim=int(config["hidden_dim"]),
            opponent_conditioned=bool(config.get("opponent_conditioned", False)),
            action_conditioned=bool(config.get("action_conditioned", False)),
        )
        policy.load_state_dict(selection_state["model_state_dict"])
        policy.eval()
        return policy

    selection_policy = load_selection_policy(team_selection_checkpoint)
    opponent_selection_policy = load_selection_policy(opponent_team_selection_checkpoint)
    rng = np.random.default_rng(seed)
    # Matched evaluations deliberately reuse battle seeds across policy arms.
    # Give each evaluator invocation a distinct account name so Showdown does
    # not reject a new room while an earlier matched arm is still disconnecting.
    account_nonce = next(_EVALUATION_ACCOUNT_NONCE) % 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    if player_view_output is not None:
        player_view_output.parent.mkdir(parents=True, exist_ok=True)
        player_view_output.write_text("", encoding="utf-8")
    player_view_records_written = 0
    wins = losses = draws = timeouts = 0
    for game in range(games):
        torch.manual_seed(seed + game)
        roster_seed = seed if team_seed is None else team_seed
        # Keep each side's random preview choice independent. Otherwise a
        # trained player-1 selector consumes no RNG value while random player
        # 1 does, shifting player 2's sampled team and breaking matched-seed
        # selector comparisons.
        preview_rngs = {
            "player1": np.random.default_rng(seed + 2 * game),
            "player2": np.random.default_rng(seed + 2 * game + 1),
        }
        def load_fixed_team(path: Path | None) -> list[dict[str, Any]] | None:
            if path is None:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("pokemon", payload.get("team"))
            if not isinstance(payload, list) or len(payload) != 6:
                raise ValueError(f"{path} must contain exactly six Pokémon sets (or an object with a pokemon/team list)")
            return payload

        fixed_team1 = load_fixed_team(team1_json)
        fixed_team2 = load_fixed_team(team2_json)
        team1_data = fixed_team1 or generate_showdown_champions_team_data(roster_seed + 2 * game, battle_format)
        team2_data = fixed_team2 or generate_showdown_champions_team_data(
            roster_seed + 2 * game + 1,
            battle_format,
            archetype=(opponent if opponent in {"no_guard_fissure", "perish_trap"} else None),
        )
        if fixed_team2 is None and opponent in {"no_guard_fissure", "perish_trap"}:
            team2_data = shuffle_archetype_team_slots(
                team2_data, roster_seed + 2 * game + 1 + 7_000_000
            )
        env = ShowdownDoublesEnv(
            battle_format=battle_format,
            server_configuration=LOCAL_SERVER_CONFIGURATION,
            # Include the process id so a timed-out run cannot poison the
            # next benchmark with Showdown's `nametaken` response.
            account_configuration1=AccountConfiguration(
                f"EA{(seed + game) & 0xffffffff:08x}{os.getpid() % 1000:03d}{account_nonce:03d}", None
            ),
            account_configuration2=AccountConfiguration(
                f"EB{(seed + game) & 0xffffffff:08x}{os.getpid() % 1000:03d}{account_nonce:03d}", None
            ),
            team=pack_showdown_team(team_to_showdown_text(team1_data)),
            # The local server emits showteam after /team.  Let poke-env
            # consume that exchange before exposing battle orders.
            accept_open_team_sheet=False,
            start_listening=True,
            choose_on_teampreview=True,
            fake=False,
            strict=False,
        )
        env.agent2.update_team(pack_showdown_team(team_to_showdown_text(team2_data)))
        if team_selection_mode in ("random", "policy"):
            commands = {}
            preview_actions: dict[str, int] = {}
            preview_teams = {"player1": team1_data, "player2": team2_data}
            for side, name, data, opponent_data in (
                ("player1", env.possible_agents[0], team1_data, team2_data),
                ("player2", env.possible_agents[1], team2_data, team1_data),
            ):
                fixed_preview_action = (
                    preview_action_player1 if side == "player1" else preview_action_player2
                )
                if side == "player2" and fixed_team2 is None and fixed_preview_action is None and opponent in {
                    "no_guard_fissure", "perish_trap"
                }:
                    fixed_preview_action = archetype_preview_action(data, opponent)
                side_policy = (
                    selection_policy if name == env.possible_agents[0]
                    else opponent_selection_policy
                    if opponent_selection_policy is not None
                    else selection_policy if team_selection_side == "both"
                    else None
                )
                use_policy = side_policy is not None and (
                    team_selection_side == "both"
                    or name == env.possible_agents[0]
                    or opponent_selection_policy is not None
                )
                if fixed_preview_action is not None:
                    if not 0 <= fixed_preview_action < 90:
                        raise ValueError("fixed preview actions must be in [0, 90)")
                    action = fixed_preview_action
                elif use_policy:
                    with torch.no_grad():
                        own_features = team_features(data, side_policy.feature_dim).unsqueeze(0)
                        opponent_team_features = team_features(
                            opponent_data, side_policy.feature_dim,
                            revealed_only=True,
                        ).unsqueeze(0)
                        action = int(side_policy(
                            own_features,
                            opponent_features=opponent_team_features
                            if side_policy.opponent_conditioned else None,
                        ).argmax(-1).item())
                else:
                    action = random_action(preview_rngs["player1" if name == env.possible_agents[0] else "player2"])
                preview_actions[side] = action
                commands[name] = preview_command(action)
            from scripts.run_showdown_self_play import install_atomic_team_preview
            install_atomic_team_preview(env, commands)
        else:
            preview_actions = {}
            preview_teams = {"player1": team1_data, "player2": team2_data}
        timed_out = False
        action_adjustments = 0
        max_damage_score_gate_overrides = 0
        first_turn_max_damage_overrides = 0
        evaluated_policy_decisions = [0, 0]
        phase_trace: list[dict[str, Any]] = []
        player_view_trajectory: list[dict[str, Any]] = []
        # Keep the same public, closed-sheet context used during headless
        # rollout collection. Checkpoints with a public-context input must
        # receive it at evaluation time too.
        public_history = PublicBattleHistory()
        processed_replay_events = 0
        try:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(episode_timeout)
            env.reset(seed=seed + game)
            steps = 0
            while env.agents:
                # The two rooms are independent and can report their terminal
                # request on different websocket callbacks.  Poke-env removes
                # both agents only when battle1 finishes, so do not keep
                # polling a finished battle's empty action mask.
                if any(
                    battle is not None and battle.finished
                    for battle in (env.battle1, env.battle2)
                ):
                    break
                observed = current_observations(env)
                replay_events = getattr(env.battle1, "_replay_data", ())
                new_replay_events = replay_events[processed_replay_events:]
                public_history.consume(
                    [
                        "|" + "|".join(str(part) for part in event[1:])
                        for event in new_replay_events
                        if len(event) > 1
                    ]
                )
                processed_replay_events = len(replay_events)
                public_context_by_agent: dict[str, np.ndarray] = {}
                for index, agent_name in enumerate(env.possible_agents):
                    model_for_agent = (
                        model if index == 0 and player == "policy"
                        else opponent_model if index == 1 and opponent == "policy"
                        else None
                    )
                    if model_for_agent is None or not getattr(model_for_agent, "public_context_dim", 0):
                        continue
                    battle = env.battle1 if index == 0 else env.battle2
                    own_team = team1_data if index == 0 else team2_data
                    opponent_team = team2_data if index == 0 else team1_data
                    public_opponent_team = [
                        {"species": str(member.get("species") or member.get("name", ""))}
                        for member in opponent_team
                    ]
                    public_context_by_agent[agent_name] = encode_public_mechanics(
                        public_history.for_battle(battle),
                        observed[agent_name]["observation"],
                        own_team,
                        public_opponent_team,
                        battle_format,
                    )
                if any(
                    any(
                        not part.any()
                        for part in np.array_split(
                            np.asarray(payload["action_mask"], dtype=np.int8), 2
                        )
                    )
                    for payload in observed.values()
                ):
                    time.sleep(0.01)
                    continue
                actions = {}
                predicted_max_damage_action = None
                predicted_player1_max_damage_action = None
                if ((player == "counterplay")
                        or (allow_privileged_opponent_action_oracle and player == "policy")) and env.agent2_to_move:
                    opponent_agent = env.possible_agents[1]
                    if opponent == "no_guard_fissure":
                        predicted_max_damage_action = choose_no_guard_fissure_action(
                            observed[opponent_agent]["action_mask"], env.battle2
                        )
                    elif opponent == "perish_trap":
                        predicted_max_damage_action = choose_perish_trap_action(
                            observed[opponent_agent]["action_mask"], env.battle2
                        )
                    else:
                        predicted_max_damage_action = _choose_max_damage_action(
                            observed[opponent_agent]["action_mask"], env.battle2
                        )
                if ((opponent == "counterplay")
                        or (allow_privileged_opponent_action_oracle
                            and opponent == "policy" and opponent_prior_weight > 0)) and env.agent1_to_move:
                    player_agent = env.possible_agents[0]
                    predicted_player1_max_damage_action = _choose_max_damage_action(
                        observed[player_agent]["action_mask"], env.battle1
                    )
                predicted_policy_opponent_action = None
                policy_opponent_scores = None
                policy_opponent_teacher_first = None
                if (counterplay_predict_policy_opponent and opponent == "policy"
                        and env.agent2_to_move):
                    opponent_agent = env.possible_agents[1]
                    if opponent_prior_weight > 0:
                        teacher_action, policy_opponent_scores, _ = _choose_counterplay_action(
                            observed[opponent_agent]["action_mask"], env.battle2, env.battle1,
                            predicted_player1_max_damage_action,
                            incoming_weight=counterplay_incoming_weight,
                            support_actions=counterplay_support_actions,
                            ko_bonus=counterplay_ko_bonus,
                            respect_protect_streaks=counterplay_respect_protect_streaks,
                            cap_overkill=counterplay_cap_overkill,
                            return_action_scores=True,
                        )
                        policy_opponent_teacher_first = int(teacher_action[0])
                    opponent_agent_policy = PolicyAgent(
                        opponent_model, team2_data, team1_data,
                        deterministic=deterministic, device=device,
                        public_context_features=public_context_by_agent.get(opponent_agent),
                    )
                    predicted_policy_opponent_action = opponent_agent_policy.predict_native_with_value(
                        observed[opponent_agent]["observation"],
                        observed[opponent_agent]["action_mask"],
                        predicted_opponent_action=(
                            predicted_player1_max_damage_action
                            if allow_privileged_opponent_action_oracle else None
                        ),
                        counterplay_action_scores=policy_opponent_scores,
                        counterplay_prior_weight=opponent_prior_weight,
                        counterplay_teacher_first_action=policy_opponent_teacher_first,
                    )[0]
                if opponent == "counterplay" and env.agent1_to_move:
                    player_agent = env.possible_agents[0]
                    opponent_agent = env.possible_agents[1]
                    player_mask = np.asarray(
                        observed[player_agent]["action_mask"], dtype=np.int8
                    )
                    opponent_mask = np.asarray(
                        observed[opponent_agent]["action_mask"], dtype=np.int8
                    )
                    if all(
                        part.any()
                        for action_mask in (player_mask, opponent_mask)
                        for part in np.array_split(action_mask, 2)
                    ):
                        predicted_player1_max_damage_action = _choose_max_damage_action(
                            player_mask, env.battle1
                        )
                        try:
                            predicted_policy_opponent_action = _choose_counterplay_action(
                                opponent_mask,
                                env.battle2,
                                env.battle1,
                                predicted_player1_max_damage_action,
                                incoming_weight=counterplay_incoming_weight,
                                support_actions=counterplay_support_actions,
                                ko_bonus=counterplay_ko_bonus,
                                respect_protect_streaks=counterplay_respect_protect_streaks,
                                cap_overkill=counterplay_cap_overkill,
                            )
                        except (ValueError, IndexError):
                            # A fresh individual mask can briefly contain no
                            # compatible joint pair while the other room is
                            # still updating. The candidate can fall back to
                            # its regular prediction; the teacher is recomputed
                            # from the fresh mask when its request arrives.
                            predicted_policy_opponent_action = None
                for index, agent in enumerate(env.agents):
                    # DoublesEnv can return a battle update for only one of
                    # its two clients. Actions for the other battle are not
                    # requested and its cached mask may already be stale.
                    # Do not ask an opponent/policy to choose (or enqueue) an
                    # action until that client has a fresh request.
                    if not (env.agent1_to_move if index == 0 else env.agent2_to_move):
                        continue
                    is_evaluated_policy = (
                        (index == 0 and player == "policy")
                        or (index == 1 and opponent == "policy")
                    )
                    is_intervention_target = (
                        is_evaluated_policy
                        and (
                            intervention_target_player is None
                            or index == intervention_target_player
                        )
                    )
                    if (
                        opening_action_steps
                        and is_intervention_target
                        and evaluated_policy_decisions[index] in opening_action_steps
                    ):
                        battle = env.battle1 if index == 0 else env.battle2
                        actions[agent] = _choose_max_damage_action(
                            observed[agent]["action_mask"], battle
                        )
                        first_turn_max_damage_overrides += 1
                    elif index == 0 and player == "policy":
                        counterplay_scores = None
                        counterplay_pair_scores = None
                        counterplay_teacher_first = None
                        effective_prior_weight = counterplay_prior_weight
                        if public_counterplay_prior_weight > 0:
                            (
                                teacher_action,
                                counterplay_scores,
                                _,
                                counterplay_pair_scores,
                            ) = _choose_public_counterplay_action(
                                observed[agent]["action_mask"], env.battle1, public_history,
                                incoming_weight=public_incoming_weight,
                                ko_bonus=public_ko_bonus,
                                switch_penalty=public_counterplay_switch_penalty,
                                support_actions=public_counterplay_support_actions,
                                redirect_actions=public_counterplay_redirection,
                                joint_outgoing=public_counterplay_joint_outgoing,
                                sequential_damage=public_counterplay_sequential_damage,
                                threat_move_objective=public_counterplay_move_objective,
                                threat_aggregation=counterplay_threat_aggregation,
                                return_action_scores=True,
                            )
                            counterplay_teacher_first = int(teacher_action[0])
                            effective_prior_weight = public_counterplay_prior_weight
                        elif counterplay_prior_weight > 0:
                            teacher_action, counterplay_scores, _ = _choose_counterplay_action(
                                observed[agent]["action_mask"], env.battle1, env.battle2,
                                predicted_policy_opponent_action
                                if predicted_policy_opponent_action is not None
                                else predicted_max_damage_action,
                                incoming_weight=counterplay_incoming_weight,
                                support_actions=counterplay_support_actions,
                                ko_bonus=counterplay_ko_bonus,
                                respect_protect_streaks=counterplay_respect_protect_streaks,
                                cap_overkill=counterplay_cap_overkill,
                                return_action_scores=True,
                            )
                            counterplay_teacher_first = int(teacher_action[0])
                        policy = PolicyAgent(
                            model, team1_data, team2_data,
                            deterministic=deterministic, device=device,
                            public_context_features=public_context_by_agent.get(agent),
                        )
                        proposed = policy.predict_native_with_value(
                            observed[agent]["observation"], observed[agent]["action_mask"],
                            predicted_opponent_action=(
                                (predicted_policy_opponent_action
                                 if predicted_policy_opponent_action is not None
                                 else predicted_max_damage_action)
                                if allow_privileged_opponent_action_oracle else None
                            ),
                            counterplay_action_scores=counterplay_scores,
                            counterplay_pair_scores=counterplay_pair_scores,
                            counterplay_prior_weight=effective_prior_weight,
                            counterplay_prior_min_margin=public_counterplay_prior_min_margin,
                            counterplay_teacher_first_action=counterplay_teacher_first,
                        )[0]
                        actions[agent], adjusted = sanitize_native_action(
                            observed[agent]["action_mask"], proposed
                        )
                    elif index == 0 and player == "max_damage":
                        actions[agent] = _choose_max_damage_action(observed[agent]["action_mask"], env.battle1)
                    elif index == 0 and player == "counterplay":
                        actions[agent] = _choose_counterplay_action(
                            observed[agent]["action_mask"], env.battle1, env.battle2,
                            predicted_max_damage_action,
                            incoming_weight=counterplay_incoming_weight,
                            support_actions=counterplay_support_actions,
                            ko_bonus=counterplay_ko_bonus,
                            respect_protect_streaks=counterplay_respect_protect_streaks,
                            cap_overkill=counterplay_cap_overkill,
                        )
                    elif index == 0 and player == "public_counterplay":
                        actions[agent] = _choose_public_counterplay_action(
                            observed[agent]["action_mask"], env.battle1, public_history,
                            incoming_weight=public_incoming_weight,
                            ko_bonus=public_ko_bonus,
                            setup_bonus=public_counterplay_setup_bonus,
                            switch_penalty=public_counterplay_switch_penalty,
                            support_actions=public_counterplay_support_actions,
                            redirect_actions=public_counterplay_redirection,
                            joint_outgoing=public_counterplay_joint_outgoing,
                            sequential_damage=public_counterplay_sequential_damage,
                            threat_move_objective=public_counterplay_move_objective,
                            threat_aggregation=counterplay_threat_aggregation,
                        )
                    elif index == 0 and player == "random":
                        actions.update(choose_masked_actions({agent: observed[agent]}, rng))
                    else:
                        if opponent == "random":
                            actions.update(choose_masked_actions({agent: observed[agent]}, rng))
                        elif opponent == "max_damage":
                            if predicted_max_damage_action is None:
                                predicted_max_damage_action = _choose_max_damage_action(
                                    observed[agent]["action_mask"], env.battle2
                                )
                            actions[agent] = predicted_max_damage_action
                        elif opponent == "counterplay":
                            actions[agent] = (
                                predicted_policy_opponent_action
                                if predicted_policy_opponent_action is not None
                                else _choose_counterplay_action(
                                    observed[agent]["action_mask"], env.battle2, env.battle1,
                                    predicted_player1_max_damage_action,
                                    incoming_weight=counterplay_incoming_weight,
                                    support_actions=counterplay_support_actions,
                                    ko_bonus=counterplay_ko_bonus,
                                    respect_protect_streaks=counterplay_respect_protect_streaks,
                                    cap_overkill=counterplay_cap_overkill,
                                )
                            )
                        elif opponent == "public_counterplay":
                            actions[agent] = _choose_public_counterplay_action(
                                observed[agent]["action_mask"], env.battle2, public_history,
                                incoming_weight=public_incoming_weight,
                                ko_bonus=public_ko_bonus,
                                setup_bonus=public_counterplay_setup_bonus,
                                switch_penalty=public_counterplay_switch_penalty,
                                support_actions=public_counterplay_support_actions,
                                redirect_actions=public_counterplay_redirection,
                                joint_outgoing=public_counterplay_joint_outgoing,
                                sequential_damage=public_counterplay_sequential_damage,
                                threat_move_objective=public_counterplay_move_objective,
                                threat_aggregation=counterplay_threat_aggregation,
                            )
                        elif opponent == "no_guard_fissure":
                            actions[agent] = choose_no_guard_fissure_action(
                                observed[agent]["action_mask"], env.battle2
                            )
                        elif opponent == "perish_trap":
                            actions[agent] = choose_perish_trap_action(
                                observed[agent]["action_mask"], env.battle2
                            )
                        elif opponent == "policy":
                            if predicted_policy_opponent_action is not None:
                                actions[agent] = predicted_policy_opponent_action
                            else:
                                opponent_scores = None
                                opponent_pair_scores = None
                                opponent_teacher_first = None
                                if public_counterplay_prior_weight > 0:
                                    (
                                        teacher_action,
                                        opponent_scores,
                                        _,
                                        opponent_pair_scores,
                                    ) = _choose_public_counterplay_action(
                                        observed[agent]["action_mask"], env.battle2,
                                        public_history,
                                        incoming_weight=public_incoming_weight,
                                        ko_bonus=public_ko_bonus,
                                        setup_bonus=public_counterplay_setup_bonus,
                                        switch_penalty=public_counterplay_switch_penalty,
                                        support_actions=public_counterplay_support_actions,
                                        redirect_actions=public_counterplay_redirection,
                                        joint_outgoing=public_counterplay_joint_outgoing,
                                        sequential_damage=public_counterplay_sequential_damage,
                                        threat_move_objective=public_counterplay_move_objective,
                                        threat_aggregation=counterplay_threat_aggregation,
                                        return_action_scores=True,
                                    )
                                    opponent_teacher_first = int(teacher_action[0])
                                    opponent_prior_weight = public_counterplay_prior_weight
                                elif opponent_prior_weight > 0:
                                    teacher_action, opponent_scores, _ = _choose_counterplay_action(
                                        observed[agent]["action_mask"], env.battle2, env.battle1,
                                        predicted_player1_max_damage_action,
                                        incoming_weight=counterplay_incoming_weight,
                                        support_actions=counterplay_support_actions,
                                        ko_bonus=counterplay_ko_bonus,
                                        respect_protect_streaks=counterplay_respect_protect_streaks,
                                        cap_overkill=counterplay_cap_overkill,
                                        return_action_scores=True,
                                    )
                                    opponent_teacher_first = int(teacher_action[0])
                                opponent_agent = PolicyAgent(
                                    opponent_model, team2_data, team1_data,
                                    deterministic=deterministic, device=device,
                                    public_context_features=public_context_by_agent.get(agent),
                                )
                                actions[agent] = opponent_agent.predict_native_with_value(
                                    observed[agent]["observation"], observed[agent]["action_mask"],
                                    predicted_opponent_action=(
                                        predicted_player1_max_damage_action
                                        if allow_privileged_opponent_action_oracle else None
                                    ),
                                    counterplay_action_scores=opponent_scores,
                                    counterplay_pair_scores=opponent_pair_scores,
                                    counterplay_prior_weight=opponent_prior_weight,
                                    counterplay_prior_min_margin=public_counterplay_prior_min_margin,
                                    counterplay_teacher_first_action=opponent_teacher_first,
                                )[0]
                        else:
                            raise ValueError(f"unknown opponent: {opponent}")
                for action_agent, proposed in actions.items():
                    if max_damage_score_gate and (
                        intervention_target_player is None
                        or index == intervention_target_player
                    ):
                        action_index = env.possible_agents.index(action_agent)
                        gate_battle = None
                        if action_index == 0 and player == "policy":
                            gate_battle = env.battle1
                        elif action_index == 1 and opponent == "policy" and opponent_model is not None:
                            gate_battle = env.battle2
                        if gate_battle is not None:
                            gate_action = _choose_max_damage_action(
                                observed[action_agent]["action_mask"], gate_battle
                            )
                            policy_score = sum(
                                _score_max_damage_action(int(value), gate_battle, slot)
                                for slot, value in enumerate(proposed)
                            )
                            gate_score = sum(
                                _score_max_damage_action(int(value), gate_battle, slot)
                                for slot, value in enumerate(gate_action)
                            )
                            required_improvement = max(
                                1e-6, abs(policy_score) * max_damage_score_gate_margin
                            )
                            if gate_score > policy_score + required_improvement:
                                proposed = gate_action
                                max_damage_score_gate_overrides += 1
                    actions[action_agent], adjusted = sanitize_native_action(
                        observed[action_agent]["action_mask"], proposed
                    )
                    action_adjustments += adjusted
                    action_index = env.possible_agents.index(action_agent)
                    if (
                        (action_index == 0 and player == "policy")
                        or (action_index == 1 and opponent == "policy")
                    ):
                        evaluated_policy_decisions[action_index] += 1
                if player_view_output is not None:
                    player_view_trajectory.append({
                        "turn": int(getattr(env.battle1, "turn", 0)),
                        "observations": {
                            agent: np.asarray(payload["observation"], dtype=np.float32).tolist()
                            for agent, payload in observed.items()
                        },
                        "action_masks": {
                            agent: np.asarray(payload["action_mask"], dtype=np.int8).tolist()
                            for agent, payload in observed.items()
                        },
                        "actions": {
                            agent: np.asarray(action, dtype=np.int64).tolist()
                            for agent, action in actions.items()
                        },
                        "opponent_species": {
                            env.possible_agents[0]: [
                                str(pokemon.get("species") or pokemon.get("name", ""))
                                for pokemon in team2_data
                            ],
                            env.possible_agents[1]: [
                                str(pokemon.get("species") or pokemon.get("name", ""))
                                for pokemon in team1_data
                            ],
                        },
                        "public_mechanics": {
                            env.possible_agents[0]: public_history.for_battle(env.battle1),
                            env.possible_agents[1]: public_history.for_battle(env.battle2),
                        },
                    })
                decision_trace: dict[str, Any] = {}
                if trace_decisions:
                    for index, side in enumerate(("player1", "player2")):
                        agent = env.possible_agents[index]
                        battle = env.battle1 if index == 0 else env.battle2
                        if agent not in actions or battle is None:
                            continue
                        decision_trace[side] = {
                            "active": [
                                getattr(pokemon, "species", None)
                                if pokemon is not None else None
                                for pokemon in battle.active_pokemon[:2]
                            ],
                            "actions": [
                                _trace_action(int(action), battle, position)
                                for position, action in enumerate(actions[agent])
                            ],
                        }
                _, rewards, terminated, truncated, infos = env.step(actions)
                steps += 1
                if player_view_output is not None and player_view_trajectory:
                    transition = player_view_trajectory[-1]
                    transition["rewards"] = {
                        agent: float(value) for agent, value in rewards.items()
                    }
                    transition["reward_components"] = {
                        agent: dict(infos.get(agent, {}).get("reward_components", {}))
                        for agent in env.possible_agents
                    }
                    transition["terminated"] = {
                        agent: bool(value) for agent, value in terminated.items()
                    }
                    transition["truncated"] = {
                        agent: bool(value) for agent, value in truncated.items()
                    }
                if trace_decisions:
                    components = infos.get(env.possible_agents[0], {}).get("reward_components", {})
                    post_state: dict[str, Any] = {}
                    for index, side in enumerate(("player1", "player2")):
                        agent = env.possible_agents[index]
                        battle = env.battle1 if index == 0 else env.battle2
                        if agent not in actions or battle is None:
                            continue
                        own_team = list(battle.team.values())
                        post_state[side] = [
                            {
                                "species": getattr(mon, "species", None),
                                "hp_fraction": (
                                    float(mon.current_hp) / float(mon.max_hp)
                                    if mon.current_hp is not None and mon.max_hp else 0.0
                                ),
                                "fainted": bool(mon.fainted),
                                "ability": getattr(mon, "ability", None),
                                "status": getattr(getattr(mon, "status", None), "name", None),
                                "effects": {
                                    getattr(effect, "name", str(effect)): int(count)
                                    for effect, count in (getattr(mon, "effects", {}) or {}).items()
                                },
                            }
                            for mon in own_team
                        ]
                    phase_trace.append({
                        "step": steps,
                        "hp_progress": float(components.get("hp_progress", 0.0)),
                        "terminal_outcome": float(components.get("terminal_outcome", 0.0)),
                        "total": float(components.get("total", rewards.get(env.possible_agents[0], 0.0))),
                        "decisions": decision_trace,
                        "post_state": post_state,
                    })
                # Showdown's turn limit is a valid draw even if the websocket
                # terminal message has not reached poke-env yet.
                if max(
                    getattr(battle, "turn", 0) for battle in (env.battle1, env.battle2)
                    if battle is not None
                ) >= 100:
                    break
                if all(terminated.values()) or all(truncated.values()):
                    break
            player_reward = float(rewards.get(env.possible_agents[0], 0.0))
            if player_reward > 0:
                wins += 1
            elif player_reward < 0:
                losses += 1
            else:
                draws += 1
            record = {"game": game, "team_seed": roster_seed, "result": "win" if player_reward > 0 else "loss" if player_reward < 0 else "draw", "steps": steps, "action_adjustments": action_adjustments, "counterplay_prior_weight": counterplay_prior_weight, "max_damage_score_gate_overrides": max_damage_score_gate_overrides, "max_damage_score_gate_margin": max_damage_score_gate_margin, "first_turn_max_damage_overrides": first_turn_max_damage_overrides, "max_damage_action_overrides": first_turn_max_damage_overrides, "private_opponent_action_oracle_enabled": bool(allow_privileged_opponent_action_oracle)}
            if trace_decisions:
                record.update({
                    "format": battle_format,
                    "opponent": opponent,
                    "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint else None,
                    "teams": {
                        side: [pokemon.get("speciesId", pokemon.get("species", pokemon.get("name", ""))) for pokemon in team]
                        for side, team in preview_teams.items()
                    },
                    "preview_actions": preview_actions,
                    "selected_teams": {
                        side: {
                            "leads": [preview_teams[side][index].get("speciesId", preview_teams[side][index].get("species", preview_teams[side][index].get("name", ""))) for index in decode_action(action).leads],
                            "back": [preview_teams[side][index].get("speciesId", preview_teams[side][index].get("species", preview_teams[side][index].get("name", ""))) for index in decode_action(action).back],
                            "out": [preview_teams[side][index].get("speciesId", preview_teams[side][index].get("species", preview_teams[side][index].get("name", ""))) for index in decode_action(action).out],
                        }
                        for side, action in preview_actions.items()
                    },
                    "phase_trace": phase_trace,
                })
            if player_view_output is not None and not timed_out and player_view_trajectory:
                terminal_rewards = {
                    env.possible_agents[0]: float(player_reward),
                    env.possible_agents[1]: float(-player_reward),
                }
                if player_view_trajectory[-1].get("reward_components") is None:
                    player_view_trajectory[-1]["reward_components"] = {}
                for agent, outcome in terminal_rewards.items():
                    player_view_trajectory[-1]["reward_components"].setdefault(
                        agent, {}
                    )["terminal_outcome"] = outcome
                episode_record = {
                    "episode": f"{battle_format}:{seed}:{game}",
                    "format": battle_format,
                    "opponent": opponent,
                    "teams": {"player1": team1_data, "player2": team2_data},
                    "agent_to_team": {
                        env.possible_agents[0]: "player1",
                        env.possible_agents[1]: "player2",
                    },
                    "trajectory": player_view_trajectory,
                    "rewards": terminal_rewards,
                    "terminated": {
                        env.possible_agents[0]: True,
                        env.possible_agents[1]: True,
                    },
                    "truncated": {
                        env.possible_agents[0]: False,
                        env.possible_agents[1]: False,
                    },
                    "steps": steps,
                    "action_adjustments": action_adjustments,
                    "max_damage_score_gate_overrides": max_damage_score_gate_overrides,
                    "first_turn_max_damage_overrides": first_turn_max_damage_overrides,
                    "timed_out": False,
                    "teams_differ": team1_data != team2_data,
                    "preview_actions": preview_actions,
                }
                with player_view_output.open("a", encoding="utf-8") as stream:
                    for view in project_episode_to_player_views(episode_record):
                        stream.write(json.dumps(view, sort_keys=True) + "\n")
                        player_view_records_written += 1
        except EpisodeTimeout as exc:
            timed_out = True
            timeouts += 1
            record = {"game": game, "team_seed": roster_seed, "result": "timeout", "error": str(exc),
                      "action_adjustments": action_adjustments,
                      "max_damage_score_gate_overrides": max_damage_score_gate_overrides,
                      "counterplay_prior_weight": counterplay_prior_weight}
            if trace_decisions:
                record.update({"format": battle_format, "opponent": opponent,
                               "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint else None,
                               "teams": {side: [pokemon.get("speciesId", pokemon.get("species", "")) for pokemon in team]
                                         for side, team in preview_teams.items()},
                               "preview_actions": preview_actions, "phase_trace": phase_trace})
        except Exception as exc:
            # Keep a bounded benchmark running when the server rejects a
            # battle or a previous websocket has not finished closing.
            timed_out = True
            timeouts += 1
            record = {"game": game, "team_seed": roster_seed, "result": "error", "error": str(exc),
                      "action_adjustments": action_adjustments,
                      "max_damage_score_gate_overrides": max_damage_score_gate_overrides,
                      "counterplay_prior_weight": counterplay_prior_weight}
            if trace_decisions:
                record.update({"format": battle_format, "opponent": opponent,
                               "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint else None,
                               "teams": {side: [pokemon.get("speciesId", pokemon.get("species", "")) for pokemon in team]
                                         for side, team in preview_teams.items()},
                               "preview_actions": preview_actions, "phase_trace": phase_trace})
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            try:
                env.close(force=timed_out, wait=not timed_out)
            except Exception:
                # A server-side validation failure can tear down one battle
                # before poke-env creates the second; preserve the record.
                pass
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    completed = wins + losses + draws
    rate = wins / completed if completed else 0.0
    margin = 1.96 * np.sqrt(rate * (1 - rate) / completed) if completed else 0.0
    return {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "games": games,
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": rate,
        "win_rate_ci95": [max(0.0, rate - margin), min(1.0, rate + margin)],
        "action_adjustments": sum(
            json.loads(line).get("action_adjustments", 0)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "max_damage_score_gate_overrides": sum(
            json.loads(line).get("max_damage_score_gate_overrides", 0)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "first_turn_max_damage_overrides": sum(
            json.loads(line).get("first_turn_max_damage_overrides", 0)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "max_damage_action_overrides": sum(
            json.loads(line).get("max_damage_action_overrides", 0)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "max_damage_score_gate_margin": max_damage_score_gate_margin,
        "private_opponent_action_oracle_enabled": bool(allow_privileged_opponent_action_oracle),
        "player_view_output": str(player_view_output) if player_view_output else None,
        "player_view_records_written": player_view_records_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-damage-score-gate", action="store_true",
                        help="replace policy actions when the max-damage immediate score is higher")
    parser.add_argument("--max-damage-score-gate-margin", type=float, default=0.0,
                        help="minimum relative immediate-damage advantage required for the gate to replace a policy action")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--team-seed", type=int,
        help="Optional independent seed for generated teams, allowing roster-matched evaluations.",
    )
    parser.add_argument("--team1-json", type=Path,
                        help="Fixed six-Pokémon JSON team for player 1 (array or object with pokemon/team list).")
    parser.add_argument("--team2-json", type=Path,
                        help="Fixed six-Pokémon JSON team for player 2 (array or object with pokemon/team list).")
    parser.add_argument("--preview-action-player1", type=int,
                        help="Force a team-preview action index (0-89) for player 1, for controlled tests.")
    parser.add_argument("--preview-action-player2", type=int,
                        help="Force a team-preview action index (0-89) for player 2, for controlled tests.")
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--episode-timeout", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using deterministic evaluation.")
    parser.add_argument(
        "--opponent", choices=("random", "max_damage", "policy", "counterplay", "public_counterplay", "no_guard_fissure", "perish_trap"),
        default="random",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--player", choices=("policy", "random", "max_damage", "counterplay", "public_counterplay"), default="policy")
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument(
        "--opponent-team-selection-checkpoint", type=Path,
        help="Optional selector for player 2 when comparing two team-selection policies.",
    )
    parser.add_argument("--team-selection-mode", choices=("random", "policy"), default="random")
    parser.add_argument("--team-selection-side", choices=("both", "player1"), default="both")
    parser.add_argument("--trace-decisions", action="store_true",
                        help="Record preview selections, teams, and per-step rewards for analysis.")
    parser.add_argument(
        "--player-view-output", type=Path,
        help="Optional JSONL path for closed-sheet per-player observation/outcome records, usable to train the state-value model.",
    )
    parser.add_argument("--counterplay-incoming-weight", type=float, default=1.0)
    parser.add_argument(
        "--counterplay-support-actions", action=argparse.BooleanOptionalAction,
        default=False,
        help="Score Helping Hand and useful Tailwind/Trick Room choices in counterplay mode.",
    )
    parser.add_argument("--counterplay-ko-bonus", type=float, default=0.7)
    parser.add_argument(
        "--first-turn-max-damage", action="store_true",
        help="use Max Damage for the evaluated policy's first legal joint action",
    )
    parser.add_argument(
        "--max-damage-action-steps", type=int, nargs="+",
        help="evaluated-policy decision indices to replace with Max Damage in every game",
    )
    parser.add_argument(
        "--allow-privileged-opponent-action-oracle", action="store_true",
        help="Enable legacy opponent-action prediction from private battle views (diagnostic only).",
    )
    parser.add_argument("--counterplay-prior-weight", type=float,
                        help="Override the checkpoint's counterplay score prior weight.")
    parser.add_argument(
        "--public-counterplay-prior-weight", type=float, default=0.0,
        help="Add a public-view-only Counterplay action-score prior to policy logits.",
    )
    parser.add_argument(
        "--public-counterplay-prior-min-margin", type=float, default=0.0,
        help="Apply the public prior only when its top action exceeds second place by this many normalized score units.",
    )
    parser.add_argument("--public-counterplay-switch-penalty", type=float, default=0.05)
    parser.add_argument("--public-counterplay-support-actions", action="store_true")
    parser.add_argument("--public-counterplay-redirection", action="store_true")
    parser.add_argument("--public-counterplay-joint-outgoing", action="store_true")
    parser.add_argument("--public-counterplay-sequential-damage", action="store_true")
    parser.add_argument(
        "--public-counterplay-move-objective", choices=("damage", "hp_fraction"),
        default="damage",
    )
    parser.add_argument(
        "--counterplay-predict-policy-opponent", action="store_true",
        help="Condition player1 counterplay scores and action features on the frozen policy's predicted move.",
    )
    parser.add_argument(
        "--counterplay-respect-protect-streaks", action=argparse.BooleanOptionalAction,
        default=True,
        help="Account for the reduced success chance of consecutive Protect-like moves.",
    )
    parser.add_argument(
        "--counterplay-cap-overkill", action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap joint damage at each target's remaining HP and award each KO once.",
    )
    args = parser.parse_args()
    options = vars(args)
    options["deterministic"] = not options.pop("stochastic")
    print(json.dumps(evaluate(**options), sort_keys=True))


if __name__ == "__main__":
    main()
