"""Run native poke-env self-play battles against a local/remote Showdown server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from poke_env.ps_client import ServerConfiguration
from poke_env.teambuilder import Teambuilder
from websockets.exceptions import WebSocketException

from src.env.showdown_env import ShowdownDoublesEnv
from src.env.poke_env_cleanup import close_poke_env_nonblocking
from src.baselines.native_damage import (
    choose_counterplay_action,
    choose_max_damage_action,
)
from src.baselines.gimmick_opponents import (
    choose_no_guard_fissure_action, choose_perish_trap_action,
)
from src.training.observability import TrainingTelemetry
from src.training.native_ppo import NATIVE_ACTION_FEATURE_ENCODER_VERSION
from src.training.perspective_data import project_episode_to_player_views
from src.models.policy import TransformerPolicy
from src.training.self_play import PolicyAgent
from src.training.team_selection import (
    ACTION_COUNT, TeamSelectionPolicy, decode_action, encode_action, preview_command,
    random_action, team_features,
)


class EpisodeTimeout(RuntimeError):
    """Raised when a native battle exceeds its batch time budget."""


LOCAL_SERVER_CONFIGURATION = ServerConfiguration(
    websocket_url=os.environ.get(
        "POKEMON_SHOWDOWN_WEBSOCKET_URL",
        "ws://localhost:8000/showdown/websocket",
    ),
    authentication_url=os.environ.get(
        "POKEMON_SHOWDOWN_AUTH_URL",
        "http://localhost:8000/action.php?",
    ),
)


def _raise_episode_timeout(_signum: int, _frame: Any) -> None:
    raise EpisodeTimeout("native battle exceeded episode timeout")


def _raise_episode_sigterm(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt("native battle interrupted by SIGTERM")


def team_to_showdown_text(team: List[Dict[str, Any]]) -> str:
    """Convert the repository's team dictionaries to Showdown text format."""
    entries: List[str] = []
    for pokemon in team[:6]:
        lines = [str(pokemon.get("species") or pokemon.get("name", "Ditto"))]
        if pokemon.get("item"):
            lines[0] += f" @ {pokemon['item']}"
        if pokemon.get("ability"):
            ability = str(pokemon["ability"]).lower().replace("unseen-hand", "Unseen Fist")
            lines.append(f"Ability: {ability}")
        if pokemon.get("level"):
            lines.append(f"Level: {pokemon['level']}")
        if pokemon.get("evs"):
            evs = pokemon["evs"]
            ev_text = " / ".join(f"{value} {stat.upper()}" for stat, value in evs.items() if value)
            if ev_text:
                lines.append(f"EVs: {ev_text}")
        if pokemon.get("nature"):
            lines.append(f"{pokemon['nature']} Nature")
        if pokemon.get("ivs"):
            ivs = pokemon["ivs"]
            iv_text = " / ".join(f"{value} {stat.upper()}" for stat, value in ivs.items() if value != 31)
            if iv_text:
                lines.append(f"IVs: {iv_text}")
        for move in pokemon.get("moves", [])[:4]:
            lines.append(f"- {str(move).lower().replace('protected', 'Protect')}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def generate_showdown_champions_team(seed: int, format_id: str) -> str:
    """Use Showdown's RandomChampionsTeams builder and return packed text."""
    showdown_path = os.environ.get("POKEMON_SHOWDOWN_PATH")
    if not showdown_path:
        raise RuntimeError(
            "POKEMON_SHOWDOWN_PATH must point to the Pokemon Showdown checkout"
        )
    result = subprocess.run(
        ["node", "scripts/showdown_random_team.js", str(seed), format_id],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "POKEMON_SHOWDOWN_PATH": showdown_path},
    )
    return team_to_showdown_text(json.loads(result.stdout))


def generate_showdown_champions_team_data(
    seed: int, format_id: str, archetype: str | None = None
) -> List[Dict[str, Any]]:
    """Generate the raw set dictionaries for policy encoding."""
    showdown_path = os.environ.get("POKEMON_SHOWDOWN_PATH")
    if not showdown_path:
        # Keep local evaluation usable when only the standalone server is
        # available.  The checked-in corpus contains legal six-Pokemon teams
        # and is deterministic, so it is a practical fallback for benchmarks.
        corpus = Path("data/teams/champions_mb.json")
        if format_id == "gen9championsvgc2026regmb" and corpus.exists():
            teams = json.loads(corpus.read_text(encoding="utf-8"))
            if teams:
                return teams[seed % len(teams)]["pokemon"][:6]
        raise RuntimeError(
            "POKEMON_SHOWDOWN_PATH must point to a Pokemon Showdown checkout with "
            f"format {format_id}; no cross-regulation team fallback is allowed"
        )
    command = ["node", "scripts/showdown_random_team.js", str(seed), format_id]
    if archetype:
        command.append(archetype)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "POKEMON_SHOWDOWN_PATH": showdown_path},
    )
    return json.loads(result.stdout)


def shuffle_archetype_team_slots(
    team: List[Dict[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    """Return a deterministic slot permutation without changing team sets."""
    order = np.random.default_rng(seed).permutation(len(team))
    return [team[int(index)] for index in order]


def archetype_preview_action(team: List[Dict[str, Any]], archetype: str) -> int:
    """Select an archetype's core and preferred leads by species, not slot.

    Gimmick teams are shuffled during rollouts and evaluation so the learner
    cannot exploit the builder's original first-slots convention.
    """
    priorities = {
        "no_guard_fissure": ("machamp", "gardevoir", "snorlax"),
        "perish_trap": ("gengar", "primarina"),
    }
    if archetype not in priorities:
        raise ValueError(f"no archetype preview policy for {archetype!r}")

    def species_key(pokemon: Dict[str, Any]) -> str:
        raw = pokemon.get("speciesId") or pokemon.get("species") or pokemon.get("name", "")
        return "".join(character.lower() for character in str(raw) if character.isalnum())

    index_by_species = {species_key(pokemon): index for index, pokemon in enumerate(team)}
    core = priorities[archetype]
    missing = [species for species in core if species not in index_by_species]
    if missing:
        raise ValueError(f"{archetype} team is missing core species: {missing}")
    core_indices = [index_by_species[species] for species in core]
    selected = core_indices + [
        index for index in range(len(team)) if index not in core_indices
    ][: 4 - len(core_indices)]
    leads = core_indices[:2]
    return encode_action(selected, leads)


def pack_showdown_team(team_text: str) -> str:
    """Convert Showdown text into the packed representation expected by poke-env."""
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(team_text))


def install_atomic_team_preview(env: ShowdownDoublesEnv, commands: Dict[str, str]) -> None:
    """Install one-shot ``/team`` handlers, avoiding sequential preview steps."""
    for agent_name, command in commands.items():
        player = env.agent1 if agent_name == env.possible_agents[0] else env.agent2
        def handler(battle: Any, command: str = command) -> str:
            # The environment's default VGC handler waits for two native
            # switch actions.  A preview decision is one atomic Showdown
            # command, so return it directly.  Do not mutate Pokemon state:
            # Showdown's request/update pair is authoritative and poke-env
            # updates selected flags while parsing that update.
            return command
        player.teampreview = handler


def choose_masked_actions(
    observations: Dict[str, Dict[str, Any]], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Choose one legal native poke-env action per active Pokémon."""
    actions: Dict[str, np.ndarray] = {}
    for agent, payload in observations.items():
        if payload.get("wait"):
            actions[agent] = np.asarray([0, 0], dtype=np.int64)
            continue
        mask = np.asarray(payload["action_mask"], dtype=np.int8)
        if mask.size % 2:
            raise ValueError(f"Expected two individual action masks for {agent}, got {mask.size}")
        half = mask.size // 2
        individual = []
        for position, position_mask in enumerate((mask[:half], mask[half:])):
            legal = np.flatnonzero(position_mask)
            if legal.size == 0:
                raise RuntimeError(
                    f"No legal action available for {agent}; wait={payload.get('wait')}; "
                    f"empty_position={position}; "
                    f"active_request_summary={payload.get('active_request_summary')}; "
                    f"legal_indices_by_position={[np.flatnonzero(mask[i * half:(i + 1) * half]).tolist() for i in range(2)]}"
                )
            choices = legal
            if position == 1 and individual:
                # During team preview and forced switches, submitting the same
                # slot twice is invalid in native doubles.
                distinct = legal[legal != individual[0]]
                if distinct.size:
                    choices = distinct
            individual.append(int(rng.choice(choices)))
        actions[agent] = sanitize_native_action(mask, np.asarray(individual, dtype=np.int64))[0]
    return actions


def sanitize_native_action(
    action_mask: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, int]:
    """Make a doubles action legal and report any substitutions."""
    mask = np.asarray(action_mask, dtype=np.int8)
    selected = np.asarray(action, dtype=np.int64).copy()
    if mask.ndim != 1 or mask.size % 2 or selected.shape != (2,):
        raise ValueError("native action and mask must describe two slots")
    half = mask.size // 2
    adjustments = 0
    for slot in range(2):
        legal = np.flatnonzero(mask[slot * half:(slot + 1) * half])
        if not legal.size:
            raise RuntimeError("native action mask has no legal action")
        if selected[slot] < 0 or selected[slot] >= half or not mask[slot * half + selected[slot]]:
            selected[slot] = int(legal[0])
            adjustments += 1
    if selected[0] == selected[1] and selected[0] in range(1, 7):
        alternatives = np.flatnonzero(mask[half:])
        alternatives = alternatives[alternatives != selected[0]]
        if alternatives.size:
            selected[1] = int(alternatives[0])
            adjustments += 1
    if selected[0] == selected[1] == 0:
        alternatives = np.flatnonzero(mask[half:])
        alternatives = alternatives[alternatives != 0]
        if alternatives.size:
            selected[1] = int(alternatives[0])
            adjustments += 1
    return selected, adjustments


def current_observations(
    env: ShowdownDoublesEnv, *, force_team_preview: bool = False
) -> Dict[str, Dict[str, Any]]:
    if env.battle1 is None or env.battle2 is None:
        raise RuntimeError("DoublesEnv has no active battles")

    def synchronized_mask(battle: Any) -> np.ndarray:
        mask = np.asarray(env.get_action_mask(battle), dtype=np.int8)
        if not force_team_preview and not battle.teampreview:
            return mask
        # During preview (and the short transition immediately after it),
        # selected flags and active Pokemon can lag valid orders by one async
        # callback. Build switch masks from those orders instead.
        preview_mask = np.zeros_like(mask)
        species = [pokemon.base_species for pokemon in battle.team.values()]
        for position, orders in enumerate(battle.valid_orders):
            offset = position * 126
            for order in orders:
                pokemon = getattr(order, "order", None)
                name = getattr(pokemon, "base_species", None)
                if name in species:
                    preview_mask[offset + species.index(name) + 1] = 1
        return preview_mask

    result = {
        env.agents[0]: {
            "observation": env.embed_battle(env.battle1),
            "action_mask": synchronized_mask(env.battle1),
        },
        env.agents[1]: {
            "observation": env.embed_battle(env.battle2),
            "action_mask": synchronized_mask(env.battle2),
        },
    }
    # poke-env updates active Pokémon and valid orders in separate websocket
    # callbacks. Require two identical snapshots before acting, otherwise an
    # old switch list can be paired with a new battle state.
    signature = tuple(
        (
            tuple(getattr(p, "base_species", None) if p is not None else None
                  for p in battle.active_pokemon),
            tuple(
                tuple(str(order)
                      for order in orders)
                for orders in battle.valid_orders
            ),
            bool(battle.teampreview),
            bool(getattr(battle, "_wait", False)),
        )
        for battle in (env.battle1, env.battle2)
    )
    previous = getattr(env, "_stable_observation_signature", None)
    env._stable_observation_signature = signature
    if previous != signature:
        for payload in result.values():
            payload["action_mask"] = np.zeros_like(payload["action_mask"])
    return result


def team_preview_ready(env: ShowdownDoublesEnv) -> bool:
    """Return whether both sides have synchronized preview orders available."""
    for battle in (env.battle1, env.battle2):
        if battle is None:
            return False
        if battle.teampreview or any(pokemon is None for pokemon in battle.active_pokemon):
            for orders in battle.valid_orders:
                if not orders or not all(
                    hasattr(getattr(order, "order", None), "base_species")
                    for order in orders
                ):
                    return False
    return True


def sanitize_team_preview_actions(
    actions: Dict[str, np.ndarray], env: ShowdownDoublesEnv
) -> Dict[str, np.ndarray]:
    """Keep preview orders in the native switch-index space during async setup."""
    sanitized: Dict[str, np.ndarray] = {}
    for agent, action in actions.items():
        battle = env.battle1 if agent == env.agents[0] else env.battle2
        if battle is None:
            sanitized[agent] = action
            continue
        choices = [
            index
            for index, pokemon in enumerate(battle.team.values(), start=1)
            if not pokemon.selected_in_teampreview
        ]
        if len(choices) < 2:
            choices = list(range(1, len(battle.team) + 1))
        selected = np.asarray(action, dtype=np.int64).copy()
        selected[0] = choices[0]
        selected[1] = choices[1]
        sanitized[agent] = selected
    return sanitized


def choose_current_actions(env: ShowdownDoublesEnv, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Choose actions from the battles used by the next native step.

    The observation returned by ``DoublesEnv.step`` can lag the server's
    action space while simultaneous switch/team-preview updates are queued.
    Reading the masks from the environment's current battle objects keeps the
    action index and order list synchronized.
    """
    return choose_masked_actions(current_observations(env), rng)


def run_self_play(
    *,
    episodes: int,
    output_path: Path,
    battle_format: str,
    seed: int,
    team_seed: int | None = None,
    team1_json: Path | None = None,
    team2_json: Path | None = None,
    preview_action_player1: int | None = None,
    preview_action_player2: int | None = None,
    server_configuration: ServerConfiguration | None = None,
    episode_timeout: float = 120.0,
    challenge_timeout: float = 60.0,
    policy_checkpoint: Path | None = None,
    opponent_checkpoint: Path | None = None,
    opponent_checkpoints: List[Path] | None = None,
    device: str = "cpu",
    policy_temperature: float = 1.0,
    telemetry_path: Path | None = None,
    run_id: str = "native-rollouts",
    opponent_tag: str | None = None,
    opponent: str = "policy",
    player_strategy: str | None = None,
    opponent_schedule: List[str] | None = None,
    opponent_policy_deterministic: bool = False,
    team_preview: bool = False,
    team_selection_checkpoint: Path | None = None,
    team_selection_mode: str = "random",
    team_selection_temperature: float = 0.0,
    collect_max_damage_labels: bool = False,
    collect_counterplay_labels: bool = False,
    counterplay_incoming_weight: float = 1.0,
    counterplay_ko_bonus: float = 0.7,
    counterplay_speed_control_scale: float = 1.0,
    counterplay_support_actions: bool = False,
    counterplay_cap_overkill: bool = True,
    team_selection_side: str = "both",
    counterplay_prior_weight: float | None = None,
    counterplay_predict_policy_opponent: bool = False,
) -> None:
    """Run native Showdown battles and append one safe view per player."""
    if not np.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be finite and positive")
    if not np.isfinite(episode_timeout) or episode_timeout <= 0:
        raise ValueError("episode_timeout must be finite and positive")
    if not np.isfinite(challenge_timeout) or challenge_timeout <= 0:
        raise ValueError("challenge_timeout must be finite and positive")
    if not np.isfinite(team_selection_temperature) or team_selection_temperature < 0:
        raise ValueError("team_selection_temperature must be finite and nonnegative")
    if counterplay_prior_weight is not None and (
        not np.isfinite(counterplay_prior_weight) or counterplay_prior_weight < 0
    ):
        raise ValueError("counterplay_prior_weight must be finite and nonnegative")
    if player_strategy is None:
        player_strategy = "policy" if policy_checkpoint is not None else "random"
    if player_strategy not in {"policy", "counterplay", "random"}:
        raise ValueError(f"invalid player strategy: {player_strategy}")
    if player_strategy == "policy" and policy_checkpoint is None:
        raise ValueError("policy_checkpoint is required when player_strategy='policy'")
    opponents = opponent_schedule or [opponent]
    invalid_opponents = set(opponents) - {
        "policy", "random", "max_damage", "counterplay", "no_guard_fissure", "perish_trap"
    }
    if not opponents or invalid_opponents:
        raise ValueError(f"invalid opponent schedule: {sorted(invalid_opponents)}")
    rng = np.random.default_rng(seed)
    roster_seed = seed if team_seed is None else team_seed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_state = None
    checkpoint_model_config: Dict[str, Any] = {}
    policy_model = None
    opponent_model = None
    frozen_opponent_paths = list(opponent_checkpoints or [])
    if opponent_checkpoint is not None and opponent_checkpoint not in frozen_opponent_paths:
        frozen_opponent_paths.insert(0, opponent_checkpoint)
    frozen_opponent_models: Dict[str, TransformerPolicy] = {}
    frozen_opponent_prior_weights: Dict[str, float] = {}
    if policy_checkpoint is not None:
        checkpoint_state = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
        if counterplay_prior_weight is None:
            counterplay_prior_weight = float(
                checkpoint_state.get("counterplay_prior_weight", 0.0)
            )
        policy_model = TransformerPolicy.from_checkpoint_state(checkpoint_state)
        policy_model.load_state_dict(checkpoint_state["model_state_dict"], strict=False)
        policy_model.to(device).eval()
        checkpoint_model_config = policy_model.export_config()
    for frozen_path in frozen_opponent_paths:
        opponent_state = torch.load(frozen_path, map_location="cpu", weights_only=True)
        opponent_model = TransformerPolicy.from_checkpoint_state(opponent_state)
        opponent_model.load_state_dict(opponent_state["model_state_dict"], strict=False)
        opponent_model.to(device).eval()
        frozen_opponent_models[str(frozen_path)] = opponent_model
        frozen_opponent_prior_weights[str(frozen_path)] = float(
            opponent_state.get("counterplay_prior_weight", 0.0)
        )
    if counterplay_prior_weight is None:
        counterplay_prior_weight = 0.0
    selection_policy = None
    if team_selection_checkpoint is not None:
        state = torch.load(team_selection_checkpoint, map_location="cpu", weights_only=True)
        selection_policy = TeamSelectionPolicy(**state["model_config"])
        selection_policy.load_state_dict(state["model_state_dict"])
        selection_policy.eval()
    telemetry = (
        TrainingTelemetry(telemetry_path, run_id)
        if telemetry_path is not None
        else None
    )
    if telemetry is not None:
        telemetry.start(
            phase="rollout",
            format=battle_format,
            episodes=episodes,
            opponent_schedule=opponents,
            policy_checkpoint=str(policy_checkpoint) if policy_checkpoint else None,
            opponent_checkpoints=[str(path) for path in frozen_opponent_paths],
            native_action_feature_variant=checkpoint_model_config.get("native_action_feature_variant")
            if policy_checkpoint is not None
            else None,
            native_action_feature_dim=checkpoint_model_config.get("native_action_feature_dim")
            if policy_checkpoint is not None
            else 0,
            policy_temperature=policy_temperature,
            seed=seed,
            opponent_tag=opponent_tag,
        )

    # Reuse the same poke-env clients and their asyncio loop across episodes.
    # DoublesEnv starts a daemon event-loop thread in __init__ and close()
    # resets battles but does not stop that loop. Constructing a fresh env for
    # every game leaks threads and eventually stalls long rollouts.
    env: ShowdownDoublesEnv | None = None
    policy_opponent_count = 0
    def load_fixed_team(path: Path | None) -> List[Dict[str, Any]] | None:
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
    for episode in range(episodes):
        episode_opponent = opponents[episode % len(opponents)]
        episode_opponent_checkpoint = None
        if episode_opponent == "policy" and frozen_opponent_paths:
            episode_opponent_checkpoint = frozen_opponent_paths[
                policy_opponent_count % len(frozen_opponent_paths)
            ]
            policy_opponent_count += 1
        team1_data = fixed_team1 or generate_showdown_champions_team_data(roster_seed + episode * 2, battle_format)
        team2_data = fixed_team2 or generate_showdown_champions_team_data(
            roster_seed + episode * 2 + 1,
            battle_format,
            archetype=(
                episode_opponent
                if episode_opponent in {"no_guard_fissure", "perish_trap"}
                else None
            ),
        )
        retry = 0
        while team1_data == team2_data and retry < 10:
            retry += 1
            team2_data = generate_showdown_champions_team_data(
                roster_seed + episode * 2 + 1 + retry * 100_000,
                battle_format,
                archetype=(
                    episode_opponent
                    if episode_opponent in {"no_guard_fissure", "perish_trap"}
                    else None
                ),
            )
        if fixed_team2 is None and episode_opponent in {"no_guard_fissure", "perish_trap"}:
            team2_data = shuffle_archetype_team_slots(
                team2_data, roster_seed + episode * 2 + 1 + 7_000_000
            )
        team1_text = team_to_showdown_text(team1_data)
        team2_text = team_to_showdown_text(team2_data)
        team1 = pack_showdown_team(team1_text)
        team2 = pack_showdown_team(team2_text)
        if team1 == team2:
            raise RuntimeError(f"Showdown generated identical teams for episode {episode}")
        env_options: Dict[str, Any] = {
            "battle_format": battle_format,
            "server_configuration": server_configuration or LOCAL_SERVER_CONFIGURATION,
            "team": team1,
            # Keep the agent's information set closed even if the selected
            # Showdown format offers open-team-sheet consent.
            "accept_open_team_sheet": False,
            "start_listening": True,
            "choose_on_teampreview": False,
            "fake": False,
            "strict": False,
            "challenge_timeout": challenge_timeout,
        }
        env_options["choose_on_teampreview"] = team_preview
        if env is None:
            env = ShowdownDoublesEnv(**env_options)
        else:
            env.agent1.update_team(team1)
            env.agent2.update_team(team2)
        agent_to_team = {
            env.possible_agents[0]: "player1",
            env.possible_agents[1]: "player2",
        }
        preview_actions: Dict[str, int] = {}
        preview_log_probs: Dict[str, float] = {}
        preview_rngs = {
            "player1": np.random.default_rng(seed + episode * 2),
            "player2": np.random.default_rng(seed + episode * 2 + 1),
        }
        if team_preview:
            commands = {}
            for name, data, opponent_data in (
                (env.possible_agents[0], team1_data, team2_data),
                (env.possible_agents[1], team2_data, team1_data),
            ):
                team_name = agent_to_team[name]
                fixed_preview_action = (
                    preview_action_player1 if team_name == "player1" else preview_action_player2
                )
                if (team_name == "player2" and fixed_team2 is None and fixed_preview_action is None
                        and episode_opponent in {"no_guard_fissure", "perish_trap"}):
                    fixed_preview_action = archetype_preview_action(data, episode_opponent)
                if fixed_preview_action is not None:
                    if not 0 <= fixed_preview_action < ACTION_COUNT:
                        raise ValueError(f"preview action for {team_name} must be in [0, {ACTION_COUNT})")
                    action = fixed_preview_action
                elif (selection_policy is not None and team_selection_mode == "policy"
                        and (team_selection_side == "both" or agent_to_team[name] == "player1")):
                    with torch.no_grad():
                        logits = selection_policy(
                            team_features(data, selection_policy.feature_dim).unsqueeze(0),
                            opponent_features=team_features(
                                opponent_data, selection_policy.feature_dim,
                                revealed_only=True,
                            ).unsqueeze(0)
                            if selection_policy.opponent_conditioned else None,
                        )
                    if team_selection_temperature > 0:
                        action_log_probs = torch.log_softmax(
                            logits / team_selection_temperature, dim=-1
                        )
                        action_probs = action_log_probs.exp().squeeze(0).cpu().numpy()
                        action = int(preview_rngs[team_name].choice(
                            len(action_probs), p=action_probs
                        ))
                        preview_log_probs[team_name] = float(
                            action_log_probs[0, action].item()
                        )
                    else:
                        action = int(logits.argmax(dim=-1).item())
                else:
                    action = random_action(rng)
                preview_actions[agent_to_team[name]] = action
                commands[name] = preview_command(action)
            install_atomic_team_preview(env, commands)
        policies = None
        if policy_model is not None and player_strategy in {"policy", "counterplay"}:
            policies = {
                env.possible_agents[0]: PolicyAgent(
                    policy_model, team1_data, team2_data, device=device,
                    temperature=policy_temperature,
                ),
            }
            if episode_opponent == "policy":
                episode_opponent_model = frozen_opponent_models.get(
                    str(episode_opponent_checkpoint), policy_model
                )
                policies[env.possible_agents[1]] = PolicyAgent(
                    episode_opponent_model,
                    team2_data, team1_data, device=device,
                    deterministic=opponent_policy_deterministic,
                    temperature=policy_temperature,
                )
        episode_opponent_prior_weight = frozen_opponent_prior_weights.get(
            str(episode_opponent_checkpoint), float(counterplay_prior_weight)
        )
        steps = 0
        action_adjustments = 0
        timed_out = False
        episode_started_at = time.perf_counter()
        observation_seconds = 0.0
        action_selection_seconds = 0.0
        showdown_step_seconds = 0.0
        trajectory: List[Dict[str, Any]] = []
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        try:
                signal.signal(signal.SIGTERM, _raise_episode_sigterm)
                signal.signal(signal.SIGALRM, _raise_episode_timeout)
                signal.setitimer(signal.ITIMER_REAL, float(episode_timeout))
                observations, _ = env.reset(seed=seed + episode)
                # Teampreview callbacks submit atomic /team commands directly.
                # Do not send native switch actions during this phase: doing so
                # races the server command and corrupts the post-preview index
                # mapping.
                if team_preview:
                    while any(
                        battle is None
                        or battle.teampreview
                        or any(pokemon is None for pokemon in battle.active_pokemon)
                        for battle in (env.battle1, env.battle2)
                    ):
                        time.sleep(0.01)
                preview_orders_submitted = 2
                while env.agents:
                    if team_preview and preview_orders_submitted < 2:
                        if preview_orders_submitted < 2:
                            if not team_preview_ready(env):
                                time.sleep(0.01)
                                continue
                        elif any(
                            pokemon is None
                            for battle in (env.battle1, env.battle2)
                            if battle is not None
                            for pokemon in battle.active_pokemon
                        ):
                            time.sleep(0.01)
                            continue
                    observation_started_at = time.perf_counter()
                    step_observations = current_observations(
                        env,
                        force_team_preview=team_preview and preview_orders_submitted < 2,
                    )
                    observation_seconds += time.perf_counter() - observation_started_at
                    # Showdown briefly exposes an empty mask while preview
                    # orders are being applied.  Waiting here avoids sending
                    # an incomplete action dictionary (and the resulting
                    # ``No legal action`` failure).
                    if any(
                        any(
                            not part.any()
                            for part in np.array_split(
                                np.asarray(payload["action_mask"], dtype=np.int8), 2
                            )
                        )
                        for payload in step_observations.values()
                    ):
                        time.sleep(0.01)
                        continue
                    action_started_at = time.perf_counter()
                    policy_outputs = {}
                    max_damage_prediction = None
                    player1_max_damage_prediction = None
                    opponent_agent = env.possible_agents[1]
                    if (player_strategy in {"policy", "counterplay"}
                            and env.agent2_to_move):
                        if episode_opponent == "no_guard_fissure":
                            max_damage_prediction = choose_no_guard_fissure_action(
                                step_observations[opponent_agent]["action_mask"],
                                env.battle2,
                            )
                        elif episode_opponent == "perish_trap":
                            max_damage_prediction = choose_perish_trap_action(
                                step_observations[opponent_agent]["action_mask"],
                                env.battle2,
                            )
                        else:
                            max_damage_prediction = choose_max_damage_action(
                                step_observations[opponent_agent]["action_mask"], env.battle2
                            )
                    if (player_strategy in {"policy", "counterplay"}
                            and episode_opponent == "policy" and env.agent1_to_move):
                        player_agent = env.possible_agents[0]
                        player1_max_damage_prediction = choose_max_damage_action(
                            step_observations[player_agent]["action_mask"], env.battle1
                        )
                    policy_opponent_prediction = None
                    opponent_scores = None
                    opponent_teacher_first = None
                    if (counterplay_predict_policy_opponent
                            and episode_opponent == "policy" and env.agent2_to_move
                            and opponent_agent in policies):
                        opponent_payload = step_observations[opponent_agent]
                        if episode_opponent_prior_weight > 0:
                            opponent_teacher_action, opponent_scores, _ = choose_counterplay_action(
                                opponent_payload["action_mask"], env.battle2, env.battle1,
                                player1_max_damage_prediction,
                                incoming_weight=counterplay_incoming_weight,
                                support_actions=counterplay_support_actions,
                                ko_bonus=counterplay_ko_bonus,
                                speed_control_scale=counterplay_speed_control_scale,
                                cap_overkill=counterplay_cap_overkill,
                                return_action_scores=True,
                            )
                            opponent_teacher_first = int(opponent_teacher_action[0])
                        # Predict the frozen policy's actual joint action once and
                        # reuse it below. Player 1's counterplay prior and action
                        # features should be conditioned on the policy opponent,
                        # not on a max-damage stand-in.
                        policy_outputs[opponent_agent] = policies[opponent_agent].predict_native_with_value(
                            opponent_payload["observation"],
                            opponent_payload["action_mask"],
                            predicted_opponent_action=player1_max_damage_prediction,
                            counterplay_action_scores=opponent_scores,
                            counterplay_prior_weight=episode_opponent_prior_weight,
                            counterplay_teacher_first_action=opponent_teacher_first,
                            use_critic=True,
                        )
                        policy_opponent_prediction = policy_outputs[opponent_agent][0]
                    if (episode_opponent == "counterplay" and env.agent1_to_move
                            and env.agent2_to_move and player1_max_damage_prediction is not None):
                        opponent_payload = step_observations[opponent_agent]
                        opponent_mask = np.asarray(opponent_payload["action_mask"], dtype=np.int8)
                        if all(part.any() for part in np.array_split(opponent_mask, 2)):
                            try:
                                policy_opponent_prediction = choose_counterplay_action(
                                    opponent_mask,
                                    env.battle2,
                                    env.battle1,
                                    player1_max_damage_prediction,
                                    incoming_weight=counterplay_incoming_weight,
                                    support_actions=counterplay_support_actions,
                                    ko_bonus=counterplay_ko_bonus,
                                    speed_control_scale=counterplay_speed_control_scale,
                                    cap_overkill=counterplay_cap_overkill,
                                )
                            except (ValueError, IndexError):
                                # Fresh individual masks can precede the other
                                # room's jointly compatible switch update.
                                policy_opponent_prediction = None
                    counterplay_reference_action = (
                        policy_opponent_prediction
                        if policy_opponent_prediction is not None
                        and (counterplay_predict_policy_opponent
                             and episode_opponent == "policy"
                             or episode_opponent == "counterplay")
                        else max_damage_prediction
                    )
                    counterplay_policy_scores = None
                    counterplay_policy_first_action = None
                    if (counterplay_prior_weight > 0 and policies is not None
                            and player_strategy == "policy" and env.agent1_to_move):
                        player_agent = env.possible_agents[0]
                        teacher_action, counterplay_policy_scores, _ = choose_counterplay_action(
                            step_observations[player_agent]["action_mask"],
                            env.battle1,
                            env.battle2,
                            policy_opponent_prediction
                            if policy_opponent_prediction is not None
                            else max_damage_prediction,
                            incoming_weight=counterplay_incoming_weight,
                            support_actions=counterplay_support_actions,
                            ko_bonus=counterplay_ko_bonus,
                            speed_control_scale=counterplay_speed_control_scale,
                            cap_overkill=counterplay_cap_overkill,
                            return_action_scores=True,
                        )
                        counterplay_policy_first_action = int(teacher_action[0])
                    if policies is None:
                        # Use the exact masks sampled above; taking a second
                        # asynchronous snapshot can reintroduce a stale mask.
                        actions = choose_masked_actions(step_observations, rng)
                        if player_strategy == "counterplay" and env.agent1_to_move:
                            player_agent = env.possible_agents[0]
                            actions[player_agent] = choose_counterplay_action(
                                step_observations[player_agent]["action_mask"],
                                env.battle1,
                                env.battle2,
                                max_damage_prediction,
                                incoming_weight=counterplay_incoming_weight,
                                support_actions=counterplay_support_actions,
                                ko_bonus=counterplay_ko_bonus,
                                speed_control_scale=counterplay_speed_control_scale,
                                cap_overkill=counterplay_cap_overkill,
                            )
                        if (episode_opponent == "max_damage" and env.agent2_to_move
                                and max_damage_prediction is not None):
                            actions[opponent_agent] = max_damage_prediction
                        elif episode_opponent == "no_guard_fissure" and env.agent2_to_move:
                            actions[opponent_agent] = choose_no_guard_fissure_action(
                                step_observations[opponent_agent]["action_mask"], env.battle2
                            )
                        elif episode_opponent == "perish_trap" and env.agent2_to_move:
                            actions[opponent_agent] = choose_perish_trap_action(
                                step_observations[opponent_agent]["action_mask"], env.battle2
                            )
                        policy_details: Dict[str, Tuple[float, float]] = {}
                    else:
                        for agent, payload in step_observations.items():
                            if agent not in policies:
                                continue
                            if agent in policy_outputs:
                                continue
                            is_player1 = agent == env.possible_agents[0]
                            predicted_action = (
                                (policy_opponent_prediction
                                 if policy_opponent_prediction is not None
                                 else max_damage_prediction)
                                if is_player1 else player1_max_damage_prediction
                            )
                            if (not is_player1 and episode_opponent == "policy"
                                    and episode_opponent_prior_weight > 0
                                    and env.agent2_to_move):
                                opponent_teacher_action, opponent_scores, _ = choose_counterplay_action(
                                    payload["action_mask"], env.battle2, env.battle1,
                                    player1_max_damage_prediction,
                                    incoming_weight=counterplay_incoming_weight,
                                    support_actions=counterplay_support_actions,
                                    ko_bonus=counterplay_ko_bonus,
                                    speed_control_scale=counterplay_speed_control_scale,
                                    cap_overkill=counterplay_cap_overkill,
                                    return_action_scores=True,
                                )
                                opponent_teacher_first = int(opponent_teacher_action[0])
                            policy_outputs[agent] = policies[agent].predict_native_with_value(
                                payload["observation"],
                                payload["action_mask"],
                                predicted_opponent_action=predicted_action,
                                counterplay_action_scores=(
                                    counterplay_policy_scores if is_player1 else opponent_scores
                                ),
                                counterplay_prior_weight=(
                                    counterplay_prior_weight if is_player1
                                    else episode_opponent_prior_weight
                                ),
                                counterplay_teacher_first_action=(
                                    counterplay_policy_first_action
                                    if is_player1 else opponent_teacher_first
                                ),
                                use_critic=True,
                            )
                        actions = {}
                        for agent, output in policy_outputs.items():
                            actions[agent], adjusted = sanitize_native_action(
                                step_observations[agent]["action_mask"], output[0]
                            )
                            action_adjustments += adjusted
                        random_observations = {
                            agent: payload for agent, payload in step_observations.items()
                            if agent not in policies
                        }
                        if episode_opponent == "max_damage" and env.agent2_to_move:
                            if max_damage_prediction is None:
                                max_damage_prediction = choose_max_damage_action(
                                    step_observations[opponent_agent]["action_mask"],
                                    env.battle2,
                                )
                            actions[opponent_agent] = max_damage_prediction
                        elif episode_opponent == "no_guard_fissure" and env.agent2_to_move:
                            actions[opponent_agent] = choose_no_guard_fissure_action(
                                step_observations[opponent_agent]["action_mask"], env.battle2
                            )
                        elif episode_opponent == "perish_trap" and env.agent2_to_move:
                            actions[opponent_agent] = choose_perish_trap_action(
                                step_observations[opponent_agent]["action_mask"], env.battle2
                            )
                        elif episode_opponent == "counterplay" and env.agent2_to_move:
                            actions[opponent_agent] = (
                                policy_opponent_prediction
                                if policy_opponent_prediction is not None
                                else choose_counterplay_action(
                                    step_observations[opponent_agent]["action_mask"],
                                    env.battle2,
                                    env.battle1,
                                    player1_max_damage_prediction,
                                    incoming_weight=counterplay_incoming_weight,
                                    support_actions=counterplay_support_actions,
                                    ko_bonus=counterplay_ko_bonus,
                                    speed_control_scale=counterplay_speed_control_scale,
                                    cap_overkill=counterplay_cap_overkill,
                                )
                            )
                        else:
                            actions.update(choose_masked_actions(random_observations, rng))
                    # When collecting teacher-controlled trajectories, keep
                    # the student attached for features/observations but let
                    # the counterplay teacher supply player1's behavior.
                    if player_strategy == "counterplay" and env.agent1_to_move:
                        player_agent = env.possible_agents[0]
                        (
                            actions[player_agent],
                            counterplay_policy_scores,
                            counterplay_policy_first_action,
                        ) = choose_counterplay_action(
                            step_observations[player_agent]["action_mask"],
                            env.battle1,
                            env.battle2,
                            counterplay_reference_action,
                            incoming_weight=counterplay_incoming_weight,
                            support_actions=counterplay_support_actions,
                            ko_bonus=counterplay_ko_bonus,
                                speed_control_scale=counterplay_speed_control_scale,
                            cap_overkill=counterplay_cap_overkill,
                            return_action_scores=True,
                        )
                    policy_details: Dict[str, Tuple[float, float]] = {
                        agent: (output[1], output[2])
                        for agent, output in policy_outputs.items()
                    }
                    native_action_features = {
                        agent: policies[agent].last_native_action_features.tolist()
                        for agent in policy_outputs
                        if policies[agent].last_native_action_features is not None
                    }
                    teacher_actions = {}
                    teacher_action_scores = {}
                    teacher_action_conditioning = {}
                    if collect_max_damage_labels and policies is not None:
                        player_agent = env.possible_agents[0]
                        if player_agent in step_observations and env.agent1_to_move:
                            teacher_actions[player_agent] = choose_max_damage_action(
                                step_observations[player_agent]["action_mask"], env.battle1
                            ).tolist()
                    if collect_counterplay_labels and policies is not None:
                        player_agent = env.possible_agents[0]
                        if (player_agent in step_observations and env.agent1_to_move
                                and counterplay_reference_action is not None):
                            teacher_action, action_scores, conditioned_second = choose_counterplay_action(
                                step_observations[player_agent]["action_mask"],
                                env.battle1,
                                env.battle2,
                                counterplay_reference_action,
                                incoming_weight=counterplay_incoming_weight,
                                support_actions=counterplay_support_actions,
                                ko_bonus=counterplay_ko_bonus,
                                speed_control_scale=counterplay_speed_control_scale,
                                cap_overkill=counterplay_cap_overkill,
                                return_action_scores=True,
                            )
                            # Keep the factorized target internally consistent:
                            # the second action is labeled under the teacher's
                            # chosen first action, which is also the first-action
                            # target and the context used at inference time.
                            teacher_first_action = int(teacher_action[0])
                            teacher_actions[player_agent] = [
                                teacher_first_action, int(conditioned_second)
                            ]
                            teacher_action_scores[player_agent] = action_scores.tolist()
                            teacher_action_conditioning[player_agent] = teacher_first_action
                    if team_preview and preview_orders_submitted < 2:
                        actions = sanitize_team_preview_actions(actions, env)
                    action_selection_seconds += time.perf_counter() - action_started_at
                    showdown_step_started_at = time.perf_counter()
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    showdown_step_seconds += time.perf_counter() - showdown_step_started_at
                    if team_preview and preview_orders_submitted < 2:
                        preview_orders_submitted += 1
                    transition: Dict[str, Any] = {
                        "observations": {
                            agent: payload["observation"].tolist()
                            for agent, payload in step_observations.items()
                        },
                        "action_masks": {
                            agent: np.asarray(payload["action_mask"]).tolist()
                            for agent, payload in step_observations.items()
                        },
                        "actions": {agent: action.tolist() for agent, action in actions.items()},
                        "predicted_opponent_actions": (
                            {env.possible_agents[0]: np.asarray(
                                policy_opponent_prediction
                                if policy_opponent_prediction is not None
                                else max_damage_prediction
                            ).tolist()}
                            if (policy_opponent_prediction is not None
                                or max_damage_prediction is not None) else {}
                        ),
                        "counterplay_action_scores": {
                            agent: scores.tolist()
                            for agent, scores in (
                                (env.possible_agents[0], counterplay_policy_scores),
                                (
                                    env.possible_agents[1],
                                    opponent_scores if episode_opponent == "policy" else None,
                                ),
                            )
                            if scores is not None
                        },
                        "counterplay_teacher_first_action": {
                            agent: first_action
                            for agent, first_action in (
                                (env.possible_agents[0], counterplay_policy_first_action),
                                (
                                    env.possible_agents[1],
                                    opponent_teacher_first
                                    if episode_opponent == "policy" else None,
                                ),
                            )
                            if first_action is not None
                        },
                        "rewards": rewards,
                        "reward_components": {
                            agent: infos.get(agent, {}).get("reward_components", {})
                            for agent in rewards
                        },
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    if policy_details:
                        transition["log_probs"] = {
                            agent: values[0] for agent, values in policy_details.items()
                        }
                        transition["values"] = {
                            agent: values[1] for agent, values in policy_details.items()
                        }
                    if policy_details and native_action_features:
                        transition["native_action_features"] = native_action_features
                    if teacher_actions:
                        transition["teacher_actions"] = teacher_actions
                    if teacher_action_scores:
                        transition["teacher_action_scores"] = teacher_action_scores
                    if teacher_action_conditioning:
                        transition["teacher_action_conditioning"] = teacher_action_conditioning
                    trajectory.append(transition)
                    steps += 1
                    if all(
                        terminated.get(agent, False) or truncated.get(agent, False)
                        for agent in env.possible_agents
                    ):
                        break
                result = {
                    "episode": episode,
            "opponent": episode_opponent,
            "opponent_tag": opponent_tag,
            "player_strategy": player_strategy,
                    "format": battle_format,
                    "teams": {"player1": team1_data, "player2": team2_data},
                    "preview_actions": preview_actions,
                    "preview_log_probs": preview_log_probs,
                    "team_selection_temperature": team_selection_temperature,
                    "agent_to_team": agent_to_team,
                    "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
                    "opponent_checkpoint": str(episode_opponent_checkpoint) if episode_opponent_checkpoint else None,
                    "policy_temperature": policy_temperature,
                    "teams_differ": team1 != team2,
                    "steps": steps,
                    "action_adjustments": action_adjustments,
                    "native_action_feature_dim": checkpoint_model_config.get(
                        "native_action_feature_dim", 0
                    ),
                    "native_action_feature_variant": checkpoint_model_config.get(
                        "native_action_feature_variant"
                    ),
                    "native_action_feature_encoder_version": NATIVE_ACTION_FEATURE_ENCODER_VERSION,
                    "rewards": rewards,
                    "terminated": terminated,
                    "truncated": truncated,
                    "trajectory": trajectory,
                }
        except (
            EpisodeTimeout,
            asyncio.TimeoutError,
            TimeoutError,
            OSError,
            WebSocketException,
        ) as exc:
            timed_out = True
            result = {
                "episode": episode,
                "opponent": episode_opponent,
                "opponent_tag": opponent_tag,
                "format": battle_format,
                "teams": {"player1": team1_data, "player2": team2_data},
                "preview_actions": preview_actions,
                "preview_log_probs": preview_log_probs,
                "team_selection_temperature": team_selection_temperature,
                "agent_to_team": agent_to_team,
                "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
                "opponent_checkpoint": str(episode_opponent_checkpoint) if episode_opponent_checkpoint else None,
                "policy_temperature": policy_temperature,
                "counterplay_prior_weight": counterplay_prior_weight,
                "teams_differ": team1 != team2,
                "steps": steps,
                "action_adjustments": action_adjustments,
                "native_action_feature_dim": checkpoint_model_config.get(
                    "native_action_feature_dim", 0
                ),
                "native_action_feature_variant": checkpoint_model_config.get(
                    "native_action_feature_variant"
                ),
                "native_action_feature_encoder_version": NATIVE_ACTION_FEATURE_ENCODER_VERSION,
                "error": str(exc),
                "timed_out": True,
                "trajectory": trajectory,
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            unwinding_exception = sys.exc_info()[0] is not None
            needs_discard = timed_out or unwinding_exception
            try:
                if needs_discard:
                    close_poke_env_nonblocking(env)
                else:
                    env.close(force=True, wait=True)
            except Exception:
                if not needs_discard:
                    raise
            if needs_discard:
                # A failed, interrupted, or timed-out battle may leave
                # poke-env clients with active rooms. Never reuse them.
                env = None
        elapsed_seconds = max(time.perf_counter() - episode_started_at, 1e-9)
        runtime_metrics = {
            "episode_wall_seconds": elapsed_seconds,
            "observation_seconds": observation_seconds,
            "action_selection_seconds": action_selection_seconds,
            "showdown_step_seconds": showdown_step_seconds,
            "joint_turns_per_second": steps / elapsed_seconds,
            "individual_decisions_per_second": 2 * steps / elapsed_seconds,
            "mean_showdown_step_ms": (
                1000.0 * showdown_step_seconds / steps if steps else None
            ),
        }
        result["runtime_metrics"] = runtime_metrics
        with output_path.open("a", encoding="utf-8") as output:
            for player_view in project_episode_to_player_views(result):
                output.write(json.dumps(player_view, sort_keys=True) + "\n")
            output.flush()
        if telemetry is not None:
            telemetry.emit(
                "episode_completed",
                episode=episode,
                opponent=episode_opponent,
                games=1,
                steps=steps,
                timed_out=timed_out,
                **runtime_metrics,
                error=result.get("error"),
                player1_reward=result.get("rewards", {}).get("player1"),
                player2_reward=result.get("rewards", {}).get("player2"),
            )
    if telemetry is not None:
        telemetry.emit("run_completed", episodes=episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native Showdown self-play")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/logs/showdown_self_play.jsonl"))
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team-seed", type=int,
                        help="Optional independent seed for generated team rosters; use the same value across runs to repeat rosters while resampling battles.")
    parser.add_argument("--team1-json", type=Path,
                        help="Fixed six-Pokémon JSON team for player 1 (array or object with pokemon/team list).")
    parser.add_argument("--team2-json", type=Path,
                        help="Fixed six-Pokémon JSON team for player 2 (array or object with pokemon/team list).")
    parser.add_argument("--episode-timeout", type=float, default=120.0)
    parser.add_argument(
        "--challenge-timeout", type=float, default=60.0,
        help="Seconds to wait for Showdown to create a challenged battle.",
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument(
        "--opponent-checkpoint", type=Path, action="append",
        help="Frozen policy opponent checkpoint used for 'policy' episodes; repeat to rotate through a pool.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument("--run-id", default="native-rollouts")
    parser.add_argument(
        "--opponent-tag",
        help="Optional group label for a fixed opponent archetype, such as tailwind_offense.",
    )
    parser.add_argument(
        "--opponent", choices=("policy", "random", "max_damage", "counterplay", "no_guard_fissure", "perish_trap"), default="policy"
    )
    parser.add_argument(
        "--player-strategy", choices=("policy", "counterplay", "random"),
        default=None,
        help="Player 1 action strategy; counterplay is the state-aware one-ply teacher.",
    )
    parser.add_argument(
        "--opponent-schedule",
        help="Comma-separated opponent mix applied episode by episode, e.g. policy,max_damage.",
    )
    parser.add_argument(
        "--opponent-policy-deterministic",
        action="store_true",
        help="Use argmax actions for frozen policy opponents while the training policy keeps sampling.",
    )
    parser.add_argument(
        "--team-preview",
        action="store_true",
        help="Let the policy choose the four Pokemon and lead order during VGC preview.",
    )
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument("--team-selection-mode", choices=("random", "policy"), default="random")
    parser.add_argument("--team-selection-side", choices=("both", "player1"), default="both")
    parser.add_argument("--preview-action-player1", type=int)
    parser.add_argument("--preview-action-player2", type=int)
    parser.add_argument(
        "--team-selection-temperature", type=float, default=0.0,
        help="Sample policy previews at this temperature; zero uses argmax.",
    )
    parser.add_argument(
        "--collect-max-damage-labels", action="store_true",
        help="Store the max-damage heuristic's legal joint action for player1 at each observed state.",
    )
    parser.add_argument(
        "--collect-counterplay-labels", action="store_true",
        help="Store one-ply offense/survival counterplay labels against the predicted max-damage action.",
    )
    parser.add_argument("--counterplay-incoming-weight", type=float, default=1.0)
    parser.add_argument("--counterplay-ko-bonus", type=float, default=0.7)
    parser.add_argument("--counterplay-speed-control-scale", type=float, default=1.0)
    parser.add_argument("--counterplay-prior-weight", type=float,
                        help="Blend normalized counterplay scores into player1 policy logits.")
    parser.add_argument(
        "--counterplay-predict-policy-opponent", action="store_true",
        help="Condition player1 counterplay scores and action features on the frozen policy's predicted move.",
    )
    parser.add_argument(
        "--counterplay-support-actions", action=argparse.BooleanOptionalAction,
        default=False,
        help="Score Helping Hand and useful Tailwind/Trick Room choices in counterplay mode.",
    )
    parser.add_argument(
        "--counterplay-cap-overkill", action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap joint damage at each target's remaining HP and award each KO once.",
    )
    args = parser.parse_args()
    run_self_play(
        episodes=args.episodes,
        output_path=args.output,
        battle_format=args.battle_format,
        seed=args.seed,
        team_seed=args.team_seed,
        team1_json=args.team1_json,
        team2_json=args.team2_json,
        episode_timeout=args.episode_timeout,
        challenge_timeout=args.challenge_timeout,
        policy_checkpoint=args.policy_checkpoint,
        opponent_checkpoints=args.opponent_checkpoint,
        device=args.device,
        policy_temperature=args.policy_temperature,
        telemetry_path=args.telemetry_path,
        run_id=args.run_id,
        opponent_tag=args.opponent_tag,
        opponent=args.opponent,
        player_strategy=args.player_strategy,
        opponent_schedule=(
            [value.strip() for value in args.opponent_schedule.split(",") if value.strip()]
            if args.opponent_schedule
            else None
        ),
        opponent_policy_deterministic=args.opponent_policy_deterministic,
        team_preview=args.team_preview,
        team_selection_checkpoint=args.team_selection_checkpoint,
        team_selection_mode=args.team_selection_mode,
        team_selection_temperature=args.team_selection_temperature,
        preview_action_player1=args.preview_action_player1,
        preview_action_player2=args.preview_action_player2,
        collect_max_damage_labels=args.collect_max_damage_labels,
        collect_counterplay_labels=args.collect_counterplay_labels,
        counterplay_incoming_weight=args.counterplay_incoming_weight,
        counterplay_ko_bonus=args.counterplay_ko_bonus,
        counterplay_speed_control_scale=args.counterplay_speed_control_scale,
        counterplay_support_actions=args.counterplay_support_actions,
        counterplay_cap_overkill=args.counterplay_cap_overkill,
        team_selection_side=args.team_selection_side,
        counterplay_prior_weight=args.counterplay_prior_weight,
        counterplay_predict_policy_opponent=args.counterplay_predict_policy_opponent,
    )


if __name__ == "__main__":
    main()
