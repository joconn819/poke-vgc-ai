"""Run native poke-env self-play battles against a local/remote Showdown server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder

from src.env.showdown_env import ShowdownDoublesEnv
from src.models.policy import TransformerPolicy
from src.training.self_play import PolicyAgent
from src.utils.config import REGULATION_CHAMPIONS_MB
from src.utils.team_pool import TeamPool


class EpisodeTimeout(RuntimeError):
    """Raised when a native battle exceeds its batch time budget."""


def _raise_episode_timeout(_signum: int, _frame: Any) -> None:
    raise EpisodeTimeout("native battle exceeded episode timeout")


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


def generate_showdown_champions_team_data(seed: int, format_id: str) -> List[Dict[str, Any]]:
    """Generate the raw set dictionaries for policy encoding."""
    showdown_path = os.environ.get("POKEMON_SHOWDOWN_PATH")
    if not showdown_path:
        raise RuntimeError("POKEMON_SHOWDOWN_PATH must point to the Pokemon Showdown checkout")
    result = subprocess.run(
        ["node", "scripts/showdown_random_team.js", str(seed), format_id],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "POKEMON_SHOWDOWN_PATH": showdown_path},
    )
    return json.loads(result.stdout)


def pack_showdown_team(team_text: str) -> str:
    """Convert Showdown text into the packed representation expected by poke-env."""
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(team_text))


def choose_masked_actions(
    observations: Dict[str, Dict[str, Any]], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Choose one legal native poke-env action per active Pokémon."""
    actions: Dict[str, np.ndarray] = {}
    for agent, payload in observations.items():
        mask = np.asarray(payload["action_mask"], dtype=np.int8)
        if mask.size % 2:
            raise ValueError(f"Expected two individual action masks for {agent}, got {mask.size}")
        half = mask.size // 2
        individual = []
        for position, position_mask in enumerate((mask[:half], mask[half:])):
            legal = np.flatnonzero(position_mask)
            if legal.size == 0:
                raise RuntimeError(f"No legal action available for {agent}")
            choices = legal
            if position == 1 and individual:
                # During team preview and forced switches, submitting the same
                # slot twice is invalid in native doubles.
                distinct = legal[legal != individual[0]]
                if distinct.size:
                    choices = distinct
            individual.append(int(rng.choice(choices)))
        actions[agent] = np.asarray(individual, dtype=np.int64)
    return actions


def current_observations(env: ShowdownDoublesEnv) -> Dict[str, Dict[str, Any]]:
    if env.battle1 is None or env.battle2 is None:
        raise RuntimeError("DoublesEnv has no active battles")
    return {
        env.agents[0]: {
            "observation": env.embed_battle(env.battle1),
            "action_mask": env.get_action_mask(env.battle1),
        },
        env.agents[1]: {
            "observation": env.embed_battle(env.battle2),
            "action_mask": env.get_action_mask(env.battle2),
        },
    }


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
    server_configuration: ServerConfiguration | None = None,
    episode_timeout: int = 120,
    policy_checkpoint: Path | None = None,
    device: str = "cpu",
) -> None:
    """Run native Showdown battles and append one result per episode."""
    pool = TeamPool("data/teams/champions_mb.json", REGULATION_CHAMPIONS_MB)
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        team1_data = generate_showdown_champions_team_data(seed + episode * 2, battle_format)
        team2_data = generate_showdown_champions_team_data(seed + episode * 2 + 1, battle_format)
        team1_text = team_to_showdown_text(team1_data)
        team2_text = team_to_showdown_text(team2_data)
        team1 = pack_showdown_team(team1_text)
        team2 = pack_showdown_team(team2_text)
        if team1 == team2:
            raise RuntimeError(f"Showdown generated identical teams for episode {episode}")
        env_options: Dict[str, Any] = {
            "battle_format": battle_format,
            "server_configuration": server_configuration or LocalhostServerConfiguration,
            "team": team1,
            "start_listening": True,
            "choose_on_teampreview": False,
            "fake": False,
            "strict": False,
        }
        env = ShowdownDoublesEnv(
            **env_options,
        )
        env.agent2.update_team(team2)
        policies = None
        if policy_checkpoint is not None:
            model1 = TransformerPolicy()
            checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
            model1.load_state_dict(checkpoint["model_state_dict"])
            model1.eval()
            model2 = TransformerPolicy()
            model2.load_state_dict(checkpoint["model_state_dict"])
            model2.eval()
            policies = {
                env.possible_agents[0]: PolicyAgent(
                    model1, team1_data, team2_data, device=device
                ),
                env.possible_agents[1]: PolicyAgent(
                    model2, team2_data, team1_data, device=device
                ),
            }
        steps = 0
        timed_out = False
        trajectory: List[Dict[str, Any]] = []
        try:
                signal.signal(signal.SIGALRM, _raise_episode_timeout)
                signal.alarm(episode_timeout)
                observations, _ = env.reset(seed=seed + episode)
                while env.agents:
                    step_observations = current_observations(env)
                    if policies is None:
                        actions = choose_current_actions(env, rng)
                        policy_details: Dict[str, Tuple[float, float]] = {}
                    else:
                        policy_outputs = {
                            agent: policies[agent].predict_native_with_value(
                                payload["observation"], payload["action_mask"]
                            )
                            for agent, payload in step_observations.items()
                        }
                        actions = {agent: output[0] for agent, output in policy_outputs.items()}
                        policy_details = {
                            agent: (output[1], output[2])
                            for agent, output in policy_outputs.items()
                        }
                    observations, rewards, terminated, truncated, _ = env.step(actions)
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
                        "rewards": rewards,
                    }
                    if policy_details:
                        transition["log_probs"] = {
                            agent: values[0] for agent, values in policy_details.items()
                        }
                        transition["values"] = {
                            agent: values[1] for agent, values in policy_details.items()
                        }
                    trajectory.append(transition)
                    steps += 1
                    if all(terminated.values()) or all(truncated.values()):
                        break
                result = {
                    "episode": episode,
                    "teams_differ": team1 != team2,
                    "steps": steps,
                    "rewards": rewards,
                    "terminated": terminated,
                    "truncated": truncated,
                    "trajectory": trajectory,
                }
        except EpisodeTimeout as exc:
            timed_out = True
            result = {
                "episode": episode,
                "teams_differ": team1 != team2,
                "steps": steps,
                "error": str(exc),
                "timed_out": True,
                "trajectory": trajectory,
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            env.close(force=not timed_out, wait=not timed_out)
        with output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native Showdown self-play")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/logs/showdown_self_play.jsonl"))
    parser.add_argument("--battle-format", default="gen9championsvgc2026regmb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-timeout", type=int, default=120)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_self_play(
        episodes=args.episodes,
        output_path=args.output,
        battle_format=args.battle_format,
        seed=args.seed,
        episode_timeout=args.episode_timeout,
        policy_checkpoint=args.policy_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
