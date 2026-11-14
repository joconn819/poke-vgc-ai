"""Accept local Showdown challenges and play with the latest complete checkpoint.

The checkpoint is loaded when team preview begins, then held for that whole
battle. Select a complete checkpoint when starting the bot; concurrent training
and automatic refresh are optional.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data.normalize import to_id_str
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder
from poke_env.environment.doubles_env import DoublesEnv

from scripts.run_showdown_self_play import team_to_showdown_text
from scripts.run_showdown_self_play import pack_showdown_team
from src.baselines.native_damage import (
    choose_max_damage_action,
    score_max_damage_action,
)
from src.belief.closed_sheet import ClosedSheetBeliefTracker
from src.env.showdown_env import ShowdownDoublesEnv
from src.env.public_battle_history import PublicBattleHistory
from src.features.public_mechanics import encode_public_mechanics
from src.models.policy import TransformerPolicy
from src.models.state_value import StateValueEvaluator, load_state_value_checkpoint
from src.search.local_showdown import LocalShowdownSearchPool, choose_opening_search_action
from src.training.self_play import PolicyAgent
from src.training.team_selection import (
    ACTIONS, TeamSelectionPolicy, preview_command, team_features,
)


FORMAT = "gen9championsdoubles2026mcclosed"


def load_team(seed: int, format_id: str, showdown_path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["node", "scripts/showdown_random_team.js", str(seed), format_id],
        check=True, capture_output=True, text=True,
        env={**os.environ, "POKEMON_SHOWDOWN_PATH": str(showdown_path)},
    )
    team = json.loads(result.stdout)
    if not isinstance(team, list) or len(team) != 6:
        raise ValueError("Showdown team generator did not return six Pokemon")
    return team


def species_only(team: Any) -> list[dict[str, str]]:
    """Project parsed opponent data down to species names only."""
    result = []
    for pokemon in (team or {}).values():
        species = getattr(pokemon, "species", None)
        if species and not any(row["species"] == species for row in result):
            result.append({"species": species})
    return result[:6]


class RandomChampionsTeambuilder(Teambuilder):
    """Generate a fresh legal team for each accepted challenge."""

    def __init__(self, seed: int, format_id: str, showdown_path: Path):
        self.seed = seed
        self.format_id = format_id
        self.showdown_path = showdown_path
        self.team_number = 0
        self.last_team_data: list[dict[str, Any]] | None = None

    def yield_team(self) -> str:
        current_seed = self.seed + self.team_number
        self.last_team_data = load_team(current_seed, self.format_id, self.showdown_path)
        self.team_number += 1
        return pack_showdown_team(team_to_showdown_text(self.last_team_data))


class LanCheckpointPlayer(Player):
    def __init__(
        self,
        *,
        checkpoint: Path,
        team_builder: RandomChampionsTeambuilder,
        showdown_path: Path,
        search_turn_one: bool = False,
        search_max_candidates: int = 4,
        search_hidden_worlds: int = 2,
        search_chance_samples: int = 2,
        search_timeout_seconds: float = 3.0,
        search_action_selection: str = "equilibrium",
        search_workers: int = 4,
        search_seed: int = 1,
        state_value_checkpoint: Path | None = None,
        state_value_device: str = "auto",
        team_selection_checkpoint: Path | None = None,
        max_damage_score_gate: bool = False,
        first_n_max_damage: int = 0,
        max_damage_action_steps: set[int] | None = None,
        search_pool: LocalShowdownSearchPool | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint
        self.team_builder = team_builder
        self.showdown_path = showdown_path
        self.search_turn_one = search_turn_one
        self.search_max_candidates = search_max_candidates
        self.search_hidden_worlds = search_hidden_worlds
        self.search_chance_samples = search_chance_samples
        self.search_timeout_seconds = search_timeout_seconds
        self.search_action_selection = search_action_selection
        self.search_workers = search_workers
        self.search_seed = search_seed
        self.search_pool = search_pool
        self.state_value_checkpoint = state_value_checkpoint
        self.team_selection_checkpoint = team_selection_checkpoint
        self.max_damage_score_gate = max_damage_score_gate
        self.first_n_max_damage = first_n_max_damage
        self.max_damage_action_steps = set(max_damage_action_steps or ())
        self.team_selection_policy: TeamSelectionPolicy | None = None
        self.team_selection_config: dict[str, Any] | None = None
        if team_selection_checkpoint is not None:
            selector_state = torch.load(
                team_selection_checkpoint, map_location="cpu", weights_only=True
            )
            self.team_selection_config = selector_state["model_config"]
            self.team_selection_policy = TeamSelectionPolicy(
                self.team_selection_config["feature_dim"],
                self.team_selection_config["hidden_dim"],
                opponent_conditioned=self.team_selection_config.get("opponent_conditioned", False),
                action_conditioned=self.team_selection_config.get("action_conditioned", False),
            )
            self.team_selection_policy.load_state_dict(selector_state["model_state_dict"])
            self.team_selection_policy.eval()
        if state_value_device == "auto":
            state_value_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_value_device = state_value_device
        self.state_value_evaluator: StateValueEvaluator | None = (
            load_state_value_checkpoint(state_value_checkpoint, map_location=state_value_device)
            if state_value_checkpoint is not None else None
        )
        self._search_counter = 0
        self._team_preview_choices: dict[str, list[str]] = {}
        self._beliefs: dict[str, ClosedSheetBeliefTracker] = {}
        self._public_histories: dict[str, PublicBattleHistory] = {}
        self._public_replay_offsets: dict[str, int] = {}
        self._search_attempted: set[str] = set()
        self._agents: dict[str, PolicyAgent] = {}
        self._models: dict[str, TransformerPolicy] = {}
        self._checkpoint_versions: dict[str, tuple[int, int]] = {}
        self._battle_decision_steps: dict[str, int] = {}
        self.decision_count = 0
        self.search_decision_count = 0
        self.decision_latencies_ms: list[float] = []
        self.invalid_action_fallbacks = 0
        self.max_damage_score_gate_overrides = 0
        self.first_n_max_damage_decisions = 0
        self.indexed_max_damage_decisions = 0
        self.public_context_updates = 0
        self.public_context_nonzero_updates = 0
        self.public_context_max_protect_streak = 0
        self.public_context_possible_fakeout_snapshots = 0

    def _agent_for(self, battle: DoubleBattle) -> PolicyAgent:
        tag = battle.battle_tag
        for old_tag in list(self._agents):
            old_battle = self.battles.get(old_tag)
            if old_battle is not None and old_battle.finished and old_tag != tag:
                self._agents.pop(old_tag, None)
                self._models.pop(old_tag, None)
                self._checkpoint_versions.pop(old_tag, None)
                self._battle_decision_steps.pop(old_tag, None)
                self._team_preview_choices.pop(old_tag, None)
                self._beliefs.pop(old_tag, None)
                self._public_histories.pop(old_tag, None)
                self._public_replay_offsets.pop(old_tag, None)
                self._search_attempted.discard(old_tag)
        if tag in self._agents:
            # Opponent data is refreshed only from poke-env's parsed, revealed
            # battle state and projected to species-only before policy use.
            self._agents[tag].opponent_team = species_only(battle.opponent_team)
            if tag in self._beliefs:
                self._beliefs[tag].update_from_battle(battle)
            self._refresh_public_context(battle, self._agents[tag])
            return self._agents[tag]
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model = TransformerPolicy.from_checkpoint_state(state)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
        team_data = self.team_builder.last_team_data
        if team_data is None:
            raise RuntimeError("team builder did not provide the accepted battle's team")
        agent = PolicyAgent(
            model=model,
            player_team=team_data,
            opponent_team=species_only(battle.opponent_team),
            deterministic=True,
            device="cpu",
        )
        stat = self.checkpoint_path.stat()
        self._models[tag] = model  # Pin weights in memory until this battle ends.
        self._agents[tag] = agent
        self._beliefs[tag] = ClosedSheetBeliefTracker(
            getattr(pokemon, "species", "")
            for pokemon in getattr(battle, "teampreview_opponent_team", ())
            if getattr(pokemon, "species", None)
        )
        self._beliefs[tag].update_from_battle(battle)
        self._public_histories[tag] = PublicBattleHistory()
        self._public_replay_offsets[tag] = 0
        self._refresh_public_context(battle, agent)
        self._checkpoint_versions[tag] = (stat.st_mtime_ns, stat.st_size)
        print(f"battle={tag} checkpoint={self.checkpoint_path} version={stat.st_mtime_ns}:{stat.st_size}", flush=True)
        return agent

    def _refresh_public_context(self, battle: DoubleBattle, agent: PolicyAgent) -> None:
        """Consume only the public protocol events parsed by poke-env."""
        if not getattr(agent.model, "public_context_dim", 0):
            return
        tag = battle.battle_tag
        tracker = self._public_histories.setdefault(tag, PublicBattleHistory())
        events = getattr(battle, "_replay_data", ()) or ()
        offset = self._public_replay_offsets.get(tag, 0)
        if offset > len(events):
            # A reconnect may replace the battle object's replay buffer.
            tracker = PublicBattleHistory()
            self._public_histories[tag] = tracker
            offset = 0
        tracker.consume("|" + "|".join(str(part) for part in row) for row in events[offset:])
        self._public_replay_offsets[tag] = len(events)
        mechanics = tracker.for_battle(battle)
        context = encode_public_mechanics(
            mechanics,
            ShowdownDoublesEnv.embed_battle(None, battle),
            agent.player_team,
            agent.opponent_team,
            self.format,
        )
        agent.public_context_features = context
        self.public_context_updates += 1
        if np.any(context[:72]):
            self.public_context_nonzero_updates += 1
        self.public_context_max_protect_streak = max(
            self.public_context_max_protect_streak,
            int(round(float(context[:12].max(initial=0.0)) * 6.0)),
        )
        if np.any(context[60:66] > 0.5):
            self.public_context_possible_fakeout_snapshots += 1

    def teampreview(self, battle: DoubleBattle) -> str:
        # Pin the battle policy before preview; only revealed opponent species
        # are passed into the closed-sheet selector.
        self._agent_for(battle)
        roster = list(battle.team.values())
        if len(roster) != 6:
            raise RuntimeError(f"expected six own preview Pokemon, got {len(roster)}")
        selected_slots: list[int]
        if self.team_selection_policy is None:
            slots = list(range(1, len(roster) + 1))
            np.random.shuffle(slots)
            selected_slots = slots[:4]
            command = "/team " + "".join(str(slot) for slot in selected_slots)
        else:
            own_team = self.team_builder.last_team_data
            if own_team is None or len(own_team) != 6:
                raise RuntimeError("team selector requires the generated six-Pokemon team")
            cfg = self.team_selection_config or {}
            own_features = team_features(own_team, int(cfg["feature_dim"]))[None, ...]
            opponent_roster = [
                {"species": str(pokemon.species)}
                for pokemon in getattr(battle, "teampreview_opponent_team", ())
                if getattr(pokemon, "species", None)
            ][:6]
            if len(opponent_roster) != 6:
                # At preview the server should reveal all six species. Fail
                # closed to the unconditioned selector input only if this
                # selector was trained without opponent conditioning.
                if cfg.get("opponent_conditioned", False):
                    raise RuntimeError("opponent species preview is incomplete for conditioned selector")
                opponent_roster = [{"species": ""} for _ in range(6)]
            opponent_features = None
            if cfg.get("opponent_conditioned", False):
                opponent_features = team_features(
                    opponent_roster, int(cfg["feature_dim"]), revealed_only=True
                )[None, ...]
            with torch.inference_mode():
                scores = self.team_selection_policy(
                    own_features, opponent_features=opponent_features
                )[0]
                action_index = int(scores.argmax().item())
            action = ACTIONS[action_index]
            selected_slots = [index + 1 for index in action.leads + action.back]
            command = preview_command(action)
            print(
                "team_preview " + json.dumps({
                    "battle": battle.battle_tag,
                    "selector": str(self.team_selection_checkpoint),
                    "action": action_index,
                    "command": command,
                }, sort_keys=True),
                flush=True,
            )
        for slot in selected_slots:
            roster[slot - 1]._selected_in_teampreview = True
        selected_species = [str(roster[slot - 1].species) for slot in selected_slots]
        self._team_preview_choices[battle.battle_tag] = selected_species
        return command

    @staticmethod
    def _species_index(team: list[dict[str, Any]], species: str) -> int:
        wanted = to_id_str(species)
        for index, member in enumerate(team):
            if to_id_str(str(member.get("species", ""))) == wanted:
                return index
        raise ValueError(f"species {species!r} is absent from the battle roster")

    def _first_turn_search_order(self, battle: DoubleBattle) -> DoubleBattleOrder | None:
        """Search the opening move using only our team and public opponent facts."""
        if not self.search_turn_one or battle.battle_tag in self._search_attempted:
            return None
        tag = battle.battle_tag
        if battle.turn > 1:
            return None
        if any(battle.force_switch):
            return None
        self._search_attempted.add(tag)
        own_team = self.team_builder.last_team_data
        own_selection_species = self._team_preview_choices.get(tag)
        if own_team is None or own_selection_species is None or len(own_selection_species) != 4:
            self.logger.warning(
                "First-turn search unavailable: turn=%s own_team=%s own_selection_species=%s",
                battle.turn,
                own_team is not None,
                own_selection_species,
            )
            return None
        self._search_counter += 1
        started = time.perf_counter()
        own_preview = [self._species_index(own_team, species) for species in own_selection_species]
        action, report = choose_opening_search_action(
            battle,
            own_team,
            own_preview,
            revealed_facts=(
                self._beliefs[tag].public_facts() if tag in self._beliefs else None
            ),
            format_id=self.format,
            showdown_path=self.showdown_path,
            seed=self.search_seed + self._search_counter * 1009,
            hidden_worlds=self.search_hidden_worlds,
            max_candidates=self.search_max_candidates,
            chance_samples=self.search_chance_samples,
            action_selection_method=self.search_action_selection,
            workers=self.search_workers,
            evaluator=self.state_value_evaluator,
            search_pool=self.search_pool,
            timeout_seconds=self.search_timeout_seconds,
        )
        order = DoublesEnv.action_to_order(action, battle, strict=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(
            "first_turn_search " + json.dumps({
                "battle": tag,
                "selected_action": report["selected_action"]["command"],
                "matrix_game_method": report["matrix_game_method"],
                "matrix_game_value": report["matrix_game_value"],
                "matrix_game_exploitability": report["matrix_game_exploitability"],
                "mixed_row_strategy": report["mixed_row_strategy"],
                "worlds": report["world_count"],
                "chance_samples": report["chance_samples"],
                "simulated_turn_advances": report["simulated_turn_advances"],
                "search_elapsed_ms": round(elapsed_ms, 1),
                "search_subprocess_wall_ms": round(report["search_subprocess_wall_ms"], 1),
                "hidden_hypothesis_setup_ms": round(report["hidden_hypothesis_setup_ms"], 1),
                "frontier_application_ms": round(report.get("frontier_application_ms", 0.0), 1),
                "value_inference_ms": round(report.get("value_inference_ms", 0.0), 1),
                "action_translation_ms": round(report["action_translation_ms"], 1),
                "search_adapter_wall_ms": round(report["search_adapter_wall_ms"], 1),
                "search_workers": report["search_workers"],
                "frontier": report["frontier"],
            }, sort_keys=True),
            flush=True,
        )
        return order

    def choose_move(self, battle: DoubleBattle):
        started = time.perf_counter()
        tag = battle.battle_tag
        for old_tag in list(self._battle_decision_steps):
            old_battle = self.battles.get(old_tag)
            if old_battle is not None and old_battle.finished and old_tag != tag:
                self._battle_decision_steps.pop(old_tag, None)
        decision_step = self._battle_decision_steps.get(tag, 0)
        self._battle_decision_steps[tag] = decision_step + 1
        if (decision_step < self.first_n_max_damage
                or decision_step in self.max_damage_action_steps):
            try:
                early_mask = np.asarray(
                    ShowdownDoublesEnv.get_action_mask(battle), dtype=np.int8
                )
                early_action = choose_max_damage_action(early_mask, battle)
                early_order = DoublesEnv.action_to_order(
                    np.asarray(early_action, dtype=np.int64), battle, strict=True
                )
            except (IndexError, ValueError, TypeError, AttributeError) as error:
                self.logger.warning(
                    "Early max-damage action failed; using the learned policy: %s", error
                )
            else:
                self.first_n_max_damage_decisions += 1
                if decision_step in self.max_damage_action_steps:
                    self.indexed_max_damage_decisions += 1
                self.decision_count += 1
                self.decision_latencies_ms.append((time.perf_counter() - started) * 1000.0)
                return early_order
        agent = self._agent_for(battle)
        if self.search_turn_one:
            try:
                searched_order = self._first_turn_search_order(battle)
            except Exception as error:
                self.logger.warning("First-turn search failed; using policy action: %s", error)
                searched_order = None
            if searched_order is not None:
                self.decision_count += 1
                self.search_decision_count += 1
                self.decision_latencies_ms.append((time.perf_counter() - started) * 1000.0)
                return searched_order
        observation = ShowdownDoublesEnv.embed_battle(None, battle)
        mask = np.asarray(ShowdownDoublesEnv.get_action_mask(battle), dtype=np.int8)
        action, _, _ = agent.predict_native_with_value(observation, mask)
        if self.max_damage_score_gate:
            # poke-env's battle object contains this player's public view; the
            # gate only scores currently legal actions from revealed facts.
            try:
                gate_action = choose_max_damage_action(mask, battle)
                policy_score = sum(
                    score_max_damage_action(int(value), battle, slot)
                    for slot, value in enumerate(action)
                )
                gate_score = sum(
                    score_max_damage_action(int(value), battle, slot)
                    for slot, value in enumerate(gate_action)
                )
                if gate_score > policy_score + 1e-6:
                    action = gate_action
                    self.max_damage_score_gate_overrides += 1
            except (IndexError, ValueError, TypeError, AttributeError) as error:
                self.logger.warning(
                    "Immediate-damage gate failed; using policy action: %s", error
                )
        try:
            order = DoublesEnv.action_to_order(
                np.asarray(action, dtype=np.int64), battle, strict=True
            )
        except (IndexError, ValueError) as error:
            self.logger.warning("Policy order was invalid; using legal fallback: %s", error)
            self.invalid_action_fallbacks += 1
            order = self.choose_random_doubles_move(battle)
        self.decision_count += 1
        self.decision_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return order


async def serve(args: argparse.Namespace) -> None:
    showdown_path = args.showdown_path.resolve()
    team_builder = RandomChampionsTeambuilder(args.seed, args.format, showdown_path)
    server = ServerConfiguration(
        websocket_url=f"ws://{args.server_host}:{args.port}/showdown/websocket",
        authentication_url=f"http://{args.server_host}:{args.port}/action.php?",
    )
    search_pool = (
        LocalShowdownSearchPool(
            format_id=args.format,
            showdown_path=showdown_path,
            workers=args.search_workers,
            timeout_seconds=args.search_timeout_seconds,
        )
        if args.search_turn_one else None
    )
    player = LanCheckpointPlayer(
        checkpoint=args.checkpoint,
        team_builder=team_builder,
        showdown_path=showdown_path,
        search_turn_one=args.search_turn_one,
        search_max_candidates=args.search_max_candidates,
        search_hidden_worlds=args.search_hidden_worlds,
        search_chance_samples=args.search_chance_samples,
        search_timeout_seconds=args.search_timeout_seconds,
        search_action_selection=args.search_action_selection,
        search_workers=args.search_workers,
        search_seed=args.seed,
        state_value_checkpoint=args.state_value_checkpoint,
        state_value_device=args.state_value_device,
        team_selection_checkpoint=args.team_selection_checkpoint,
        max_damage_score_gate=args.max_damage_score_gate,
        first_n_max_damage=args.first_n_max_damage,
        max_damage_action_steps=set(args.max_damage_action_steps or ()),
        search_pool=search_pool,
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.format,
        server_configuration=server,
        team=team_builder,
        accept_open_team_sheet=False,
        max_concurrent_battles=1,
        start_timer_on_battle_start=True,
    )
    print(f"Listening for challenges as {player.username} on {args.format}; server={server.websocket_url}", flush=True)
    try:
        await player.accept_challenges(None, args.battles)
    finally:
        if search_pool is not None:
            search_pool.close()
        latency = np.asarray(player.decision_latencies_ms, dtype=np.float64)
        latency_summary = (
            {
                "mean": float(latency.mean()),
                "p50": float(np.percentile(latency, 50)),
                "p95": float(np.percentile(latency, 95)),
                "max": float(latency.max()),
            }
            if latency.size else None
        )
        print(
            "finished " + json.dumps({
                "accepted_battles": len(player.battles),
                "decisions": player.decision_count,
                "search_decisions": player.search_decision_count,
                "decision_latency_ms": latency_summary,
                "invalid_action_fallbacks": player.invalid_action_fallbacks,
                "max_damage_score_gate_overrides": player.max_damage_score_gate_overrides,
                "first_n_max_damage_decisions": player.first_n_max_damage_decisions,
                "indexed_max_damage_decisions": player.indexed_max_damage_decisions,
                "public_context_updates": player.public_context_updates,
                "public_context_nonzero_updates": player.public_context_nonzero_updates,
                "public_context_max_protect_streak": player.public_context_max_protect_streak,
                "public_context_possible_fakeout_snapshots": player.public_context_possible_fakeout_snapshots,
            }, sort_keys=True),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase10/live_training/checkpoints/latest.pt"))
    parser.add_argument("--team-selection-checkpoint", type=Path,
                        help="optional learned bring-four / lead-two policy for team preview")
    parser.add_argument(
        "--max-damage-score-gate", action="store_true",
        help=("replace a policy action with the legal immediate-damage action "
              "when its public-state score is strictly higher"),
    )
    parser.add_argument(
        "--first-n-max-damage", type=int, default=0,
        help=("use the legal Max Damage action for the first N battle decisions "
              "of each match, then return to the learned policy"),
    )
    parser.add_argument(
        "--max-damage-action-steps", type=int, nargs="+", default=None,
        help=("use the legal Max Damage action at these zero-based decision "
              "indices; the measured Round430 opening hybrid is `1 4`"),
    )
    parser.add_argument("--showdown-path", type=Path, default=Path(os.environ.get("POKEMON_SHOWDOWN_PATH", "/home/justin/p/poke_learning/pokemon-showdown")))
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--server-host", default="192.168.1.153", help="LAN IP or localhost of the local Showdown server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--username", default="VGCCheckpointBot")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--battles", type=int, default=1000000)
    parser.add_argument("--search-turn-one", action="store_true", help="use robust one-turn search for the opening move")
    parser.add_argument("--search-max-candidates", type=int, default=4)
    parser.add_argument("--search-hidden-worlds", type=int, default=2)
    parser.add_argument("--search-chance-samples", type=int, default=2,
                        help="repeated simulator outcomes averaged per joint-action pair")
    parser.add_argument(
        "--search-timeout-seconds", type=float, default=3.0,
        help="opening-search deadline; on expiry use the policy action",
    )
    parser.add_argument(
        "--search-action-selection", choices=("maximin", "equilibrium"),
        default="equilibrium",
        help="pure worst-case action or seeded regret-matched mixed strategy",
    )
    parser.add_argument("--search-workers", type=int, default=4,
                        help="CPU Node workers for independent hidden-world searches")
    parser.add_argument("--state-value-checkpoint", type=Path,
                        help="evaluate one-turn Showdown search leaves with a closed-sheet scalar value model")
    parser.add_argument("--state-value-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="device for state-value inference when learned search frontier is enabled")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.team_selection_checkpoint is not None and not args.team_selection_checkpoint.is_file():
        parser.error(f"team-selection checkpoint does not exist: {args.team_selection_checkpoint}")
    if (args.search_max_candidates < 4 or args.search_hidden_worlds < 1
            or args.search_chance_samples < 1 or args.search_workers < 1
            or args.search_timeout_seconds <= 0):
        parser.error("search requires at least four candidates, one hidden world, and one chance sample")
    if args.state_value_checkpoint is not None and not args.state_value_checkpoint.is_file():
        parser.error(f"state-value checkpoint does not exist: {args.state_value_checkpoint}")
    if args.state_value_device == "cuda" and not torch.cuda.is_available():
        parser.error("--state-value-device cuda was requested but CUDA is unavailable")
    if args.first_n_max_damage < 0:
        parser.error("--first-n-max-damage must be nonnegative")
    if args.max_damage_action_steps is not None:
        if args.first_n_max_damage:
            parser.error("choose either --first-n-max-damage or --max-damage-action-steps")
        if any(step < 0 for step in args.max_damage_action_steps):
            parser.error("--max-damage-action-steps must contain nonnegative indices")
        if len(set(args.max_damage_action_steps)) != len(args.max_damage_action_steps):
            parser.error("--max-damage-action-steps must not contain duplicates")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        print("LAN bot stopped", flush=True)


if __name__ == "__main__":
    main()
