"""Shared closed-sheet opening search adapter for local Showdown battles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import re
import subprocess
import time
from typing import Any, Sequence

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data.normalize import to_id_str
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from src.belief.closed_sheet import (
    ClosedSheetBeliefTracker,
    ShowdownHiddenTeamSampler,
    possible_selected_rosters,
    sample_showdown_hidden_teams,
)
from src.models.state_value import StateValueEvaluator
from src.search.frontier import apply_state_value_frontier
from src.search.robust_actions import (
    combine_hidden_world_response_columns,
    select_robust_action,
)


ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = ROOT / "scripts" / "search_showdown_first_turn.js"


class LocalShowdownSearchPool:
    """Keep local Showdown search processes warm across decisions/matches."""

    def __init__(
        self,
        *,
        format_id: str,
        showdown_path: str | Path,
        workers: int = 4,
        timeout_seconds: float = 7.0,
    ) -> None:
        if workers < 1 or timeout_seconds <= 0:
            raise ValueError("search pool needs workers and a positive timeout")
        self.format_id = format_id
        self.showdown_path = Path(showdown_path).resolve()
        self.timeout_seconds = timeout_seconds
        self.processes = [self._start_worker() for _ in range(workers)]
        self.stdout_buffers = [bytearray() for _ in self.processes]
        self.hidden_sampler = ShowdownHiddenTeamSampler(
            format_id=format_id, showdown_path=self.showdown_path
        )

    def _start_worker(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["node", str(SEARCH_SCRIPT), self.format_id],
            cwd=ROOT,
            env={
                **os.environ,
                "POKEMON_SHOWDOWN_PATH": str(self.showdown_path),
                "POKE_SEARCH_PERSISTENT": "1",
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    @property
    def worker_count(self) -> int:
        return len(self.processes)

    def run_requests(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        if len(requests) > self.worker_count:
            raise ValueError("more search shards than persistent workers")
        if not hasattr(self, "stdout_buffers"):
            # Keep the pool usable by lightweight wrappers/subclasses that
            # provide workers without running the process-launching __init__.
            self.stdout_buffers = [bytearray() for _ in self.processes]
        elif len(self.stdout_buffers) != len(self.processes):
            raise RuntimeError("search worker and stdout-buffer counts differ")
        started = time.perf_counter()
        request_timeout = (
            self.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if request_timeout <= 0:
            raise TimeoutError("local Showdown search deadline expired before dispatch")
        deadline = started + request_timeout
        processes = self.processes[:len(requests)]
        for index, (process, request) in enumerate(zip(processes, requests)):
            if process.poll() is not None:
                process = self._start_worker()
                self.processes[index] = process
                processes[index] = process
                self.stdout_buffers[index] = bytearray()
            if process.stdin is None:
                raise RuntimeError("persistent search worker stdin is unavailable")
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        if any(process.stdout is None for process in processes):
            raise RuntimeError("persistent search worker stdout is unavailable")
        pending = {
            process.stdout.fileno(): index
            for index, process in enumerate(processes)
            if process.stdout is not None
        }
        reports: list[dict[str, Any] | None] = [None] * len(processes)
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                # Persistent workers execute a request synchronously. Leaving
                # a timed-out request alive would keep burning CPU after the
                # caller has fallen back, and a later request could queue
                # behind the stale work. Stop only workers that have not
                # replied; run_requests will replace them on the next call.
                for worker_index in pending.values():
                    process = processes[worker_index]
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1.0)
                raise TimeoutError("persistent local Showdown search exceeded its deadline")
            ready, _, _ = select.select(list(pending), [], [], remaining)
            if not ready:
                continue
            for output_fd in ready:
                index = pending[output_fd]
                chunk = os.read(output_fd, 65536)
                if not chunk:
                    pending.pop(output_fd, None)
                    process = processes[index]
                    raise RuntimeError(
                        f"persistent search worker exited with status {process.poll()}"
                    )
                buffer = self.stdout_buffers[index]
                buffer.extend(chunk)
                newline_at = buffer.find(b"\n")
                if newline_at < 0:
                    continue
                line = bytes(buffer[:newline_at])
                del buffer[:newline_at + 1]
                pending.pop(output_fd, None)
                payload = json.loads(line.decode("utf-8"))
                if "error" in payload:
                    raise RuntimeError(
                        f"local Showdown search failed: {payload.get('error')}\n"
                        f"{payload.get('stack', '')}"
                    )
                reports[index] = payload
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [report for report in reports if report is not None], elapsed_ms

    def close(self) -> None:
        self.hidden_sampler.close()
        for process in self.processes:
            if process.stdin is not None:
                process.stdin.close()
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)

    def __enter__(self) -> "LocalShowdownSearchPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _species_rows(team: Any) -> list[dict[str, str]]:
    members = team.values() if isinstance(team, dict) else team or ()
    result: list[dict[str, str]] = []
    for pokemon in members:
        species = getattr(pokemon, "species", None)
        if species and not any(row["species"] == str(species) for row in result):
            result.append({"species": str(species)})
    return result[:6]


def _search_command_to_order(
    command: str,
    battle: DoubleBattle,
    position: int,
    own_team: list[dict[str, Any]],
    own_preview: list[int],
) -> SingleBattleOrder:
    if command == "pass":
        return PassBattleOrder()
    switch = re.fullmatch(r"switch\s+(\d+)", command)
    if switch:
        selected_slot = int(switch.group(1)) - 1
        member = own_team[own_preview[selected_slot]]
        species_id = to_id_str(str(member.get("species", "")))
        for candidate in battle.valid_orders[position]:
            target = getattr(candidate, "order", None)
            if hasattr(target, "species") and to_id_str(str(target.species)) == species_id:
                return candidate
        raise ValueError(f"search switch is not a current legal order: {command}")
    move = re.fullmatch(
        r"move\s+(\d+)(?:\s+([+-]?\d+))?(?:\s+(mega|terastallize))?",
        command,
    )
    if move:
        move_slot = int(move.group(1)) - 1
        active = battle.active_pokemon[position]
        if active is None:
            raise ValueError("move search result has no active Pokemon")
        known_moves = list(active.moves.values())[:4]
        if move_slot >= len(known_moves):
            raise ValueError(f"search move slot is not known: {command}")
        move_id = known_moves[move_slot].id
        target = int(move.group(2)) if move.group(2) is not None else 0
        modifier = move.group(3)
        for candidate in battle.valid_orders[position]:
            if (
                getattr(getattr(candidate, "order", None), "id", None) == move_id
                and getattr(candidate, "move_target", 0) == target
                and bool(getattr(candidate, "mega", False)) == (modifier == "mega")
                and bool(getattr(candidate, "terastallize", False))
                == (modifier == "terastallize")
            ):
                return candidate
        raise ValueError(f"search move is not a current legal order: {command}")
    raise ValueError(f"unsupported search action: {command!r}")


def _legal_move_matches(
    battle: DoubleBattle, position: int, semantic: dict[str, Any]
) -> list[Any]:
    """Match a searched move to current legal orders, preserving target choice.

    A sampled branch can encode a forced continuation target differently from
    the live request. In that case accept a translation only when the live
    request has exactly one legal order for the same move and mechanics flags;
    never guess among multiple targets.
    """
    same_move = []
    exact = []
    for candidate in battle.valid_orders[position]:
        target = getattr(candidate, "order", None)
        if getattr(target, "id", None) != semantic.get("move_id"):
            continue
        if bool(getattr(candidate, "mega", False)) != bool(semantic.get("mega", False)):
            continue
        if bool(getattr(candidate, "terastallize", False)) != bool(semantic.get("terastallize", False)):
            continue
        same_move.append(candidate)
        if int(getattr(candidate, "move_target", 0)) == int(semantic.get("target", 0)):
            exact.append(candidate)
    if exact:
        return exact
    return same_move if len(same_move) == 1 else []


def choose_opening_search_action(
    battle: DoubleBattle,
    own_team: list[dict[str, Any]],
    own_preview: Sequence[int],
    *,
    opponent_preview_species: Sequence[str] | None = None,
    revealed_facts: dict[str, dict[str, Any]] | None = None,
    format_id: str,
    showdown_path: str | Path,
    seed: int,
    hidden_worlds: int = 2,
    max_candidates: int = 4,
    max_response_candidates: int | None = None,
    chance_samples: int = 4,
    candidate_damage_samples: int = 4,
    action_selection_method: str = "equilibrium",
    regret_iterations: int = 1000,
    return_response_candidates: bool = False,
    workers: int = 4,
    evaluator: StateValueEvaluator | None = None,
    search_pool: LocalShowdownSearchPool | None = None,
    timeout_seconds: float = 7.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a legal opening action using only own sets and public opponent facts.

    The opponent's hidden team sets are sampled from the local Showdown set
    builder using the six revealed species and accumulated public facts. Search
    worlds include each possible legal back pair consistent with the revealed
    leads. The learned evaluator receives only player-view frontier features.
    """
    adapter_started = time.perf_counter()
    if timeout_seconds <= 0:
        raise ValueError("search decision timeout must be positive")
    deadline = adapter_started + timeout_seconds

    def remaining_seconds() -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("opening search exceeded its decision deadline")
        return remaining
    if len(own_preview) != 4 or len(set(own_preview)) != 4:
        raise ValueError("own_preview must contain four distinct team indices")
    response_candidate_limit = (
        max_candidates if max_response_candidates is None
        else int(max_response_candidates)
    )
    if (hidden_worlds < 1 or max_candidates < 4
            or response_candidate_limit < 1
            or chance_samples < 1 or candidate_damage_samples < 1 or workers < 1):
        raise ValueError(
            "search needs one world, at least four own candidates, at least one response candidate, and one chance sample"
        )
    if action_selection_method not in {"maximin", "equilibrium"}:
        raise ValueError("action_selection_method must be 'maximin' or 'equilibrium'")
    if regret_iterations < 1:
        raise ValueError("regret_iterations must be positive")
    if battle.turn > 1 or any(battle.force_switch):
        raise ValueError("opening search is available only at the initial move request")

    own_leads: list[int] = []
    for active in battle.active_pokemon[:2]:
        if active is None:
            raise ValueError("both own leads must be visible before opening search")
        own_leads.append(next(
            index for index, member in enumerate(own_team)
            if to_id_str(str(member.get("species", ""))) == to_id_str(str(active.species))
        ))
    own_back = [index for index in own_preview if index not in own_leads]
    if len(own_back) != 2:
        raise ValueError("own preview does not match the active leads")

    if opponent_preview_species is not None:
        opponent_species = [{"species": str(species)} for species in opponent_preview_species]
    else:
        opponent_species = _species_rows(
            getattr(battle, "teampreview_opponent_team", ())
        ) or _species_rows(getattr(battle, "opponent_team", {}))
    if len(opponent_species) != 6:
        raise ValueError("six public opponent preview species are required")
    opponent_leads: list[int] = []
    for active in battle.opponent_active_pokemon[:2]:
        if active is None:
            raise ValueError("both opponent leads must be publicly visible")
        opponent_leads.append(next(
            index for index, member in enumerate(opponent_species)
            if to_id_str(member["species"]) == to_id_str(str(active.species))
        ))
    if len(set(opponent_leads)) != 2:
        raise ValueError("opponent leads are absent from the public preview")

    belief = ClosedSheetBeliefTracker(row["species"] for row in opponent_species)
    belief.update_from_battle(battle)
    facts = belief.public_facts()
    # A Transform event plus our own current move request exposes the copied
    # opponent species and moves to this player. Condition sampled hidden sets
    # on those public observations so hypotheses cannot give our own active
    # Pokémon different move menus (for example, Imposter copying a lead).
    role = str(getattr(battle, "player_role", "p1")).lower()
    opponent_role = "p2" if role.startswith("p1") else "p1"
    last_request = getattr(battle, "_last_request", {}) or {}
    active_requests = last_request.get("active", ())
    opponent_actives = getattr(battle, "opponent_active_pokemon", ())
    for raw_event in getattr(battle, "_replay_data", ()) or ():
        parts = (
            [str(part).strip() for part in raw_event]
            if isinstance(raw_event, (list, tuple))
            else [part.strip() for part in str(raw_event).split("|")]
        )
        if len(parts) < 4 or parts[1] != "-transform":
            continue
        actor_slot = parts[2].split(":", 1)[0].strip().lower()
        target_slot = parts[3].split(":", 1)[0].strip().lower()
        if (
            not actor_slot.startswith(role)
            or not target_slot.startswith(opponent_role)
            or len(actor_slot) < 3
            or len(target_slot) < 3
        ):
            continue
        active_index = {"a": 0, "b": 1}.get(actor_slot[2])
        target_index = {"a": 0, "b": 1}.get(target_slot[2])
        if active_index is None or target_index is None:
            continue
        if active_index >= len(active_requests) or target_index >= len(opponent_actives):
            continue
        target = opponent_actives[target_index]
        copied_species = to_id_str(str(getattr(target, "species", "")))
        if copied_species not in facts:
            continue
        copied_moves = {
            to_id_str(str(move.get("id", "")))
            for move in active_requests[active_index].get("moves", ())
            if isinstance(move, dict)
        }
        copied_moves.discard("")
        facts[copied_species]["moves"] = sorted(
            set(facts[copied_species].get("moves", ())) | copied_moves
        )
        # Imposter copies the opponent's complete move menu. Unlike ordinary
        # revealed move observations (which are only a subset of a set), this
        # makes the four-move set exact for every sampled hidden world.
        facts[copied_species]["moves_exact"] = True
    for key, known in (revealed_facts or {}).items():
        if key not in facts:
            continue
        merged = facts[key]
        merged["moves"] = sorted(set(merged.get("moves", ())) | set(known.get("moves", ())))
        merged["moves_exact"] = bool(
            merged.get("moves_exact") or known.get("moves_exact")
        )
        for field in ("item", "ability", "hp_fraction", "status", "tera_type"):
            if merged.get(field) is None and known.get(field) is not None:
                merged[field] = known[field]
    hypothesis_started = time.perf_counter()
    sample_error: Exception | None = None
    hypotheses = None
    # A legal random-set hypothesis can be rare when several revealed facts
    # constrain the team. Retry with deterministic seed offsets before falling
    # back to the base policy at the caller.
    for retry in range(3):
        try:
            hypotheses = sample_showdown_hidden_teams(
                [row["species"] for row in opponent_species],
                facts,
                format_id=format_id,
                count=hidden_worlds,
                seed=seed + retry * 1_000_003,
                showdown_path=showdown_path,
                sampler=search_pool.hidden_sampler if search_pool is not None else None,
                timeout_seconds=remaining_seconds(),
            )
            break
        except Exception as error:
            sample_error = error
    if hypotheses is None:
        assert sample_error is not None
        raise sample_error
    setup_ms = (time.perf_counter() - hypothesis_started) * 1000.0
    # The opponent's selected backs and their order are hidden. Include every
    # legal preview selection consistent with publicly appeared species; the
    # latter also makes the world builder reusable after opening turns.
    selected_rosters = possible_selected_rosters(
        [row["species"] for row in opponent_species],
        [opponent_species[index]["species"] for index in opponent_leads],
        belief.appeared_species(),
    )
    preview_index_by_species = {
        to_id_str(row["species"]): index
        for index, row in enumerate(opponent_species)
    }
    selected_roster_indexes = [
        [preview_index_by_species[to_id_str(species)] for species in roster]
        for roster in selected_rosters
    ]
    worlds = [
        {
            "own_team": own_team,
            "opponent_team": hidden_team,
            "own_preview": own_leads + own_back,
            "opponent_preview": selected_roster,
        }
        for hidden_team in hypotheses
        for selected_roster in selected_roster_indexes
    ]
    if search_pool is not None and (
        search_pool.format_id != format_id or search_pool.showdown_path != Path(showdown_path).resolve()
    ):
        raise ValueError("persistent search pool format or local Showdown path mismatch")
    worker_limit = search_pool.worker_count if search_pool is not None else int(workers)
    worker_count = min(int(workers), worker_limit, len(worlds))
    shard_size = (len(worlds) + worker_count - 1) // worker_count
    shards = [
        (start, worlds[start:start + shard_size])
        for start in range(0, len(worlds), shard_size)
    ]

    shard_requests = []
    for start, shard_worlds in shards:
        request: dict[str, Any] = {
            "seed": int(seed),
            "world_start_index": start,
            "max_candidates": int(max_candidates),
            "max_response_candidates": response_candidate_limit,
            "chance_samples": int(chance_samples),
            "candidate_damage_samples": int(candidate_damage_samples),
            "worlds": shard_worlds,
            "return_response_candidates": bool(return_response_candidates),
        }
        if evaluator is not None:
            request["return_frontiers"] = True
        shard_requests.append(request)

    if search_pool is not None:
        shard_reports, subprocess_wall_ms = search_pool.run_requests(
            shard_requests, timeout_seconds=remaining_seconds()
        )
    else:
        def run_shard(request: dict[str, Any]) -> dict[str, Any]:
            result = subprocess.run(
                ["node", str(SEARCH_SCRIPT), format_id],
                input=json.dumps(request, separators=(",", ":")),
                check=True,
                capture_output=True,
                text=True,
                timeout=remaining_seconds(),
                env={**os.environ, "POKEMON_SHOWDOWN_PATH": str(Path(showdown_path).resolve())},
            )
            return json.loads(result.stdout)

        subprocess_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_shard, request) for request in shard_requests]
            shard_reports = [future.result() for future in futures]
        subprocess_wall_ms = (time.perf_counter() - subprocess_started) * 1000.0
    # Native action indices are the stable choices in the acting player's
    # current request. Search command strings use per-hypothesis move slots,
    # so they are not a reliable identity across hidden-world shards.
    candidate_key = lambda candidate: json.dumps(
        candidate.get("actions", ()), sort_keys=True, separators=(",", ":")
    )
    candidate_lists = [row["own_candidates"] for row in shard_reports]
    candidate_key_sets = [
        {candidate_key(candidate) for candidate in candidates}
        for candidates in candidate_lists
    ]
    def is_legal_in_live_request(candidate: dict[str, Any]) -> bool:
        actions = candidate.get("actions", ())
        if len(actions) != 2:
            return False
        for position, semantic in enumerate(actions):
            kind = semantic.get("type")
            if kind == "pass":
                continue
            found = False
            for legal_order in battle.valid_orders[position]:
                target = getattr(legal_order, "order", None)
                if kind == "switch" and hasattr(target, "base_species"):
                    found = (
                        to_id_str(str(target.species))
                        == to_id_str(str(semantic.get("species", "")))
                    )
                elif kind == "move" and getattr(target, "id", None) == semantic.get("move_id"):
                    found = (
                        int(getattr(legal_order, "move_target", 0))
                        == int(semantic.get("target", 0))
                        and bool(getattr(legal_order, "mega", False))
                        == bool(semantic.get("mega", False))
                        and bool(getattr(legal_order, "terastallize", False))
                        == bool(semantic.get("terastallize", False))
                    )
                if found:
                    break
            if not found:
                return False
        return True

    common_candidate_keys = [
        candidate_key(candidate)
        for candidate in candidate_lists[0]
        if (
            all(candidate_key(candidate) in keys for keys in candidate_key_sets[1:])
            and is_legal_in_live_request(candidate)
        )
    ]
    if not common_candidate_keys:
        raise RuntimeError("no legal acting-player candidate is shared across hidden-world shards")
    common_candidate_key_set = set(common_candidate_keys)
    common_candidates = [
        candidate for candidate in candidate_lists[0]
        if candidate_key(candidate) in common_candidate_key_set
    ]
    for shard_index, shard in enumerate(shard_reports):
        old_candidates = shard["own_candidates"]
        old_indexes = [
            index for index, candidate in enumerate(old_candidates)
            if candidate_key(candidate) in common_candidate_key_set
        ]
        old_to_new = {old: new for new, old in enumerate(old_indexes)}
        for world in shard.get("world_reports", ()):
            for key in (
                "payoff_matrix", "payoff_standard_error",
                "worst_response_by_own_action",
            ):
                if key in world:
                    world[key] = [world[key][index] for index in old_indexes]
        if "frontier_states" in shard:
            shard["frontier_states"] = [
                {**frontier, "row_index": old_to_new[int(frontier["row_index"])]}
                for frontier in shard["frontier_states"]
                if int(frontier["row_index"]) in old_to_new
            ]
    report = dict(shard_reports[0])
    report["own_candidates"] = common_candidates
    report["world_reports"] = sorted(
        (world for shard in shard_reports for world in shard["world_reports"]),
        key=lambda world: int(world["world_index"]),
    )
    report["world_count"] = sum(int(row["world_count"]) for row in shard_reports)
    report["simulated_turn_advances"] = sum(
        int(row["simulated_turn_advances"]) for row in shard_reports
    )
    # The maximum child engine interval is the parallel critical path; wall
    # time including startup and serialization is reported separately.
    report["elapsed_ms"] = max(float(row["elapsed_ms"]) for row in shard_reports)
    report["transitions_per_second"] = report["simulated_turn_advances"] / max(
        report["elapsed_ms"] / 1000.0, 1e-9
    )
    report["search_subprocess_wall_ms"] = subprocess_wall_ms
    report["search_workers"] = worker_count
    report["hidden_team_hypothesis_count"] = len(hypotheses)
    report["hidden_selection_order_count"] = len(selected_roster_indexes)
    if evaluator is not None:
        report["frontier_states"] = [
            frontier
            for shard in shard_reports
            for frontier in shard.get("frontier_states", ())
        ]
    if evaluator is not None:
        frontier_started = time.perf_counter()
        report, inference_ms = apply_state_value_frontier(report, evaluator)
        report["value_inference_ms"] = inference_ms
        report["frontier_application_ms"] = (
            time.perf_counter() - frontier_started
        ) * 1000.0
    matrix_selection_started = time.perf_counter()
    adversarial_matrix = combine_hidden_world_response_columns([
        world["payoff_matrix"] for world in report["world_reports"]
    ])
    selected_index, matrix_result = select_robust_action(
        adversarial_matrix,
        method=action_selection_method,
        rng=np.random.default_rng(seed),
        iterations=regret_iterations,
    )
    matrix_selection_ms = (time.perf_counter() - matrix_selection_started) * 1000.0
    maximin_scores = adversarial_matrix.min(axis=1).tolist()
    report["robust_scores"] = maximin_scores
    report["matrix_game_method"] = action_selection_method
    report["matrix_game_shape"] = list(adversarial_matrix.shape)
    report["matrix_selection_ms"] = matrix_selection_ms
    report["matrix_game_iterations"] = matrix_result.iterations
    report["matrix_game_value"] = matrix_result.value
    report["matrix_game_exploitability"] = matrix_result.exploitability
    report["mixed_row_strategy"] = matrix_result.row_strategy.tolist()
    report["mixed_column_strategy"] = matrix_result.column_strategy.tolist()
    report["selected_action_index"] = selected_index
    report["selected_action"] = report["own_candidates"][selected_index]
    if evaluator is not None:
        report["frontier_value"] = float(
            matrix_result.value if action_selection_method == "equilibrium"
            else maximin_scores[selected_index]
        )
    translation_started = time.perf_counter()
    # Translate semantic actions against the live legal orders. Native indices
    # depend on a transformed Pokemon's move ordering and can differ between
    # hidden-team hypotheses even when the intended move is the same.
    semantic_actions = report["selected_action"].get("actions")
    if not isinstance(semantic_actions, list) or len(semantic_actions) != 2:
        raise ValueError("selected search candidate has no semantic joint action")
    orders = []
    forced_targetless_positions: list[int] = []
    for position, semantic in enumerate(semantic_actions):
        kind = semantic.get("type")
        if kind == "pass":
            orders.append(PassBattleOrder())
            continue
        matches = []
        for candidate in battle.valid_orders[position]:
            target = getattr(candidate, "order", None)
            if kind == "switch" and hasattr(target, "base_species"):
                if to_id_str(str(target.species)) == to_id_str(str(semantic.get("species", ""))):
                    matches.append(candidate)
        if kind == "move":
            matches = _legal_move_matches(battle, position, semantic)
        if not matches:
            if (
                kind == "move"
                and semantic.get("forced_targetless") is True
                and int(semantic.get("target", 0)) == 0
            ):
                active = battle.active_pokemon[position]
                move = next(
                    (
                        known for known in list(active.moves.values())[:4]
                        if known.id == semantic.get("move_id")
                    ),
                    None,
                ) if active is not None else None
                if move is None:
                    raise ValueError("forced targetless search move is absent from the active Pokemon")
                orders.append(SingleBattleOrder(
                    order=move,
                    mega=bool(semantic.get("mega", False)),
                    terastallize=bool(semantic.get("terastallize", False)),
                    move_target=0,
                ))
                forced_targetless_positions.append(position)
                continue
            legal_moves = [
                {
                    "move_id": getattr(getattr(candidate, "order", None), "id", None),
                    "target": int(getattr(candidate, "move_target", 0)),
                    "mega": bool(getattr(candidate, "mega", False)),
                    "terastallize": bool(getattr(candidate, "terastallize", False)),
                }
                for candidate in battle.valid_orders[position]
                if getattr(getattr(candidate, "order", None), "id", None)
            ]
            raise ValueError(
                f"search action is not legal in the live request: {semantic}; "
                f"legal moves: {legal_moves}"
            )
        orders.append(matches[0])
    order = DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
    # The bridge may need to represent a Showdown forced continuation that
    # poke-env incorrectly expands into target variants. Only those explicitly
    # marked slots bypass poke-env's legal-order membership check; every other
    # order must still be present in the live legal-order list.
    for position, single_order in enumerate(orders):
        if position in forced_targetless_positions:
            if not hasattr(single_order.order, "id") or single_order.move_target != 0:
                raise ValueError("forced targetless search order is malformed")
        elif str(single_order) not in {str(candidate) for candidate in battle.valid_orders[position]}:
            raise ValueError(f"search order is not in the live legal-order list: {single_order}")
    action = DoublesEnv.order_to_action(
        order, battle, fake=bool(forced_targetless_positions), strict=True
    )
    DoublesEnv.action_to_order(
        action, battle, fake=bool(forced_targetless_positions), strict=True
    )
    report["forced_targetless_positions"] = forced_targetless_positions
    remaining_seconds()
    report["search_subprocess_wall_ms"] = subprocess_wall_ms
    report["hidden_hypothesis_setup_ms"] = setup_ms
    report["action_translation_ms"] = (time.perf_counter() - translation_started) * 1000.0
    report["search_adapter_wall_ms"] = (time.perf_counter() - adapter_started) * 1000.0
    return np.asarray(action, dtype=np.int64), report


def choose_sanitized_state_search_action(
    battle: DoubleBattle,
    sanitized_snapshots: Sequence[dict[str, Any]],
    worlds: Sequence[dict[str, Any]],
    *,
    format_id: str,
    showdown_path: str | Path,
    seed: int,
    max_candidates: int = 4,
    max_response_candidates: int | None = None,
    max_damage_response_candidates: int = 1,
    response_selection_strategy: str = "semantic",
    frontier_calibration: tuple[float, float] | None = None,
    chance_samples: int = 1,
    max_chance_samples: int | None = None,
    confidence_multiplier: float = 2.0,
    action_selection_method: str = "equilibrium",
    regret_iterations: int = 1000,
    evaluator: StateValueEvaluator | None = None,
    search_pool: LocalShowdownSearchPool,
    preferred_own_action: Sequence[dict[str, Any]] | None = None,
    return_response_candidates: bool = False,
    timeout_seconds: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Search a live midgame state from sanitized, sampled closed-sheet worlds."""
    adapter_started = time.perf_counter()
    if timeout_seconds <= 0:
        raise ValueError("search decision timeout must be positive")
    if not sanitized_snapshots or len(sanitized_snapshots) != len(worlds):
        raise ValueError("each sanitized state snapshot must have world metadata")
    if search_pool.format_id != format_id or search_pool.showdown_path != Path(showdown_path).resolve():
        raise ValueError("persistent search pool format or local Showdown path mismatch")
    if battle.turn <= 1:
        raise ValueError("sanitized-state search is intended for decisions after the opening")
    if any(battle.force_switch):
        raise ValueError("current-state matrix search does not yet cover forced switches")
    if action_selection_method not in {"maximin", "equilibrium"}:
        raise ValueError("action_selection_method must be 'maximin' or 'equilibrium'")
    response_candidate_limit = (
        max_candidates if max_response_candidates is None
        else int(max_response_candidates)
    )
    if max_candidates < 4 or response_candidate_limit < max_candidates:
        raise ValueError("search needs four own candidates and no fewer response candidates")
    if not 1 <= max_damage_response_candidates <= response_candidate_limit:
        raise ValueError("damage response candidate count must fit the response shortlist")
    if response_selection_strategy not in {"semantic", "slot_coverage"}:
        raise ValueError("response_selection_strategy must be 'semantic' or 'slot_coverage'")
    max_chance_samples = (
        chance_samples if max_chance_samples is None else int(max_chance_samples)
    )
    if max_chance_samples < chance_samples or confidence_multiplier <= 0:
        raise ValueError("maximum chance samples must be at least the minimum, and confidence positive")
    if max_chance_samples > chance_samples and action_selection_method != "maximin":
        raise ValueError("adaptive chance sampling currently requires maximin action selection")

    request: dict[str, Any] = {
        "seed": int(seed),
        "max_candidates": int(max_candidates),
        "max_response_candidates": response_candidate_limit,
        "max_damage_response_candidates": int(max_damage_response_candidates),
        "response_selection_strategy": response_selection_strategy,
        "chance_samples": int(chance_samples),
        "max_chance_samples": int(max_chance_samples),
        "confidence_multiplier": float(confidence_multiplier),
        "sanitized_snapshots": list(sanitized_snapshots),
        "worlds": list(worlds),
        "return_response_candidates": bool(return_response_candidates),
    }
    if preferred_own_action is not None:
        if len(preferred_own_action) != 2:
            raise ValueError("preferred own action must contain two slot actions")
        request["preferred_own_action"] = list(preferred_own_action)
    if evaluator is not None:
        request["return_frontiers"] = True
    shard_reports, subprocess_wall_ms = search_pool.run_requests(
        [request], timeout_seconds=timeout_seconds
    )
    report = dict(shard_reports[0])
    if evaluator is not None:
        frontier_started = time.perf_counter()
        report, inference_ms = apply_state_value_frontier(
            report, evaluator, probability_calibration=frontier_calibration
        )
        report["value_inference_ms"] = inference_ms
        report["frontier_application_ms"] = (
            time.perf_counter() - frontier_started
        ) * 1000.0

    adversarial_matrix = combine_hidden_world_response_columns([
        world["payoff_matrix"] for world in report["world_reports"]
    ])
    selected_index, matrix_result = select_robust_action(
        adversarial_matrix,
        method=action_selection_method,
        rng=np.random.default_rng(seed),
        iterations=regret_iterations,
    )
    maximin_scores = adversarial_matrix.min(axis=1).tolist()
    report["robust_scores"] = maximin_scores
    report["matrix_game_method"] = action_selection_method
    report["matrix_game_shape"] = list(adversarial_matrix.shape)
    report["matrix_game_iterations"] = matrix_result.iterations
    report["matrix_game_value"] = matrix_result.value
    report["matrix_game_exploitability"] = matrix_result.exploitability
    report["mixed_row_strategy"] = matrix_result.row_strategy.tolist()
    report["mixed_column_strategy"] = matrix_result.column_strategy.tolist()
    report["selected_action_index"] = selected_index
    report["selected_action"] = report["own_candidates"][selected_index]
    if evaluator is not None:
        report["frontier_value"] = float(
            matrix_result.value if action_selection_method == "equilibrium"
            else maximin_scores[selected_index]
        )

    translation_started = time.perf_counter()
    semantic_actions = report["selected_action"].get("actions")
    if not isinstance(semantic_actions, list) or len(semantic_actions) != 2:
        raise ValueError("search result has no semantic joint action")
    orders = []
    forced_targetless_positions: list[int] = []
    for position, semantic in enumerate(semantic_actions):
        kind = semantic.get("type")
        if kind == "pass":
            orders.append(PassBattleOrder())
            continue
        matches = []
        for candidate in battle.valid_orders[position]:
            target = getattr(candidate, "order", None)
            if kind == "switch" and hasattr(target, "base_species"):
                if to_id_str(str(target.species)) == to_id_str(str(semantic.get("species", ""))):
                    matches.append(candidate)
        if kind == "move":
            matches = _legal_move_matches(battle, position, semantic)
        if not matches:
            if (
                kind == "move"
                and semantic.get("forced_targetless") is True
                and int(semantic.get("target", 0)) == 0
            ):
                active = battle.active_pokemon[position]
                move = next(
                    (
                        known for known in list(active.moves.values())[:4]
                        if known.id == semantic.get("move_id")
                    ),
                    None,
                ) if active is not None else None
                if move is None:
                    raise ValueError("forced targetless search move is absent from the active Pokemon")
                orders.append(SingleBattleOrder(
                    order=move,
                    mega=bool(semantic.get("mega", False)),
                    terastallize=bool(semantic.get("terastallize", False)),
                    move_target=0,
                ))
                forced_targetless_positions.append(position)
                continue
            legal_moves = [
                {
                    "move_id": getattr(getattr(candidate, "order", None), "id", None),
                    "target": int(getattr(candidate, "move_target", 0)),
                    "mega": bool(getattr(candidate, "mega", False)),
                    "terastallize": bool(getattr(candidate, "terastallize", False)),
                }
                for candidate in battle.valid_orders[position]
                if getattr(getattr(candidate, "order", None), "id", None)
            ]
            raise ValueError(
                f"search action is not legal in the live request: {semantic}; "
                f"legal moves: {legal_moves}"
            )
        orders.append(matches[0])
    order = DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
    for position, single_order in enumerate(orders):
        if position in forced_targetless_positions:
            if not hasattr(single_order.order, "id") or single_order.move_target != 0:
                raise ValueError("forced targetless search order is malformed")
        elif str(single_order) not in {str(candidate) for candidate in battle.valid_orders[position]}:
            raise ValueError(f"search order is not in the live legal-order list: {single_order}")
    action = DoublesEnv.order_to_action(
        order, battle, fake=bool(forced_targetless_positions), strict=True
    )
    DoublesEnv.action_to_order(
        action, battle, fake=bool(forced_targetless_positions), strict=True
    )
    report["forced_targetless_positions"] = forced_targetless_positions
    report["search_subprocess_wall_ms"] = subprocess_wall_ms
    report["search_adapter_wall_ms"] = (time.perf_counter() - adapter_started) * 1000.0
    report["action_translation_ms"] = (time.perf_counter() - translation_started) * 1000.0
    return np.asarray(action, dtype=np.int64), report
