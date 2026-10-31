#!/usr/bin/env python3
"""Collect full-match, closed-sheet action outcomes at a fixed midgame step.

Each candidate is replayed from the same teams, preview, seed, and deterministic
prefix. At that state, Showdown's future battle PRNG is reset to common seeds
across candidate actions, producing repeated chance outcomes for the same
observation/action pair.
"""

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
    _load_policy,
    _load_team_selector,
    _public_preview_may_have_illusion,
    run_game,
)
from scripts.evaluate_counterplay_paired import choose_damage_preview_action
from src.env.headless_showdown import HeadlessShowdownProcess, HeadlessShowdownWorker
from src.belief.closed_sheet import (
    ClosedSheetBeliefTracker,
    possible_selected_rosters,
    sample_showdown_hidden_teams,
)
from src.training.self_play import PolicyAgent
from src.training.team_selection import decode_action, random_action
from poke_env.data.normalize import to_id_str


def public_world_branches(
    *, worker: HeadlessShowdownWorker, data: dict, teams: tuple[list[dict], list[dict]],
    preview: tuple[int, int], format_id: str, showdown_path: Path,
    count: int, seed: int,
) -> list[dict]:
    """Prepare resumed snapshots from sampled worlds conditioned on public facts."""
    if count < 1:
        raise ValueError("public hidden-world count must be positive")
    species = [str(mon.get("species") or mon.get("name")) for mon in teams[1]]
    if len(species) != 6:
        raise ValueError("public hidden-world sampling needs six preview species")
    own_selection = decode_action(int(preview[0]))
    own_preview = list(own_selection.leads + own_selection.back)
    with HeadlessShowdownProcess(
        teams[0], teams[1], format_id=format_id, seed=seed,
        showdown_path=showdown_path, worker=worker,
        snapshot=data["battle_snapshot"],
        parser_snapshot=data["parser_snapshot"],
    ) as match:
        belief = ClosedSheetBeliefTracker(species)
        belief.update_from_battle(match.battles[0])
        leads = belief.opening_leads()
        selections = possible_selected_rosters(
            species, leads, belief.appeared_species()
        )
        preview_index = {to_id_str(name): i for i, name in enumerate(species)}
        selection_indexes = [
            [preview_index[to_id_str(name)] for name in selection]
            for selection in selections
        ]
        if not selection_indexes:
            raise ValueError("no public-compatible opponent selection remains")
        hypotheses = sample_showdown_hidden_teams(
            species, belief.public_facts(), format_id=format_id, count=count,
            seed=seed, showdown_path=showdown_path,
        )
        rng = np.random.default_rng(seed ^ 0x6A09E667)
        try:
            prepared = match.prepare_search_worlds({
                "acting_side": 0,
                "seed": int(seed),
                "own_preview": own_preview,
                "opponent_preview_species": species,
                "revealed_facts": belief.public_facts(),
                "hypotheses": [
                    {"opponent_team": team,
                     "opponent_preview": selection_indexes[int(rng.integers(len(selection_indexes)))]}
                    for team in hypotheses
                ],
            })
        finally:
            match.abort_match()
    return [
        {"initial_battle_snapshot": snapshot,
         "initial_parser_snapshot": data["parser_snapshot"],
         "initial_decision_counts": tuple(data["decision_counts_at_snapshot"]),
         "preview_actions_override": preview,
         "public_hidden_world_index": index,
         "sampled_opponent_team": hypotheses[index]}
        for index, snapshot in enumerate(prepared["snapshots"])
    ]


def strategic_candidate_actions(
    policy_action: list[int],
    damage_action: list[int],
    mask: list[int],
    legal_actions_by_slot: list[list[dict]],
    seed: int,
    candidate_mode: str = "broad",
    extra_policy_actions: list[tuple[str, list[int]]] | None = None,
) -> list[tuple[str, list[int]]]:
    """Add legal tactical alternatives around the same player-view state."""
    width = len(mask) // 2
    legal = [
        set(np.flatnonzero(np.asarray(mask[slot * width:(slot + 1) * width])).tolist())
        for slot in range(2)
    ]
    descriptors = [
        {int(row["action_index"]): row for row in slot_rows}
        for slot_rows in legal_actions_by_slot
    ]

    def joint_choice_is_legal(pair: list[int]) -> bool:
        if any(pair[slot] not in legal[slot] for slot in range(2)):
            return False
        selected = [descriptors[slot].get(int(pair[slot])) for slot in range(2)]
        if any(row is None for row in selected):
            return False
        if all(row.get("kind") == "pass" for row in selected):
            return False
        switches = [
            row.get("species") for row in selected if row.get("kind") == "switch"
        ]
        if len(switches) != len(set(switches)):
            return False
        mechanics = [
            row.get("mechanic") for row in selected
            if row.get("mechanic") in {"mega", "zmove", "dynamax", "tera"}
        ]
        # These team-wide mechanics can be committed by only one active slot.
        if len(mechanics) > 1:
            return False
        return True

    result = [
        ("policy", [int(v) for v in policy_action]),
    ]
    seen = {tuple(pair) for _, pair in result}
    for name, pair in extra_policy_actions or []:
        candidate = [int(v) for v in pair]
        if joint_choice_is_legal(candidate):
            # Keep the alias even when this action is already represented by
            # another candidate. It makes the extra policy's selected action
            # explicit in the saved dataset without rerunning its outcomes.
            result.append((name, candidate))
            seen.add(tuple(candidate))
    damage_pair = [int(v) for v in damage_action]
    if joint_choice_is_legal(damage_pair):
        result.append(("max_damage", damage_pair))
        seen.add(tuple(damage_pair))
    if candidate_mode not in {"broad", "policy_vs_max_damage"}:
        # Keep the two slotwise Max Damage hybrids as an isolated tactical
        # ablation without adding unrelated switch/setup/random candidates.
        for name, pair in (
            ("hybrid_policy0_damage1", [int(policy_action[0]), int(damage_action[1])]),
            ("hybrid_damage0_policy1", [int(damage_action[0]), int(policy_action[1])]),
        ):
            if joint_choice_is_legal(pair) and tuple(pair) not in seen:
                result.append((name, pair))
                seen.add(tuple(pair))
        return result
    if candidate_mode == "policy_vs_max_damage":
        return result
    for name, pair in (
        ("hybrid_policy0_damage1", [int(policy_action[0]), int(damage_action[1])]),
        ("hybrid_damage0_policy1", [int(damage_action[0]), int(policy_action[1])]),
    ):
        if joint_choice_is_legal(pair):
            result.append((name, pair))
            seen.add(tuple(pair))
    categories = ("switch", "protect", "setup", "redirection", "spread", "status")
    for category in categories:
        for slot in range(2):
            choices = [
                row for row in legal_actions_by_slot[slot]
                if category in row.get("categories", ())
                and int(row["action_index"]) in legal[slot]
            ]
            if not choices:
                continue
            choice = choices[0]
            pair = [int(policy_action[0]), int(policy_action[1])]
            pair[slot] = int(choice["action_index"])
            if not joint_choice_is_legal(pair):
                continue
            key = tuple(pair)
            if key in seen:
                continue
            seen.add(key)
            move_or_species = choice.get("move_id", choice.get("species", "pass"))
            result.append((f"{category}_slot{slot}_{move_or_species}", pair))

    rng = np.random.default_rng(seed)
    alternatives = [(int(a), int(b)) for a in legal[0] for b in legal[1]]
    rng.shuffle(alternatives)
    random_added = 0
    for pair in alternatives:
        if not joint_choice_is_legal(list(pair)):
            continue
        key = tuple(pair)
        if key in seen:
            continue
        seen.add(key)
        result.append((f"legal_perturbation_{random_added}", list(pair)))
        random_added += 1
        if random_added == 2:
            break
    return result


def policy_argmax_legal_joint_action(policy, observation: list[float],
                                     mask: list[int],
                                     legal_actions_by_slot: list[list[dict]]) -> list[int]:
    """Score the complete legal joint-action set under a policy checkpoint."""
    width = len(mask) // 2
    legal = [
        set(np.flatnonzero(np.asarray(mask[slot * width:(slot + 1) * width])).tolist())
        for slot in range(2)
    ]
    descriptors = [
        {int(row["action_index"]): row for row in slot_rows}
        for slot_rows in legal_actions_by_slot
    ]
    pairs = []
    for first in legal[0]:
        for second in legal[1]:
            selected = [descriptors[slot].get(action)
                        for slot, action in enumerate((first, second))]
            if any(row is None for row in selected):
                continue
            if all(row.get("kind") == "pass" for row in selected):
                continue
            switches = [row.get("species") for row in selected
                        if row.get("kind") == "switch"]
            if len(switches) != len(set(switches)):
                continue
            mechanics = [row.get("mechanic") for row in selected
                         if row.get("mechanic") in {"mega", "zmove", "dynamax", "tera"}]
            if len(mechanics) > 1:
                continue
            pairs.append([int(first), int(second)])
    if not pairs:
        raise ValueError("no legal joint native actions in captured state")
    scores = policy.log_prob_native_actions(
        np.asarray(observation, dtype=np.float32),
        np.asarray(mask, dtype=np.int8), pairs,
    )
    return pairs[int(np.argmax(np.asarray(scores, dtype=np.float64)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--candidate-policy-checkpoint", type=Path,
        help="add this policy's full-legal-action argmax as a counterfactual candidate",
    )
    parser.add_argument(
        "--player-strategy", choices=("policy", "public_counterplay"),
        default="public_counterplay",
        help="decision policy whose state/action will seed the counterfactual branches",
    )
    parser.add_argument(
        "--opponent-checkpoint", type=Path,
        help="alternate this frozen policy with Max Damage across state samples",
    )
    parser.add_argument(
        "--additional-opponent-checkpoint", action="append", default=[],
        metavar="NAME=PATH",
        help="add a named PPO response (strategy label will be NAME)",
    )
    parser.add_argument(
        "--team-selection-checkpoint", type=Path,
        help="apply this preview selector to the player policy and policy opponents",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--showdown-path", type=Path, default=Path(
        "/home/justin/p/poke_learning/pokemon-showdown"))
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--seed", type=int, default=2026102802)
    parser.add_argument(
        "--cluster-offset", type=int, default=0,
        help="global team-cluster index for this independent collection shard",
    )
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--states-per-cluster", type=int, default=2)
    parser.add_argument("--decision-step", type=int, default=5)
    parser.add_argument("--chance-repeats", type=int, default=4)
    parser.add_argument(
        "--hidden-worlds", type=int, default=0,
        help=("sample an even number of public-compatible hidden-team worlds per state; "
              "0 keeps the legacy actual-world branches"),
    )
    parser.add_argument(
        "--candidate-mode", choices=(
            "broad", "policy_vs_max_damage", "policy_vs_max_damage_and_hybrids"
        ),
        default="broad",
        help="evaluate all tactical alternatives or only the policy/Max Damage pair",
    )
    parser.add_argument(
        "--paired-opponent-responses", action="store_true",
        help=("evaluate every candidate from each captured state against both "
              "Max Damage and the supplied frozen policy, using common chance seeds"),
    )
    parser.add_argument(
        "--include-public-counterplay-response", action="store_true",
        help=("when paired responses are enabled, also evaluate each action "
              "against the closed-sheet Public Counterplay policy"),
    )
    parser.add_argument(
        "--include-random-response", action="store_true",
        help="when paired responses are enabled, also evaluate random legal play",
    )
    parser.add_argument(
        "--include-legacy-counterplay-response", action="store_true",
        help="when paired responses are enabled, also evaluate legacy Counterplay",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--preview-strategy", choices=("random", "damage"), default="random",
        help="team preview for the counterplay player; Max Damage keeps random preview",
    )
    args = parser.parse_args()
    if (args.clusters < 1 or args.cluster_offset < 0
            or args.states_per_cluster < 1 or args.decision_step < 1
            or args.chance_repeats < 1 or args.hidden_worlds < 0):
        parser.error("clusters/states/repeats must be positive, hidden-worlds nonnegative, and decision-step at least one")
    if args.hidden_worlds == 1 or args.hidden_worlds % 2:
        parser.error("--hidden-worlds must be zero or an even number of at least two")
    if args.paired_opponent_responses and args.opponent_checkpoint is None:
        parser.error("--paired-opponent-responses requires --opponent-checkpoint")
    if ((args.include_public_counterplay_response or args.include_random_response
         or args.include_legacy_counterplay_response)
            and not args.paired_opponent_responses):
        parser.error("additional response classes require paired responses")
    if args.player_strategy == "policy" and args.checkpoint is None:
        parser.error("--checkpoint is required when --player-strategy=policy")
    additional_policy_paths = {}
    for value in args.additional_opponent_checkpoint:
        if "=" not in value:
            parser.error("additional opponent checkpoints must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name.startswith("policy_") or name in additional_policy_paths:
            parser.error("additional opponent names must be unique and start with policy_")
        if name in {"policy", "max_damage", "public_counterplay", "counterplay", "random"}:
            parser.error(f"reserved additional opponent name: {name}")
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            parser.error(f"additional opponent checkpoint not found: {checkpoint_path}")
        additional_policy_paths[name] = checkpoint_path
    if additional_policy_paths and not args.paired_opponent_responses:
        parser.error("additional opponent checkpoints require --paired-opponent-responses")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The preview helper import also loads evaluate_native, which may
        # already have initialized PyTorch's interop pool. It sets the same
        # OMP-derived thread count; do not reconfigure it after initialization.
        if torch.get_num_interop_threads() != 1:
            raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(".runner.jsonl")
    if temp_output.exists():
        temp_output.unlink()
    model = _load_policy(args.checkpoint, args.device) if args.checkpoint else None
    candidate_model = (
        _load_policy(args.candidate_policy_checkpoint, args.device)
        if args.candidate_policy_checkpoint else None
    )
    opponent_model = _load_policy(args.opponent_checkpoint, args.device)
    named_opponent_models = {"policy": opponent_model}
    named_opponent_models.update({
        name: _load_policy(path, args.device)
        for name, path in additional_policy_paths.items()
    })
    team_selector, team_selector_config = _load_team_selector(
        args.team_selection_checkpoint
    )
    rows: list[dict] = []
    matches_executed = 0
    missing_states = 0
    missing_state_reasons: dict[str, int] = {}
    rejected_candidates = 0
    rejected_candidate_details: list[dict] = []
    unsupported_world_replays: list[dict] = []
    hidden_worlds_prepared = 0
    hidden_world_preparation_seconds = 0.0
    started = time.perf_counter()
    with HeadlessShowdownWorker(args.showdown_path) as worker:
        for local_cluster in range(args.clusters):
            cluster = args.cluster_offset + local_cluster
            cluster_seed = args.seed + cluster * 10007
            teams = (worker.generate_team(cluster_seed, args.format),
                     worker.generate_team(cluster_seed + 1, args.format))
            preview_override = None
            if args.preview_strategy == "damage":
                if team_selector is not None:
                    raise ValueError("damage preview cannot be combined with a learned team selector")
                preview_override = (
                    choose_damage_preview_action(teams[0], teams[1]),
                    random_action(np.random.default_rng(cluster_seed + 2)),
                )
            for state_repeat in range(args.states_per_cluster):
                state_seed = cluster_seed + state_repeat * 1009
                opponent_strategy = (
                    "policy"
                    if opponent_model is not None and (cluster + state_repeat) % 2
                    else "max_damage"
                )
                common = dict(
                    game=0, seed=state_seed, format_id=args.format,
                    showdown_path=args.showdown_path, model=model,
                    opponent_model=None, player_strategy="policy",
                    opponent_strategy=opponent_strategy, temperature=1.0,
                    device=args.device, telemetry=None, worker=worker,
                    deterministic_policy=True, teams_override=teams,
                    persist_record=False,
                    capture_policy_action_log_prob=True,
                    capture_legal_joint_action_log_probs=True,
                    capture_battle_snapshot=True,
                    team_selection_model=team_selector,
                    team_selection_config=team_selector_config,
                    opponent_team_selection_model=(
                        team_selector if opponent_strategy == "policy" else None
                    ),
                    opponent_team_selection_config=(
                        team_selector_config if opponent_strategy == "policy" else None
                    ),
                )
                common["player_strategy"] = args.player_strategy
                if opponent_strategy == "policy":
                    common["opponent_model"] = opponent_model
                policy = run_game(output_path=temp_output,
                                  capture_action_step=args.decision_step,
                                  preview_actions_override=preview_override,
                                  **common)
                matches_executed += 1
                p = policy["counterfactual_action_data"]
                if p is None:
                    missing_states += 1
                    missing_state_reasons["decision_not_reached"] = missing_state_reasons.get("decision_not_reached", 0) + 1
                    continue
                if p.get("battle_snapshot") is None:
                    raise RuntimeError("counterfactual prefix did not produce a battle snapshot")
                preview = tuple(int(v) for v in policy["preview_actions"])
                branch = dict(
                    initial_battle_snapshot=p["battle_snapshot"],
                    initial_parser_snapshot=p["parser_snapshot"],
                    initial_decision_counts=tuple(p["decision_counts_at_snapshot"]),
                    preview_actions_override=preview,
                )
                if args.hidden_worlds:
                    snapshot_request_state = p["battle_snapshot"].get("requestState")
                    if snapshot_request_state != "move":
                        missing_states += 1
                        reason = f"request_{snapshot_request_state or 'unknown'}"
                        missing_state_reasons[reason] = missing_state_reasons.get(reason, 0) + 1
                        continue
                    public_species = [
                        str(member.get("species") or member.get("name"))
                        for member in teams[1]
                    ]
                    if _public_preview_may_have_illusion(public_species):
                        missing_states += 1
                        missing_state_reasons["possible_illusion_preview"] = missing_state_reasons.get("possible_illusion_preview", 0) + 1
                        continue
                    world_started = time.perf_counter()
                    try:
                        branches = public_world_branches(
                            worker=worker, data=p, teams=teams, preview=preview,
                            format_id=args.format, showdown_path=args.showdown_path,
                            count=args.hidden_worlds,
                            seed=args.seed + cluster * 10007 + state_repeat * 1009 + 0x51ED,
                        )
                    except RuntimeError as error:
                        if "Transform" not in str(error):
                            raise
                        missing_states += 1
                        missing_state_reasons["public_transform_state"] = missing_state_reasons.get("public_transform_state", 0) + 1
                        continue
                    hidden_world_preparation_seconds += time.perf_counter() - world_started
                    hidden_worlds_prepared += len(branches)
                else:
                    branches = [branch]
                branch_common = dict(common)
                branch_common["capture_battle_snapshot"] = False
                damage_common = dict(branch_common)
                damage_common["player_strategy"] = "max_damage"
                damage = run_game(
                    output_path=temp_output,
                    capture_action_step=args.decision_step, **damage_common, **branch,
                )
                matches_executed += 1
                d = damage["counterfactual_action_data"]
                if p is None or d is None:
                    missing_states += 1
                    continue
                if (p["player1_decision_index"] != args.decision_step
                        or d["player1_decision_index"] != args.decision_step
                        or p["observation"] != d["observation"]
                        or p["legal_action_mask"] != d["legal_action_mask"]
                        or policy["preview_actions"] != damage["preview_actions"]):
                    raise RuntimeError(
                        f"policy/Max Damage reference state mismatch at team {cluster}, "
                        f"repeat {state_repeat}: p1_index={p['player1_decision_index']} "
                        f"d1_index={d['player1_decision_index']}, "
                        f"observation_equal={p['observation'] == d['observation']}, "
                        f"mask_equal={p['legal_action_mask'] == d['legal_action_mask']}, "
                        f"preview_equal={policy['preview_actions'] == damage['preview_actions']}, "
                        f"obs_diff={[i for i, (a, b) in enumerate(zip(p['observation'], d['observation'])) if a != b][:24]}, "
                        f"obs_values={[(i, p['observation'][i], d['observation'][i]) for i in range(min(len(p['observation']), len(d['observation']))) if p['observation'][i] != d['observation'][i]][:24]}, "
                        f"mask_diff={[i for i, (a, b) in enumerate(zip(p['legal_action_mask'], d['legal_action_mask'])) if a != b][:24]}"
                    )
                candidates = strategic_candidate_actions(
                    p["action"], d["action"], p["legal_action_mask"],
                    p.get("legal_actions_by_slot", [[], []]), state_seed,
                    candidate_mode=args.candidate_mode,
                    extra_policy_actions=(
                        [("candidate_policy_argmax", policy_argmax_legal_joint_action(
                            PolicyAgent(
                                candidate_model, teams[0],
                                [{"species": str(member.get("species") or member.get("name"))}
                                 for member in teams[1]
                                 if member.get("species") or member.get("name")],
                                deterministic=True,
                                temperature=1.0, device=args.device,
                                public_context_features=(
                                    np.asarray(p["public_context_features"], dtype=np.float32)
                                    if p.get("public_context_features") is not None else None
                                ),
                            ), p["observation"], p["legal_action_mask"],
                            p.get("legal_actions_by_slot", [[], []]),
                        ))]
                        if candidate_model is not None else None
                    ),
                )
                action_log_probs = {
                    tuple(int(v) for v in row["action"]): float(row["log_probability"])
                    for row in (p.get("legal_joint_action_log_probabilities") or [])
                }
                response_names = ["max_damage", "policy"] if args.paired_opponent_responses else [opponent_strategy]
                if args.paired_opponent_responses and args.include_public_counterplay_response:
                    response_names.append("public_counterplay")
                if args.paired_opponent_responses and args.include_random_response:
                    response_names.append("random")
                if args.paired_opponent_responses and args.include_legacy_counterplay_response:
                    response_names.append("counterplay")
                if args.paired_opponent_responses:
                    response_names.extend(additional_policy_paths)
                outcome_cache: dict[tuple[int, int], dict[str, list[int]]] = {}
                rejected_action_keys: set[tuple[int, int]] = set()
                state_row_start = len(rows)
                state_invalid_replay = False
                for name, action in candidates:
                    key = (int(action[0]), int(action[1]))
                    if key in rejected_action_keys:
                        continue
                    if key in outcome_cache:
                        outcomes_by_opponent = outcome_cache[key]
                    else:
                        outcomes_by_opponent = {}
                        action_rejected = False
                        for response_name in response_names:
                            outcomes = []
                            response_common = dict(branch_common)
                            is_policy_response = response_name in named_opponent_models
                            response_common["opponent_strategy"] = (
                                "policy" if is_policy_response else response_name
                            )
                            response_common["opponent_model"] = named_opponent_models.get(response_name)
                            response_common["opponent_team_selection_model"] = (
                                team_selector if is_policy_response else None
                            )
                            response_common["opponent_team_selection_config"] = (
                                team_selector_config if is_policy_response else None
                            )
                            for world_branch in branches:
                              for chance_repeat in range(args.chance_repeats):
                                chance_seed = (
                                    args.seed + cluster * 10007 + state_repeat * 1009
                                    + 1000003 + chance_repeat * 7919
                                )
                                branch_args = {
                                    key: value for key, value in world_branch.items()
                                    if key not in {
                                        "public_hidden_world_index", "sampled_opponent_team"
                                    }
                                }
                                world_response_common = dict(response_common)
                                if args.hidden_worlds:
                                    world_response_common["teams_override"] = (
                                        teams[0], world_branch["sampled_opponent_team"]
                                    )
                                try:
                                    result = run_game(
                                        output_path=temp_output,
                                        first_turn_forced_action=action,
                                        forced_action_step=args.decision_step,
                                        capture_action_step=args.decision_step,
                                        reseed_random_at_decision=(
                                            args.decision_step, chance_seed
                                        ),
                                        **world_response_common, **branch_args,
                                    )
                                except RuntimeError as error:
                                    if "No legal action available" not in str(error):
                                        raise
                                    state_invalid_replay = True
                                    unsupported_world_replays.append({
                                        "team_cluster": cluster,
                                        "state_repeat": state_repeat,
                                        "hidden_world_index": world_branch.get(
                                            "public_hidden_world_index", 0
                                        ),
                                        "opponent_strategy": response_name,
                                        "candidate": name,
                                        "action": list(key),
                                        "unsupported_replay": "no_legal_action_in_resumed_branch",
                                    })
                                    break
                                matches_executed += 1
                                replay = result["counterfactual_action_data"]
                                if replay is not None and replay.get("forced_action_accepted") is False:
                                    rejected_candidates += 1
                                    legal_descriptors = p.get("legal_actions_by_slot", [[], []])
                                    chosen_descriptors = []
                                    for slot, action_index in enumerate(key):
                                        chosen_descriptors.append(next(
                                            (row for row in legal_descriptors[slot]
                                             if int(row["action_index"]) == action_index),
                                            None,
                                        ))
                                    rejected_candidate_details.append({
                                        "team_cluster": cluster,
                                        "state_repeat": state_repeat,
                                        "hidden_world_index": world_branch.get("public_hidden_world_index", 0),
                                        "opponent_strategy": response_name,
                                        "candidate": name,
                                        "action": list(key),
                                        "selected_action_descriptors": chosen_descriptors,
                                        "showdown_rejection": replay.get(
                                            "forced_action_rejection_error"
                                        ),
                                    })
                                    action_rejected = True
                                    break
                                if replay is not None and [
                                    int(value) for value in replay.get("action", ())
                                ] != list(key):
                                    raise RuntimeError(
                                        f"forced replay selected {replay.get('action')} instead of "
                                        f"requested {list(key)} at team {cluster}, "
                                        f"repeat {state_repeat}, opponent response {response_name}"
                                    )
                                if (replay is None
                                        or replay["player1_decision_index"] != args.decision_step
                                        or replay["observation"] != p["observation"]
                                        or replay["legal_action_mask"] != p["legal_action_mask"]):
                                    raise RuntimeError(
                                        f"forced replay diverged before decision at team {cluster}, "
                                        f"repeat {state_repeat}, action {action}, "
                                        f"opponent response {response_name}"
                                    )
                                outcomes.append(int(result["winner"] == "Headless P1"))
                              if action_rejected:
                                break
                              if state_invalid_replay:
                                break
                            if state_invalid_replay:
                                break
                            if action_rejected:
                                break
                            outcomes_by_opponent[response_name] = outcomes
                        if action_rejected:
                            rejected_action_keys.add(key)
                            continue
                        if state_invalid_replay:
                            break
                        outcome_cache[key] = outcomes_by_opponent
                    if state_invalid_replay:
                        break
                    outcomes = [
                        outcome
                        for response_outcomes in outcomes_by_opponent.values()
                        for outcome in response_outcomes
                    ]
                    rows.append({
                        "team_cluster": cluster,
                        "cluster": cluster * args.states_per_cluster + state_repeat,
                        "team_seed": cluster_seed,
                        "state_repeat": state_repeat,
                        "decision_step": args.decision_step,
                        "candidate": name,
                        "opponent_strategy": (
                            "paired" if args.paired_opponent_responses
                            else opponent_strategy
                        ),
                        "action": action,
                        "policy_log_probability": action_log_probs.get(
                            tuple(int(v) for v in action)
                        ),
                        "selected_action_log_probability": p.get(
                            "policy_log_probability_of_selected_action"
                        ),
                        "observation": p["observation"],
                        "legal_action_mask": p["legal_action_mask"],
                        # Preserve policy inputs for direct actor updates. The
                        # own set is observable; the opponent receives species
                        # only, as revealed in the closed-sheet preview.
                        "player_team": teams[0],
                        "opponent_public_team": [
                            {"species": str(member.get("species") or member.get("name"))}
                            for member in teams[1]
                            if member.get("species") or member.get("name")
                        ],
                        "native_action_features": p.get("native_action_features"),
                        "public_context_features": p.get("public_context_features"),
                        "outcomes": outcomes,
                        **({"outcomes_by_opponent": outcomes_by_opponent}
                           if args.paired_opponent_responses else {}),
                        "outcome_rate": float(np.mean(outcomes)),
                        "wins": int(sum(outcomes)),
                        "repeats": len(outcomes),
                        "public_hidden_worlds": len(branches),
                        "chance_repeats_per_world": args.chance_repeats,
                    })
                if state_invalid_replay:
                    del rows[state_row_start:]
                    missing_states += 1
                    missing_state_reasons["invalid_world_replay_no_legal_action"] = (
                        missing_state_reasons.get("invalid_world_replay_no_legal_action", 0) + 1
                    )
                    continue
            if (local_cluster + 1) % 8 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(json.dumps({
                    "clusters_completed": local_cluster + 1,
                    "states_captured": len({(r['team_cluster'], r['state_repeat']) for r in rows}),
                    "matches_executed": matches_executed,
                    "matches_per_second": matches_executed / elapsed,
                    "missing_states": missing_states,
                    "missing_state_reasons": missing_state_reasons,
                }), flush=True)
    if temp_output.exists():
        temp_output.unlink()

    by_name: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        group = "random_legal" if row["candidate"].startswith("legal_perturbation_") else row["candidate"]
        by_name.setdefault(group, {}).setdefault(row["team_cluster"], []).append(float(row["outcome_rate"]))
    grouped = {name: {cluster: float(np.mean(vals)) for cluster, vals in cluster_rows.items()}
               for name, cluster_rows in by_name.items()}
    rng = np.random.default_rng(args.seed ^ 0x5EEDC0DE)
    summary: dict[str, dict] = {}
    for name, values_by_cluster in grouped.items():
        keys = sorted(values_by_cluster)
        values = np.asarray([values_by_cluster[k] for k in keys])
        boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        summary[name] = {
            "team_clusters": int(values.size),
            "mean_win_rate": float(values.mean()),
            "cluster_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    base = grouped.get("policy", {})
    for name, values_by_cluster in grouped.items():
        keys = sorted(set(base) & set(values_by_cluster))
        delta = np.asarray([values_by_cluster[k] - base[k] for k in keys])
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        summary[name]["paired_delta_vs_policy"] = float(delta.mean())
        summary[name]["paired_cluster_bootstrap_95_ci"] = np.quantile(
            boot, [0.025, 0.975]
        ).tolist()

    stratified: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        candidate = "random_legal" if row["candidate"].startswith("legal_perturbation_") else row["candidate"]
        if row.get("outcomes_by_opponent"):
            for opponent_name, outcomes in row["outcomes_by_opponent"].items():
                key = f"{opponent_name}:{candidate}"
                stratified.setdefault(key, {}).setdefault(row["team_cluster"], []).append(
                    float(np.mean(outcomes))
                )
        else:
            key = f"{row['opponent_strategy']}:{candidate}"
            stratified.setdefault(key, {}).setdefault(row["team_cluster"], []).append(
                float(row["outcome_rate"])
            )
    stratified_summary: dict[str, dict] = {}
    for name, cluster_rows in stratified.items():
        values = np.asarray([np.mean(cluster_rows[c]) for c in sorted(cluster_rows)])
        boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        stratified_summary[name] = {
            "team_clusters": int(values.size),
            "mean_win_rate": float(values.mean()),
            "cluster_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    for opponent_name in ("max_damage", "policy"):
        base_by_cluster = stratified.get(f"{opponent_name}:policy", {})
        for name, cluster_rows in stratified.items():
            if not name.startswith(f"{opponent_name}:"):
                continue
            candidate = name.split(":", 1)[1]
            values_by_cluster = {
                cluster: float(np.mean(rows_for_cluster))
                for cluster, rows_for_cluster in cluster_rows.items()
            }
            common_clusters = sorted(set(base_by_cluster) & set(values_by_cluster))
            if common_clusters:
                delta = np.asarray([
                    values_by_cluster[c] - float(np.mean(base_by_cluster[c]))
                    for c in common_clusters
                ])
                boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
                stratified_summary[name]["paired_delta_vs_policy"] = float(delta.mean())
                stratified_summary[name]["paired_cluster_bootstrap_95_ci"] = np.quantile(
                    boot, [0.025, 0.975]
                ).tolist()

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "candidate_policy_checkpoint": (
            str(args.candidate_policy_checkpoint)
            if args.candidate_policy_checkpoint else None
        ),
        "player_strategy": args.player_strategy,
        "format": args.format,
        "seed": args.seed,
        "decision_step": args.decision_step,
        "opponent_checkpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
        "opponent_mix": (
            "paired_future_responses_max_damage_and_frozen_policy"
            if args.paired_opponent_responses
            else "max_damage_and_frozen_policy_equal_by_state"
            if opponent_model is not None
            else "max_damage_only"
        ),
        "clusters_requested": args.clusters,
        "states_per_cluster": args.states_per_cluster,
        "chance_repeats": args.chance_repeats,
        "hidden_worlds_requested": args.hidden_worlds,
        "hidden_worlds_prepared": hidden_worlds_prepared,
        "hidden_world_preparation_seconds": hidden_world_preparation_seconds,
        "hidden_world_prior": (
            "local_showdown_random_champions_conditioned_on_public_facts"
            if args.hidden_worlds else "actual_captured_battle_world"
        ),
        "candidate_mode": args.candidate_mode,
        "preview_strategy": args.preview_strategy,
        "states_captured": len({(r["team_cluster"], r["state_repeat"]) for r in rows}),
        "matches_executed": matches_executed,
        "elapsed_seconds": elapsed,
        "matches_per_second": matches_executed / max(elapsed, 1e-9),
        "missing_states": missing_states,
        "missing_state_reasons": missing_state_reasons,
        "rejected_candidates": rejected_candidates,
        "rejected_candidate_details": rejected_candidate_details,
        "unsupported_world_replays": unsupported_world_replays,
        "candidate_summary": summary,
        "candidate_summary_by_opponent": stratified_summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
