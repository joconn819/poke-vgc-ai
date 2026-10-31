"""Evaluate the support-aware counterplay baseline on side-swapped teams."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from scripts.evaluate_native import evaluate
from scripts.run_showdown_self_play import generate_showdown_champions_team_data
from src.training.team_selection import ACTIONS, TeamSelectionPolicy, random_action, team_features

_GEN9_DATA = GenData.from_gen(9)
_PRIOR_DATA = json.loads(
    (Path(__file__).resolve().parents[1] / "src/features/randomteam_full_set_priors.json")
    .read_text(encoding="utf-8")
)


def choose_damage_preview_action(
    own_team: list[dict], opponent_team: list[dict], *,
    lead_weight: float = 0.35, defense_weight: float = 0.08,
    role_synergy_weight: float = 0.0,
    damage_model: str = "base",
) -> int:
    """Choose four Pokemon and leads using only the two visible preview rosters.

    This deliberately simple heuristic ranks own sets by their best expected
    damage against the opposing preview species, then rewards useful lead
    pairs. It is an experiment, not a claim of optimal preview play.
    """
    chart = _GEN9_DATA.type_chart
    if damage_model not in {"base", "known_set"}:
        raise ValueError("damage_model must be 'base' or 'known_set'")

    nature_table = {
        "lonely": (1.1, 0.9, 1.0, 1.0, 1.0), "brave": (1.1, 1.0, 1.0, 1.0, 0.9),
        "adamant": (1.1, 1.0, 0.9, 1.0, 1.0), "naughty": (1.1, 1.0, 1.0, 0.9, 1.0),
        "bold": (0.9, 1.1, 1.0, 1.0, 1.0), "relaxed": (1.0, 1.1, 1.0, 1.0, 0.9),
        "impish": (1.0, 1.1, 0.9, 1.0, 1.0), "lax": (1.0, 1.1, 1.0, 0.9, 1.0),
        "timid": (0.9, 1.0, 1.0, 1.0, 1.1), "hasty": (1.0, 0.9, 1.0, 1.0, 1.1),
        "jolly": (1.0, 1.0, 0.9, 1.0, 1.1), "naive": (1.0, 1.0, 1.0, 0.9, 1.1),
        "modest": (0.9, 1.0, 1.1, 1.0, 1.0), "mild": (1.0, 0.9, 1.1, 1.0, 1.0),
        "quiet": (1.0, 1.0, 1.1, 1.0, 0.9), "rash": (1.0, 1.0, 1.1, 0.9, 1.0),
        "calm": (0.9, 1.0, 1.0, 1.1, 1.0), "gentle": (1.0, 0.9, 1.0, 1.1, 1.0),
        "sassy": (1.0, 1.0, 1.0, 1.1, 0.9), "careful": (1.0, 1.0, 0.9, 1.1, 1.0),
    }

    def known_stats(pokemon: dict, dex: dict) -> dict[str, float]:
        base = dex.get("baseStats", {})
        if damage_model == "base":
            return {key: float(base.get(key, 80)) for key in ("hp", "atk", "def", "spa", "spd", "spe")}
        level = float(pokemon.get("level", 50))
        evs, ivs = pokemon.get("evs", {}), pokemon.get("ivs", {})
        nature = nature_table.get(to_id_str(pokemon.get("nature", "")), (1, 1, 1, 1, 1))
        nature_stat = {"atk": 0, "def": 1, "spa": 2, "spd": 3, "spe": 4}
        result = {}
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            raw = (2 * float(base.get(stat, 80)) + float(ivs.get(stat, 31))
                   + float(evs.get(stat, 0)) // 4) * level / 100
            value = raw + level + 10 if stat == "hp" else (raw + 5) * nature[nature_stat[stat]]
            result[stat] = float(int(value))
        return result

    def species_data(pokemon: dict) -> tuple[dict, list[str], list[str]]:
        species_id = to_id_str(pokemon.get("species") or pokemon.get("name", ""))
        dex = _GEN9_DATA.pokedex.get(species_id, {})
        return dex, list(dex.get("types", ())), list(pokemon.get("moves", ()))

    own = [species_data(pokemon) for pokemon in own_team]
    foes = [species_data(pokemon) for pokemon in opponent_team]
    own_moves = [
        {to_id_str(move) for move in pokemon.get("moves", ())}
        for pokemon in own_team
    ]
    own_abilities = [to_id_str(pokemon.get("ability", "")) for pokemon in own_team]
    own_speeds = [
        float(dex.get("baseStats", {}).get("spe", 80)) for dex, _, _ in own
    ]

    own_stats = [known_stats(pokemon, data[0]) for pokemon, data in zip(own_team, own)]
    def max_damage_fraction(attacker_index: int, attacker: tuple, defender: tuple) -> float:
        dex, types, moves = attacker
        foe_dex, foe_types, _ = defender
        foe_stats = foe_dex.get("baseStats", {})
        level = 50
        hp = max((2 * float(foe_stats.get("hp", 80)) + 31) * level / 100 + level + 10, 1)
        best = 0.0
        for move_name in moves:
            move = _GEN9_DATA.moves.get(to_id_str(move_name), {})
            power = float(move.get("basePower", 0) or 0)
            if power <= 0 or move.get("category") not in {"Physical", "Special"}:
                continue
            category = move["category"]
            attack_stat = float(own_stats[attacker_index]["atk" if category == "Physical" else "spa"])
            defense_stat = max(float(foe_stats.get("def" if category == "Physical" else "spd", 80)), 1)
            damage = ((2 * level / 5 + 2) * power * attack_stat / defense_stat / 50 + 2)
            move_type = str(move.get("type", "")).upper()
            if move_type in {str(value).upper() for value in types}:
                damage *= 1.5
            for foe_type in foe_types:
                damage *= float(chart.get(move_type, {}).get(str(foe_type).upper(), 1.0))
            accuracy = move.get("accuracy", 1.0)
            if isinstance(accuracy, bool):
                damage *= float(accuracy)
            elif isinstance(accuracy, (int, float)):
                damage *= float(accuracy) / 100.0
            if str(move.get("target", "")).lower() in {"alladjacentfoes", "alladjacent", "all"}:
                damage *= 0.75
            if damage_model == "known_set":
                ability = to_id_str(own_team[attacker_index].get("ability", ""))
                item = to_id_str(own_team[attacker_index].get("item", ""))
                if ability in {"hugepower", "purepower"} and category == "Physical":
                    damage *= 2.0
                elif ability in {"overgrow", "blaze", "torrent", "swarm"}:
                    threshold = {"overgrow": ("grass",), "blaze": ("fire",), "torrent": ("water",), "swarm": ("bug",)}[ability]
                    if move_type.lower() in threshold:
                        damage *= 1.5
                if item == "lifeorb":
                    damage *= 1.3
                elif item == "choiceband" and category == "Physical":
                    damage *= 1.5
                elif item == "choicespecs" and category == "Special":
                    damage *= 1.5
                elif item == "expertbelt" and any(
                    float(chart.get(move_type, {}).get(str(foe_type).upper(), 1.0)) > 1.0
                    for foe_type in foe_types
                ):
                    damage *= 1.2
            best = max(best, min(damage / hp, 1.0))
        return best

    offense = [
        float(np.mean([max_damage_fraction(i, pokemon, foe) for foe in foes]))
        for i, pokemon in enumerate(own)
    ]
    defense = []
    for _, own_types, _ in own:
        # Unknown opposing moves are represented by opposing STAB types.
        incoming = [
            float(np.prod([
                chart.get(str(foe_type).upper(), {}).get(str(own_type).upper(), 1.0)
                for own_type in own_types
            ]))
            for _, foe_types, _ in foes
            for foe_type in foe_types
        ]
        defense.append(-float(np.mean(incoming)) if incoming else 0.0)

    best_index = 0
    best_score = -float("inf")
    for index, action in enumerate(ACTIONS):
        selected = action.leads + action.back
        lead_a, lead_b = action.leads
        score = sum(offense[i] + float(defense_weight) * defense[i] for i in selected)
        score += float(lead_weight) * (offense[lead_a] + offense[lead_b])
        lead_moves = (own_moves[lead_a], own_moves[lead_b])
        lead_abilities = (own_abilities[lead_a], own_abilities[lead_b])
        lead_speeds = (own_speeds[lead_a], own_speeds[lead_b])
        partner_damage = (offense[lead_b], offense[lead_a])
        synergy = 0.0
        if any("fakeout" in moves for moves in lead_moves):
            score += 0.05 * max(offense[lead_a], offense[lead_b])
        if any(moves & {"ragepowder", "followme"} for moves in lead_moves):
            score += 0.05 * max(offense[lead_a], offense[lead_b])
        for slot, (moves, ability, speed, partner_offense) in enumerate(zip(
            lead_moves, lead_abilities, lead_speeds, partner_damage
        )):
            partner_speed = lead_speeds[1 - slot]
            partner_moves = lead_moves[1 - slot]
            if "fakeout" in moves:
                synergy += 0.20 * partner_offense
            if moves & {"ragepowder", "followme"}:
                synergy += 0.18 * partner_offense
            if "helpinghand" in moves:
                synergy += 0.22 * partner_offense
            if "trickroom" in moves and partner_speed <= 60:
                synergy += 0.22 * partner_offense
            if "tailwind" in moves and partner_speed >= 100:
                synergy += 0.14 * partner_offense
            weather = {
                "drought": "sun",
                "drizzle": "rain", "primordialsea": "rain",
                "sandstream": "sand", "snowwarning": "snow",
            }.get(ability)
            if weather is None:
                weather = (
                    "sun" if "sunnyday" in moves else
                    "rain" if "raindance" in moves else
                    "sand" if "sandstorm" in moves else
                    "snow" if "snowscape" in moves else None
                )
            boosted_type = {
                "sun": {"fire"}, "rain": {"water"},
                "sand": {"rock", "ground", "steel"}, "snow": {"ice"},
            }.get(weather, set())
            if boosted_type and any(
                str(_GEN9_DATA.moves.get(move, {}).get("type", "")).lower() in boosted_type
                and float(_GEN9_DATA.moves.get(move, {}).get("basePower", 0) or 0) > 0
                for move in partner_moves
            ):
                synergy += 0.18 * partner_offense
        score += float(role_synergy_weight) * synergy
        if score > best_score:
            best_score, best_index = score, index
    return best_index


def _species_profile(species_id: str, moves: list[str] | None = None) -> tuple:
    dex = _GEN9_DATA.pokedex.get(to_id_str(species_id), {})
    return dex, list(dex.get("types", ())), list(moves or ())


def _damage_fraction(attacker: tuple, defender: tuple) -> float:
    dex, types, moves = attacker
    foe_dex, foe_types, _ = defender
    stats = dex.get("baseStats", {})
    foe_stats = foe_dex.get("baseStats", {})
    level = 50
    hp = max((2 * float(foe_stats.get("hp", 80)) + 31) * level / 100 + level + 10, 1)
    best = 0.0
    for move_name in moves:
        move = _GEN9_DATA.moves.get(to_id_str(move_name), {})
        power = float(move.get("basePower", 0) or 0)
        if power <= 0 or move.get("category") not in {"Physical", "Special"}:
            continue
        category = move["category"]
        attack_stat = float(stats.get("atk" if category == "Physical" else "spa", 80))
        defense_stat = max(float(foe_stats.get("def" if category == "Physical" else "spd", 80)), 1)
        damage = ((2 * level / 5 + 2) * power * attack_stat / defense_stat / 50 + 2)
        move_type = str(move.get("type", "")).upper()
        if move_type in {str(value).upper() for value in types}:
            damage *= 1.5
        for foe_type in foe_types:
            damage *= float(_GEN9_DATA.type_chart.get(move_type, {}).get(str(foe_type).upper(), 1.0))
        accuracy = move.get("accuracy", 1.0)
        if isinstance(accuracy, bool):
            damage *= float(accuracy)
        elif isinstance(accuracy, (int, float)):
            damage *= float(accuracy) / 100.0
        if str(move.get("target", "")).lower() in {"alladjacentfoes", "alladjacent", "all"}:
            damage *= 0.75
        best = max(best, min(damage / hp, 1.0))
    return best


@lru_cache(maxsize=50_000)
def _prior_damage_fraction(attacker_species: str, defender_species: str) -> float:
    """Expected best damage under public random-team set priors for one species pair."""
    attacker_id = to_id_str(attacker_species)
    defender = _species_profile(defender_species)
    profile = _species_profile(attacker_id)
    rows = _PRIOR_DATA.get("full_species_sets", {}).get(attacker_id, ())
    if rows:
        total = sum(int(row.get("count", 0)) for row in rows)
        return float(sum(
            int(row.get("count", 0))
            * _damage_fraction((profile[0], profile[1], row.get("moves", ())), defender)
            for row in rows
        ) / max(total, 1))
    possible = _PRIOR_DATA.get("species_sets", {}).get(attacker_id, ())
    move_ids = sorted({move for row in possible for move in row.get("moves", ())})
    if not move_ids:
        return 0.0
    return _damage_fraction((profile[0], profile[1], move_ids), defender)


def choose_equilibrium_preview_action(
    own_team: list[dict], opponent_team: list[dict], *, seed: int,
    iterations: int = 1000, method: str = "equilibrium",
) -> int:
    """Sample an approximate zero-sum preview equilibrium using public inputs.

    Opponent damage is estimated from species-level random-team set priors;
    the opponent's actual moves, items, ability, and team selection are never
    read. The own side uses its private sets, as permitted at team preview.
    """
    own_species = [to_id_str(p.get("species") or p.get("name", "")) for p in own_team]
    foe_species = [to_id_str(p.get("species") or p.get("name", "")) for p in opponent_team]
    own_profiles = [
        _species_profile(species, own_team[i].get("moves", ()))
        for i, species in enumerate(own_species)
    ]
    foe_profiles = [_species_profile(species) for species in foe_species]
    outgoing = np.asarray([
        [_damage_fraction(attacker, defender) for defender in foe_profiles]
        for attacker in own_profiles
    ], dtype=np.float64)
    incoming = np.asarray([
        [_prior_damage_fraction(foe, own) for own in own_species]
        for foe in foe_species
    ], dtype=np.float64)
    payoff = np.empty((len(ACTIONS), len(ACTIONS)), dtype=np.float64)
    for row, own_action in enumerate(ACTIONS):
        own_selected = own_action.leads + own_action.back
        own_leads = own_action.leads
        for col, foe_action in enumerate(ACTIONS):
            foe_selected = foe_action.leads + foe_action.back
            foe_leads = foe_action.leads
            attack_score = float(outgoing[np.ix_(own_selected, foe_selected)].mean())
            lead_attack = float(outgoing[np.ix_(own_leads, foe_leads)].mean())
            expected_incoming = float(incoming[np.ix_(foe_leads, own_leads)].mean())
            payoff[row, col] = (
                0.60 * attack_score + 0.40 * lead_attack - 0.65 * expected_incoming
            )

    rng = np.random.default_rng(seed)
    if method == "expected":
        scores = payoff.mean(axis=1)
        best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0.0))
        return int(rng.choice(best))
    if method != "equilibrium":
        raise ValueError("preview selection method must be 'expected' or 'equilibrium'")

    row_regret = np.zeros(len(ACTIONS), dtype=np.float64)
    col_regret = np.zeros(len(ACTIONS), dtype=np.float64)
    row_sum = np.zeros(len(ACTIONS), dtype=np.float64)
    col_sum = np.zeros(len(ACTIONS), dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        row_positive = np.maximum(row_regret, 0.0)
        col_positive = np.maximum(col_regret, 0.0)
        row_strategy = row_positive / row_positive.sum() if row_positive.sum() else np.full(len(ACTIONS), 1 / len(ACTIONS))
        col_strategy = col_positive / col_positive.sum() if col_positive.sum() else np.full(len(ACTIONS), 1 / len(ACTIONS))
        row_payoffs = payoff @ col_strategy
        col_payoffs = row_strategy @ payoff
        value = float(row_strategy @ payoff @ col_strategy)
        row_regret += row_payoffs - value
        col_regret += value - col_payoffs
        row_sum += row_strategy
        col_sum += col_strategy
    mix = row_sum / row_sum.sum()
    return int(rng.choice(len(ACTIONS), p=mix))


def save_team(path: Path, team: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(team, indent=2) + "\n", encoding="utf-8")


def run_worker(args: argparse.Namespace, worker: int, pair_ids: list[int]) -> dict:
    worker_dir = args.output_dir / f"worker_{worker:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    selector = None
    selector_config = None
    if args.team_selection_checkpoint is not None:
        selector_state = torch.load(
            args.team_selection_checkpoint, map_location="cpu", weights_only=False
        )
        selector_config = selector_state["model_config"]
        selector = TeamSelectionPolicy(
            selector_config["feature_dim"], selector_config["hidden_dim"],
            opponent_conditioned=selector_config.get("opponent_conditioned", False),
            action_conditioned=selector_config.get("action_conditioned", False),
        )
        selector.load_state_dict(selector_state["model_state_dict"])
        selector.eval()

    def choose_preview(
        own_team: list[dict], opponent_team: list[dict], rng_seed: int,
        strategy: str,
    ) -> int:
        if strategy == "same":
            strategy = (
                "learned" if selector is not None
                else args.preview_strategy
            )
        if selector is not None and selector_config is not None and strategy == "learned":
            own = team_features(own_team, selector_config["feature_dim"]).unsqueeze(0)
            opponent = team_features(
                opponent_team, selector_config["feature_dim"], revealed_only=True
            ).unsqueeze(0)
            with torch.inference_mode():
                scores = selector(
                    own,
                    opponent_features=opponent
                    if selector_config.get("opponent_conditioned", False) else None,
                )
            return int(scores.argmax(dim=-1).item())
        if strategy == "damage":
            return choose_damage_preview_action(
                own_team, opponent_team,
                lead_weight=args.preview_lead_weight,
                defense_weight=args.preview_defense_weight,
                role_synergy_weight=args.preview_role_synergy_weight,
                damage_model=args.preview_damage_model,
            )
        if strategy in {"expected", "equilibrium"}:
            return choose_equilibrium_preview_action(
                own_team, opponent_team, seed=rng_seed,
                iterations=args.preview_equilibrium_iterations,
                method=strategy,
            )
        return random_action(np.random.default_rng(rng_seed))

    records = []
    started = time.perf_counter()
    for pair_index in pair_ids:
        team_seed = args.seed + pair_index * 2
        team_a = generate_showdown_champions_team_data(team_seed, args.battle_format)
        team_b = generate_showdown_champions_team_data(team_seed + 1, args.battle_format)
        retries = 0
        while team_a == team_b and retries < 10:
            retries += 1
            team_b = generate_showdown_champions_team_data(
                team_seed + 1 + retries * 100_000, args.battle_format
            )
        if team_a == team_b:
            raise RuntimeError(f"could not generate distinct teams for pair {pair_index}")
        team_a_path = worker_dir / f"pair_{pair_index:04d}_A.json"
        team_b_path = worker_dir / f"pair_{pair_index:04d}_B.json"
        save_team(team_a_path, team_a)
        save_team(team_b_path, team_b)
        opponent_preview_strategy = args.opponent_preview_strategy
        preview_a = choose_preview(
            team_a, team_b, args.seed + pair_index * 2,
            "learned" if selector is not None else args.preview_strategy,
        )
        preview_b = choose_preview(
            team_b, team_a, args.seed + pair_index * 2 + 1,
            opponent_preview_strategy,
        )
        for repeat in range(args.repeats):
            case_index = pair_index * args.repeats + repeat
            leg1_path = worker_dir / f"pair_{pair_index:04d}_repeat_{repeat:02d}_p1.jsonl"
            leg2_path = worker_dir / f"pair_{pair_index:04d}_repeat_{repeat:02d}_p2.jsonl"
            common = {
                "games": 1,
                "team_seed": team_seed,
                "battle_format": args.battle_format,
                "episode_timeout": args.episode_timeout,
                "device": "cpu",
                "deterministic": True,
                "counterplay_support_actions": True,
                "counterplay_incoming_weight": args.counterplay_incoming_weight,
                "counterplay_ko_bonus": args.counterplay_ko_bonus,
                "public_counterplay_setup_bonus": args.counterplay_setup_bonus,
                "public_counterplay_switch_penalty": args.public_counterplay_switch_penalty,
                "public_counterplay_support_actions": args.public_counterplay_support_actions,
                "public_counterplay_redirection": args.public_counterplay_redirection,
                "public_counterplay_joint_outgoing": args.public_counterplay_joint_outgoing,
                "public_counterplay_sequential_damage": args.public_counterplay_sequential_damage,
                "public_counterplay_move_objective": args.public_counterplay_move_objective,
                "counterplay_threat_aggregation": args.counterplay_threat_aggregation,
                "trace_decisions": args.trace_decisions,
            }
            leg1 = evaluate(
                checkpoint=None,
                output=leg1_path,
                seed=args.seed + case_index,
                opponent="max_damage",
                player=args.player,
                team1_json=team_a_path,
                team2_json=team_b_path,
                preview_action_player1=preview_a,
                preview_action_player2=preview_b,
                **common,
            )
            # Swap teams and preview choices. Player 1 now uses max-damage;
            # score the inverted result for counterplay on player 2.
            leg2 = evaluate(
                checkpoint=None,
                output=leg2_path,
                seed=args.seed + 100_000 + case_index,
                opponent=args.player,
                player="max_damage",
                team1_json=team_b_path,
                team2_json=team_a_path,
                preview_action_player1=preview_b,
                preview_action_player2=preview_a,
                **common,
            )
            result1 = json.loads(leg1_path.read_text(encoding="utf-8").splitlines()[0])["result"]
            result2 = json.loads(leg2_path.read_text(encoding="utf-8").splitlines()[0])["result"]
            bot2 = {"win": "loss", "loss": "win", "draw": "draw"}.get(result2, result2)
            points = sum(
                1.0 if value == "win" else 0.5 if value == "draw" else 0.0
                for value in (result1, bot2)
            )
            records.append({
                "pair": pair_index,
                "repeat": repeat,
                "team_seed": team_seed,
                "team_a": [p.get("speciesId", p.get("species", "")) for p in team_a],
                "team_b": [p.get("speciesId", p.get("species", "")) for p in team_b],
                "preview_a": preview_a,
                "preview_b": preview_b,
                "player1_result": result1,
                "player2_result": bot2,
                "counterplay_points": points,
                "complete": result1 in {"win", "loss", "draw"}
                and result2 in {"win", "loss", "draw"},
                "timeouts_or_errors": int(leg1["timeouts"] + leg2["timeouts"]),
                "action_adjustments": int(leg1["action_adjustments"] + leg2["action_adjustments"]),
                "leg1_record": str(leg1_path),
                "leg2_record": str(leg2_path),
            })
    path = worker_dir / "paired_cases.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "worker": worker,
        "pair_count": len(pair_ids),
        "cases": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "path": str(path),
        "worker_dir": str(worker_dir),
    }


def run_worker_process(args: argparse.Namespace, worker: int, pair_ids: list[int]) -> dict:
    worker_dir = args.output_dir / f"worker_{worker:02d}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-id", str(worker),
        "--pair-ids", ",".join(map(str, pair_ids)),
        "--seed", str(args.seed),
        "--repeats", str(args.repeats),
        "--episode-timeout", str(args.episode_timeout),
        "--battle-format", args.battle_format,
        "--preview-strategy", args.preview_strategy,
        "--opponent-preview-strategy", args.opponent_preview_strategy,
        "--preview-lead-weight", str(args.preview_lead_weight),
        "--preview-defense-weight", str(args.preview_defense_weight),
        "--preview-role-synergy-weight", str(args.preview_role_synergy_weight),
        "--preview-damage-model", args.preview_damage_model,
        "--preview-equilibrium-iterations", str(args.preview_equilibrium_iterations),
        "--counterplay-incoming-weight", str(args.counterplay_incoming_weight),
        "--counterplay-ko-bonus", str(args.counterplay_ko_bonus),
        "--counterplay-setup-bonus", str(args.counterplay_setup_bonus),
        "--public-counterplay-switch-penalty", str(args.public_counterplay_switch_penalty),
        "--public-counterplay-move-objective", args.public_counterplay_move_objective,
        "--counterplay-threat-aggregation", args.counterplay_threat_aggregation,
        "--output-dir", str(args.output_dir),
    ]
    if args.trace_decisions:
        command.append("--trace-decisions")
    if args.public_counterplay_support_actions:
        command.append("--public-counterplay-support-actions")
    if args.public_counterplay_redirection:
        command.append("--public-counterplay-redirection")
    if args.public_counterplay_joint_outgoing:
        command.append("--public-counterplay-joint-outgoing")
    if args.public_counterplay_sequential_damage:
        command.append("--public-counterplay-sequential-damage")
    if args.team_selection_checkpoint is not None:
        command.extend(["--team-selection-checkpoint", str(args.team_selection_checkpoint)])
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    log_path = args.output_dir / f"worker_{worker:02d}.log"
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
        raise RuntimeError(f"worker {worker} failed; see {log_path}\n{process.stderr[-4000:]}")
    result = json.loads(process.stdout.splitlines()[-1])
    result["elapsed_seconds_process"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--episode-timeout", type=int, default=60)
    parser.add_argument("--battle-format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--player", choices=("counterplay", "public_counterplay"), default="counterplay")
    parser.add_argument("--counterplay-incoming-weight", type=float, default=1.0)
    parser.add_argument("--counterplay-ko-bonus", type=float, default=0.7)
    parser.add_argument("--counterplay-setup-bonus", type=float, default=0.0)
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
        "--counterplay-threat-aggregation",
        choices=("sum", "max_per_attacker"),
        default="sum",
    )
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument(
        "--preview-strategy", choices=("random", "damage", "expected", "equilibrium"), default="random",
        help="random, damage heuristic, expected utility, or public-prior zero-sum preview",
    )
    parser.add_argument(
        "--opponent-preview-strategy",
        choices=("same", "random", "damage", "expected", "equilibrium", "learned"),
        default="same",
        help="preview rule for Max Damage; same preserves the previous symmetric benchmark",
    )
    parser.add_argument(
        "--preview-lead-weight", type=float, default=0.35,
        help="extra weight for expected damage from the selected lead pair",
    )
    parser.add_argument(
        "--preview-defense-weight", type=float, default=0.08,
        help="weight for defensive type matchup across the selected four",
    )
    parser.add_argument(
        "--preview-role-synergy-weight", type=float, default=0.0,
        help="weight for Fake Out, redirection, setup, speed-control, and weather lead synergy",
    )
    parser.add_argument(
        "--preview-damage-model", choices=("base", "known_set"), default="base",
        help="use base-stat approximation or the player's known EV/IV/nature/item/ability",
    )
    parser.add_argument(
        "--preview-equilibrium-iterations", type=int, default=1000,
        help="regret-matching iterations for equilibrium preview selection",
    )
    parser.add_argument("--trace-decisions", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--pair-ids")
    args = parser.parse_args()
    if args.worker_id is not None:
        ids = [int(value) for value in (args.pair_ids or "").split(",") if value]
        result = run_worker(args, args.worker_id, ids)
        print(json.dumps(result, sort_keys=True))
        return
    if args.pairs < 1 or args.repeats < 1 or args.workers < 1:
        parser.error("pairs, repeats, and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty: {args.output_dir}")

    worker_count = min(args.pairs, args.workers)
    ids = np.array_split(np.arange(args.pairs), worker_count)
    jobs = [(worker, [int(value) for value in chunk]) for worker, chunk in enumerate(ids)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        summaries = list(
            executor.map(lambda item: run_worker_process(args, *item), jobs)
        )
    elapsed = time.perf_counter() - started
    records = []
    for item in summaries:
        with Path(item["path"]).open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    records.sort(key=lambda row: (row["pair"], row["repeat"]))
    complete = [row for row in records if row["complete"]]
    games = 2 * len(complete)
    wins = sum(
        value == "win"
        for row in complete
        for value in (row["player1_result"], row["player2_result"])
    )
    losses = sum(
        value == "loss"
        for row in complete
        for value in (row["player1_result"], row["player2_result"])
    )
    draws = games - wins - losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    se = (score * (1.0 - score) / games) ** 0.5 if games else 0.0
    cluster_values: dict[int, list[float]] = {}
    for row in complete:
        cluster_values.setdefault(int(row["pair"]), []).append(
            float(row["counterplay_points"]) / 2.0
        )
    cluster_scores = np.asarray(
        [np.mean(values) for _, values in sorted(cluster_values.items())],
        dtype=np.float64,
    )
    cluster_interval = [0.0, 0.0]
    if cluster_scores.size:
        bootstrap_rng = np.random.default_rng(args.seed ^ 0x5EEDC1A5)
        bootstrap_indices = bootstrap_rng.integers(
            0, cluster_scores.size, size=(20_000, cluster_scores.size)
        )
        cluster_interval = [
            float(value)
            for value in np.quantile(cluster_scores[bootstrap_indices].mean(axis=1), [0.025, 0.975])
        ]
    summary = {
        "player": args.player,
        "opponent": "max_damage",
        "preview_strategy": args.preview_strategy,
        "opponent_preview_strategy": args.opponent_preview_strategy,
        "preview_lead_weight": args.preview_lead_weight,
        "preview_defense_weight": args.preview_defense_weight,
        "preview_role_synergy_weight": args.preview_role_synergy_weight,
        "preview_damage_model": args.preview_damage_model,
        "preview_equilibrium_iterations": args.preview_equilibrium_iterations,
        "counterplay_incoming_weight": args.counterplay_incoming_weight,
        "counterplay_ko_bonus": args.counterplay_ko_bonus,
        "counterplay_setup_bonus": args.counterplay_setup_bonus,
        "public_counterplay_switch_penalty": args.public_counterplay_switch_penalty,
        "public_counterplay_support_actions": args.public_counterplay_support_actions,
        "public_counterplay_redirection": args.public_counterplay_redirection,
        "public_counterplay_joint_outgoing": args.public_counterplay_joint_outgoing,
        "public_counterplay_sequential_damage": args.public_counterplay_sequential_damage,
        "public_counterplay_move_objective": args.public_counterplay_move_objective,
        "counterplay_threat_aggregation": args.counterplay_threat_aggregation,
        "trace_decisions": args.trace_decisions,
        "team_selection_checkpoint": str(args.team_selection_checkpoint) if args.team_selection_checkpoint else None,
        "counterplay_support_actions": args.player == "counterplay",
        "information_boundary": "public_only" if args.player == "public_counterplay" else "privileged_opponent_battle",
        "pairs_requested": args.pairs,
        "outcome_repeats_per_pair": args.repeats,
        "pairs_completed": len(complete),
        "games_completed": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts_or_errors": len(records) - len(complete),
        "action_adjustments": sum(row["action_adjustments"] for row in records),
        "side_swapped_score_rate": score,
        "score_rate_ci95_normal_approx": [max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)],
        "score_rate_ci95_team_pair_cluster_bootstrap": cluster_interval,
        "team_pair_clusters_completed": int(cluster_scores.size),
        "worker_count": worker_count,
        "wall_seconds": elapsed,
        "aggregate_games_per_second": games / elapsed if elapsed else 0.0,
        "seed": args.seed,
        "worker_summaries": summaries,
        "results": records,
    }
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, sort_keys=True))


if __name__ == "__main__":
    main()
