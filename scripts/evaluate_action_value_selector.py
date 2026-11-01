#!/usr/bin/env python3
"""Full-match held-out evaluation of an action-conditioned win model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_midgame_action_counterfactuals import strategic_candidate_actions
from scripts.run_headless_showdown_self_play import (
    _legal_native_action_descriptors,
    _load_policy,
    _load_team_selector,
    run_game,
)
from scripts.train_action_conditioned_win_model import ActionWinModel, _inputs
from src.baselines.native_damage import choose_max_damage_action
from src.env.headless_showdown import HeadlessShowdownWorker
from src.env.showdown_env import ShowdownDoublesEnv


def _bootstrap(values: dict[int, float], seed: int) -> list[float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, (20000, array.size))
    return np.quantile(array[indexes].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--action-model-checkpoint", type=Path, required=True)
    parser.add_argument("--team-selection-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--showdown-path", type=Path, default=Path(
        "/home/justin/p/poke_learning/pokemon-showdown"))
    parser.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    parser.add_argument("--seed", type=int, default=2026102806)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--decision-step", type=int, default=5)
    parser.add_argument(
        "--min-predicted-advantage", type=float, default=0.0,
        help=("only override the policy action when the selected action's "
              "aggregated predicted win probability exceeds the policy action "
              "by at least this margin"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.clusters < 1 or args.decision_step < 1:
        parser.error("clusters and decision-step must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some imported modules may initialize PyTorch's inter-op pool before
        # the CLI reaches this point. Inference is still safe with that pool.
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner_output = args.output.with_suffix(".runner.jsonl")
    if runner_output.exists():
        runner_output.unlink()

    policy_model = _load_policy(args.policy_checkpoint, args.device)
    opponent_model = _load_policy(args.opponent_checkpoint, args.device)
    team_selector, team_selector_config = _load_team_selector(
        args.team_selection_checkpoint
    )
    model_payload = torch.load(args.action_model_checkpoint, map_location=args.device,
                               weights_only=True)
    response_names = list(model_payload.get("opponent_response_names", ()))
    action_model = ActionWinModel(
        int(model_payload["observation_dim"]), int(model_payload["action_width"]),
        int(model_payload.get("extra_feature_dim", 0)),
        response_count=max(1, len(response_names)),
    ).to(args.device)
    action_model.load_state_dict(model_payload["state_dict"])
    action_model.eval()
    action_width = int(model_payload["action_width"])
    data_generator = np.random.default_rng(args.seed)

    def selector(observation: np.ndarray, mask: np.ndarray, policy_action: np.ndarray,
                 battle, decision_seed: int, acting_policy=None):
        damage_action = choose_max_damage_action(
            mask, battle, order_validator=ShowdownDoublesEnv.action_to_order,
        ).tolist()
        mask_list = mask.astype(int).tolist()
        candidates = strategic_candidate_actions(
            policy_action.astype(int).tolist(), damage_action, mask_list,
            _legal_native_action_descriptors(mask_list, battle), decision_seed,
        )
        legal_candidates = []
        for name, action in candidates:
            try:
                ShowdownDoublesEnv.action_to_order(
                    np.asarray(action, dtype=np.int64), battle, strict=True
                )
            except ValueError:
                continue
            legal_candidates.append((name, action))
        if not legal_candidates:
            legal_candidates = [("policy", policy_action.astype(int).tolist())]
        candidate_logps = [None] * len(legal_candidates)
        if model_payload.get("use_policy_log_probability") and acting_policy is not None:
            try:
                candidate_logps = acting_policy.log_prob_native_actions(
                    observation, mask, [action for _, action in legal_candidates]
                )
            except (ValueError, RuntimeError):
                candidate_logps = [None] * len(legal_candidates)
        action_features = getattr(acting_policy, "last_native_action_features", None)
        context_features = getattr(acting_policy, "public_context_features", None)
        features = np.stack([
            _inputs(observation.tolist(), action, action_width,
                    action_features if model_payload.get("action_feature_dim", 0) else None,
                    context_features if model_payload.get("context_dim", 0) else None,
                    candidate_logps[index],
                    bool(model_payload.get("use_policy_log_probability", False)))
            for index, (_, action) in enumerate(legal_candidates)
        ])
        with torch.no_grad():
            response_probabilities = torch.sigmoid(
                action_model(torch.as_tensor(features, dtype=torch.float32,
                                             device=args.device))
            ).cpu().numpy()
        if response_probabilities.ndim == 1:
            response_probabilities = response_probabilities[:, None]
        aggregation = str(model_payload.get("robust_aggregation", "mean"))
        if aggregation == "minimum":
            probabilities = response_probabilities.min(axis=1)
        elif aggregation == "mean":
            probabilities = response_probabilities.mean(axis=1)
        else:
            raise ValueError(f"unsupported action-value aggregation: {aggregation}")
        best_index = int(np.argmax(probabilities))
        selected_name, selected_action = legal_candidates[best_index]
        policy_action_list = policy_action.astype(int).tolist()
        policy_index = next((index for index, (_, action) in enumerate(legal_candidates)
                             if action == policy_action_list), None)
        if policy_index is None:
            policy_index = len(legal_candidates)
            legal_candidates.append(("policy", policy_action_list))
            policy_log_probability = None
            if model_payload.get("use_policy_log_probability") and acting_policy is not None:
                try:
                    policy_log_probability = acting_policy.log_prob_native_actions(
                        observation, mask, [policy_action_list]
                    )[0]
                except (ValueError, RuntimeError):
                    policy_log_probability = None
            policy_features = _inputs(
                observation.tolist(), policy_action_list, action_width,
                action_features if model_payload.get("action_feature_dim", 0) else None,
                context_features if model_payload.get("context_dim", 0) else None,
                policy_log_probability,
                bool(model_payload.get("use_policy_log_probability", False)),
            )
            with torch.no_grad():
                policy_probabilities = torch.sigmoid(action_model(
                    torch.as_tensor(policy_features, dtype=torch.float32,
                                    device=args.device)
                )).cpu().numpy()
            if policy_probabilities.ndim == 1:
                policy_probabilities = policy_probabilities[None, :]
            response_probabilities = np.concatenate(
                (response_probabilities, policy_probabilities), axis=0
            )
            if aggregation == "minimum":
                probabilities = response_probabilities.min(axis=1)
            else:
                probabilities = response_probabilities.mean(axis=1)
        predicted_advantage = float(probabilities[best_index] - probabilities[policy_index])
        margin_fallback = predicted_advantage < args.min_predicted_advantage
        if margin_fallback:
            selected_name, selected_action = "policy", policy_action_list
            selected_index = policy_index
        else:
            selected_index = best_index
        report = {
            "selected_source": "policy_margin_fallback" if margin_fallback else "action_value_model",
            "selected_candidate": selected_name,
            "selected_action": selected_action,
            "policy_action": policy_action_list,
            "max_damage_action": damage_action,
            "selected_probability": float(probabilities[selected_index]),
            "predicted_policy_probability": float(probabilities[policy_index]),
            "predicted_advantage_over_policy": predicted_advantage,
            "minimum_predicted_advantage": float(args.min_predicted_advantage),
            "aggregation": aggregation,
            "response_names": response_names,
            "selected_response_probabilities": response_probabilities[selected_index].tolist(),
            "candidates": [
                {"name": name, "action": action,
                 "predicted_probability": float(prob),
                 "response_probabilities": response_probabilities[index].tolist()}
                for index, ((name, action), prob) in enumerate(zip(legal_candidates, probabilities))
            ],
        }
        return np.asarray(selected_action, dtype=np.int64), report

    rows = []
    matches = 0
    started = time.perf_counter()
    with HeadlessShowdownWorker(args.showdown_path) as worker:
        for cluster in range(args.clusters):
            base_seed = args.seed + cluster * 10007
            teams = (worker.generate_team(base_seed, args.format),
                     worker.generate_team(base_seed + 1, args.format))
            opponent_strategy = "policy" if cluster % 2 else "max_damage"
            for bot_side in (0, 1):
                seed = base_seed + (100_003 if bot_side else 0)
                player_strategy = (
                    "policy" if bot_side == 0 or opponent_strategy == "policy"
                    else opponent_strategy
                )
                common = dict(
                    game=0, seed=seed, format_id=args.format,
                    showdown_path=args.showdown_path, output_path=runner_output,
                    model=policy_model,
                    opponent_model=opponent_model if opponent_strategy == "policy" else None,
                    player_strategy=player_strategy,
                    opponent_strategy=(opponent_strategy if bot_side == 0 else "policy"),
                    trainable_side="player1" if bot_side == 0 else "player2",
                    temperature=1.0, device=args.device, telemetry=None,
                    worker=worker, deterministic_policy=True,
                    teams_override=teams, swap_teams=bool(bot_side),
                    team_selection_model=team_selector,
                    team_selection_config=team_selector_config,
                    opponent_team_selection_model=(
                        team_selector if opponent_strategy == "policy" else None
                    ),
                    opponent_team_selection_config=(
                        team_selector_config if opponent_strategy == "policy" else None
                    ),
                    capture_action_step=args.decision_step,
                    persist_record=False,
                )
                baseline = run_game(**common)
                selected = run_game(
                    **common,
                    counterfactual_action_selector=selector,
                    counterfactual_selector_decision=args.decision_step,
                )
                matches += 2
                agent = "Headless P1" if bot_side == 0 else "Headless P2"
                bdata = (baseline.get("counterfactual_action_data_by_agent") or {}).get(agent)
                sdata = (selected.get("counterfactual_action_data_by_agent") or {}).get(agent)
                if (bdata is None) != (sdata is None):
                    raise RuntimeError(
                        f"decision reach mismatch in cluster {cluster}, side {bot_side}"
                    )
                reached = bdata is not None
                if reached and (
                    bdata["observation"] != sdata["observation"]
                    or bdata["legal_action_mask"] != sdata["legal_action_mask"]
                    or baseline["preview_actions"] != selected["preview_actions"]
                ):
                    raise RuntimeError(
                        f"paired selector state mismatch in cluster {cluster}, side {bot_side}"
                    )
                selector_report = selected["counterfactual_selector_report"]
                if reached and selector_report is None:
                    raise RuntimeError(
                        f"selector was not called in cluster {cluster}, side {bot_side}"
                    )
                if selector_report is not None and selector_report.get("selected_action_accepted") is False:
                    raise RuntimeError(
                        f"selector produced invalid joint action in cluster {cluster}, side {bot_side}"
                    )
                actual_opponent = opponent_strategy
                rows.append({
                    "cluster": cluster,
                    "bot_side": bot_side,
                    "opponent_strategy": actual_opponent,
                    "decision_reached": reached,
                    "policy_action": bdata["action"] if reached else None,
                    "selected_action": sdata["action"] if reached else None,
                    "selected_candidate": selector_report.get("selected_candidate") if selector_report else None,
                    "selected_probability": selector_report.get("selected_probability") if selector_report else None,
                    "policy_win": int(baseline["winner"] == agent),
                    "selected_win": int(selected["winner"] == agent),
                })
            if (cluster + 1) % 16 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(json.dumps({"clusters_completed": cluster + 1,
                                  "matches_per_second": matches / elapsed}), flush=True)
    if runner_output.exists():
        runner_output.unlink()

    def summarize(name: str, selected_rows: list[dict]) -> dict:
        policy_scores: dict[int, list[float]] = {}
        selected_scores: dict[int, list[float]] = {}
        deltas: dict[int, list[float]] = {}
        for row in selected_rows:
            c = row["cluster"]
            policy_scores.setdefault(c, []).append(float(row["policy_win"]))
            selected_scores.setdefault(c, []).append(float(row["selected_win"]))
            deltas.setdefault(c, []).append(float(row["selected_win"] - row["policy_win"]))
        p = {c: float(np.mean(v)) for c, v in policy_scores.items()}
        s = {c: float(np.mean(v)) for c, v in selected_scores.items()}
        d = {c: float(np.mean(v)) for c, v in deltas.items()}
        return {
            "arm": name,
            "clusters": len(p),
            "policy_win_rate": float(np.mean(list(p.values()))) if p else None,
            "selected_win_rate": float(np.mean(list(s.values()))) if s else None,
            "selected_minus_policy": float(np.mean(list(d.values()))) if d else None,
            "paired_cluster_bootstrap_95_ci": _bootstrap(d, args.seed + len(name)) if d else None,
        }

    eligible = [row for row in rows if row["decision_reached"]]
    overall = summarize("all", rows)
    conditional = summarize("decision_reached", eligible)
    stratified = {
        strategy: summarize(strategy, [r for r in rows if r["opponent_strategy"] == strategy])
        for strategy in ("max_damage", "policy")
    }
    seat_stratified = {
        f"P{side + 1}": summarize(f"P{side + 1}", [r for r in rows if r["bot_side"] == side])
        for side in (0, 1)
    }
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "policy_checkpoint": str(args.policy_checkpoint),
        "opponent_checkpoint": str(args.opponent_checkpoint),
        "action_model_checkpoint": str(args.action_model_checkpoint),
        "team_selection_checkpoint": (
            str(args.team_selection_checkpoint)
            if args.team_selection_checkpoint is not None else None
        ),
        "format": args.format,
        "seed": args.seed,
        "decision_step": args.decision_step,
        "min_predicted_advantage": args.min_predicted_advantage,
        "matches_executed": matches,
        "elapsed_seconds": elapsed,
        "matches_per_second": matches / max(elapsed, 1e-9),
        "summary": {
            "overall": overall,
            "conditional_on_reaching_decision": conditional,
            "by_opponent": stratified,
            "by_seat": seat_stratified,
            "decision_reach_rate": len(eligible) / max(len(rows), 1),
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
