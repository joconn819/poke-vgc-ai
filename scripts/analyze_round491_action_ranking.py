#!/usr/bin/env python3
"""Diagnose held-out action-value ranking and candidate-set headroom."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.train_action_conditioned_win_model import ActionWinModel, _inputs


def _bootstrap(values: dict[int, float], seed: int, rounds: int = 10_000) -> list[float]:
    rows = np.asarray(list(values.values()), dtype=np.float64)
    if rows.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rows[rng.integers(0, rows.size, (rounds, rows.size))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def _family(name: str) -> str:
    return "ppo" if name == "policy" or name.startswith("policy_round") else name


def _pick(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=202610493)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    paths = sorted(Path(path) for path in glob.glob(args.checkpoint_glob))
    if not paths:
        raise SystemExit("checkpoint glob matched no files")

    # Collapse aliases into unique legal joint actions at each held-out state.
    states: dict[int, dict[tuple[int, int], dict]] = defaultdict(dict)
    for source in payload["rows"]:
        state_id = int(source["team_cluster"])
        action = tuple(map(int, source["action"]))
        target = states[state_id].get(action)
        if target is None:
            target = dict(source)
            target["candidate_names"] = {str(source["candidate"])}
            states[state_id][action] = target
        else:
            for field in ("observation", "native_action_features", "public_context_features",
                          "outcomes_by_opponent"):
                if target[field] != source[field]:
                    raise ValueError(f"inconsistent alias field {field} at {state_id}/{action}")
            target["candidate_names"].add(str(source["candidate"]))

    eval_names = sorted(payload["rows"][0]["outcomes_by_opponent"])
    all_reports = []
    for model_no, checkpoint_path in enumerate(paths):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        response_names = list(ckpt["opponent_response_names"])
        if not set(response_names).issubset(eval_names):
            raise ValueError(f"missing outcomes for fitted heads in {checkpoint_path}")
        model = ActionWinModel(
            ckpt["observation_dim"], ckpt["action_width"], ckpt["extra_feature_dim"],
            response_count=len(response_names),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        action_width = int(ckpt["action_width"])
        use_logp = bool(ckpt["use_policy_log_probability"])

        state_rows = {}
        brier_by_head: dict[str, list[float]] = {name: [] for name in response_names}
        action_mse_by_head: dict[str, list[float]] = {name: [] for name in response_names}
        ranking_by_head: dict[str, list[float]] = {name: [] for name in response_names}
        robust_rank, family_robust_rank = [], []
        for state_id, action_map in sorted(states.items()):
            candidates = list(action_map.values())
            features = np.stack([
                _inputs(row["observation"], row["action"], action_width,
                        row["native_action_features"], row["public_context_features"],
                        row.get("policy_log_probability") if use_logp else None, use_logp)
                for row in candidates
            ])
            with torch.no_grad():
                predictions = torch.sigmoid(model(torch.as_tensor(
                    features, dtype=torch.float32, device=device
                ))).cpu().numpy()
            outcome_tensor = np.asarray([
                [row["outcomes_by_opponent"][name] for name in eval_names]
                for row in candidates
            ], dtype=np.float64)  # [actions, responses, chance repeats]
            means = outcome_tensor.mean(axis=2)
            response_index = {name: i for i, name in enumerate(response_names)}

            for j, name in enumerate(response_names):
                pred = predictions[:, j]
                # Score all independent branch outcomes, not only their means.
                brier_by_head[name].extend(
                    ((pred[:, None] - outcome_tensor[:, eval_names.index(name), :]) ** 2)
                    .reshape(-1).tolist()
                )
                action_mse_by_head[name].extend(
                    ((pred - means[:, eval_names.index(name)]) ** 2).tolist()
                )
                actual = means[:, eval_names.index(name)]
                pair_scores = []
                for a in range(len(candidates)):
                    for b in range(a + 1, len(candidates)):
                        delta = actual[a] - actual[b]
                        if abs(delta) < 1e-12:
                            continue
                        predicted_delta = pred[a] - pred[b]
                        pair_scores.append(
                            1.0 if predicted_delta * delta > 0 else
                            0.5 if abs(predicted_delta) < 1e-12 else 0.0
                        )
                if pair_scores:
                    ranking_by_head[name].append(float(np.mean(pair_scores)))

            observed_fit = means[:, [eval_names.index(name) for name in response_names]]
            predicted_fit = predictions[:, [response_index[name] for name in response_names]]
            observed_worst = observed_fit.min(axis=1)
            predicted_worst = predicted_fit.min(axis=1)
            grouped_pred = np.stack([
                predicted_fit[:, [i for i, name in enumerate(response_names)
                                if _family(name) == family_name]].mean(axis=1)
                for family_name in sorted({_family(name) for name in response_names})
            ], axis=1)
            predicted_family_worst = grouped_pred.min(axis=1)

            def pair_concordance(x: np.ndarray, y: np.ndarray) -> float | None:
                results = []
                for a in range(len(x)):
                    for b in range(a + 1, len(x)):
                        delta = y[a] - y[b]
                        if abs(delta) < 1e-12:
                            continue
                        pred_delta = x[a] - x[b]
                        results.append(1.0 if pred_delta * delta > 0 else
                                       0.5 if abs(pred_delta) < 1e-12 else 0.0)
                return float(np.mean(results)) if results else None

            for target, metric in ((robust_rank, predicted_worst),
                                   (family_robust_rank, predicted_family_worst)):
                score = pair_concordance(metric, observed_worst)
                if score is not None:
                    target.append(score)
            policy_pos = next((i for i, row in enumerate(candidates)
                               if "policy" in row["candidate_names"]), None)
            if policy_pos is None:
                continue

            # Select on one four-repeat half, score on the other half, then swap.
            split_metrics = []
            for select_idx, score_idx in ((np.arange(4), np.arange(4, 8)),
                                          (np.arange(4, 8), np.arange(4))):
                select_means = outcome_tensor[:, :, select_idx].mean(axis=2)
                score_means = outcome_tensor[:, :, score_idx].mean(axis=2)
                select_fit = select_means[:, [eval_names.index(n) for n in response_names]]
                score_all_worst = score_means.min(axis=1)
                split_metrics.append({
                    "baseline": float(score_all_worst[policy_pos]),
                    "model": float(score_all_worst[_pick(predicted_worst)]),
                    "oracle_fit_sample": float(score_all_worst[_pick(select_fit.min(axis=1))]),
                    "oracle_all_sample": float(score_all_worst[_pick(select_means.min(axis=1))]),
                    "oracle_score_sample": float(np.max(score_all_worst)),
                })
            state_rows[state_id] = split_metrics

        def state_metric(key: str) -> dict[int, float]:
            return {
                state_id: statistics.mean(row[key] - row["baseline"] for row in halves)
                for state_id, halves in state_rows.items()
            }

        method_results = {}
        state_deltas = {}
        for method in ("model", "oracle_fit_sample", "oracle_all_sample", "oracle_score_sample"):
            deltas = state_metric(method)
            state_deltas[method] = {str(key): value for key, value in deltas.items()}
            method_results[method] = {
                "mean_delta_vs_incumbent_pp": round(statistics.mean(deltas.values()) * 100, 3),
                "team_cluster_bootstrap_ci95_pp": [
                    round(x * 100, 3) for x in _bootstrap(deltas, args.seed + model_no * 10 + len(method_results))
                ],
            }

        all_reports.append({
            "checkpoint": str(checkpoint_path),
            "seed": ckpt.get("seed"),
            "heldout_states": len(state_rows),
            "candidate_actions": {
                "mean": round(statistics.mean(len(v) for v in states.values()), 3),
                "median": float(np.median([len(v) for v in states.values()])),
            },
            "branch_outcome_brier_by_fitted_response": {
                name: round(float(np.mean(values)), 5) for name, values in brier_by_head.items()
            },
            "action_mean_mse_by_fitted_response": {
                name: round(float(np.mean(values)), 5) for name, values in action_mse_by_head.items()
            },
            "within_state_pairwise_ranking_accuracy_by_fitted_response": {
                name: round(float(np.mean(values)), 4) if values else None
                for name, values in ranking_by_head.items()
            },
            "robust_worst_response_pairwise_ranking_accuracy": (
                round(float(np.mean(robust_rank)), 4) if robust_rank else None
            ),
            "family_collapsed_worst_response_pairwise_ranking_accuracy": (
                round(float(np.mean(family_robust_rank)), 4) if family_robust_rank else None
            ),
            "split_repeat_choice_results": method_results,
            "_state_deltas": state_deltas,
        })

    aggregate_methods = {}
    for method in ("model", "oracle_fit_sample", "oracle_all_sample", "oracle_score_sample"):
        per_fit = [row["_state_deltas"][method] for row in all_reports]
        state_ids = sorted(set.intersection(*(set(values) for values in per_fit)))
        # The model fits are repeated estimates on the same states. Average
        # within state first, then bootstrap states to avoid pseudo-replication.
        state_values = {
            int(state_id): statistics.mean(float(values[state_id]) for values in per_fit)
            for state_id in state_ids
        }
        aggregate_methods[method] = {
            "mean_delta_vs_incumbent_pp": round(statistics.mean(state_values.values()) * 100, 3),
            "team_cluster_bootstrap_ci95_pp": [
                round(value * 100, 3) for value in _bootstrap(
                    state_values, args.seed + 1000 + len(aggregate_methods)
                )
            ],
            "team_clusters": len(state_values),
        }
    for row in all_reports:
        row.pop("_state_deltas", None)

    report = {
        "schema_version": 1,
        "experiment": "Round491 held-out action ranking and candidate-set headroom",
        "data": str(args.data),
        "states": len(states),
        "fitted_responses": response_names,
        "evaluated_responses": eval_names,
        "repeat_split": "Select on repeats 0-3 and score on 4-7, then swap; differences averaged per held-out state.",
        "limitations": (
            "One-step candidate-set analysis only. The split-repeat oracle is a noisy upper-bound diagnostic, "
            "not a deployable policy or full-match result. Outcomes against fitted responses determine "
            "the oracle selection objective; all response classes are used for test scoring."
        ),
        "aggregate_across_fits_by_team_cluster": aggregate_methods,
        "models": all_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "states": len(states),
        "aggregate_across_fits_by_team_cluster": aggregate_methods,
        "models": [{
            "seed": row["seed"],
            "pairwise_worst_rank": row["robust_worst_response_pairwise_ranking_accuracy"],
            "model_delta": row["split_repeat_choice_results"]["model"]["mean_delta_vs_incumbent_pp"],
            "oracle_sample_delta": row["split_repeat_choice_results"]["oracle_fit_sample"]["mean_delta_vs_incumbent_pp"],
        } for row in all_reports],
    }, indent=2))


if __name__ == "__main__":
    main()
