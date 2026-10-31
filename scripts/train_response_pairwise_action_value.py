#!/usr/bin/env python3
"""Train calibrated response-wise action values with an extra pairwise loss.

This targets the Round491 diagnosis: retain absolute win-probability fitting,
while explicitly teaching the model to order actions within each state and
opponent response. Team clusters stay disjoint between fit and validation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_action_conditioned_win_model import ActionWinModel, _inputs


def _brier(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean((torch.sigmoid(logits) - labels) ** 2).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202610700)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--heldout-response", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.pairwise_weight < 0:
        parser.error("pairwise-weight must be nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", [])
    if not raw_rows:
        raise ValueError("counterfactual data is empty")
    eval_names = sorted(raw_rows[0]["outcomes_by_opponent"])
    heldout = set(args.heldout_response)
    unknown = heldout - set(eval_names)
    if unknown:
        raise ValueError(f"unknown held-out response names: {sorted(unknown)}")
    response_names = [name for name in eval_names if name not in heldout]
    if len(response_names) < 2:
        raise ValueError("at least two response heads must remain")

    # Collapse aliases but keep their shared state, action, and labels intact.
    dedup: dict[tuple[int, int, tuple[int, int]], dict] = {}
    for row in raw_rows:
        key = (int(row["team_cluster"]), int(row["state_repeat"]),
               tuple(map(int, row["action"])))
        prior = dedup.get(key)
        if prior is None:
            dedup[key] = {**row, "candidate_names": {str(row["candidate"])}}
        else:
            for field in ("observation", "native_action_features", "public_context_features",
                          "outcomes_by_opponent"):
                if prior[field] != row[field]:
                    raise ValueError(f"inconsistent duplicate action field {field}: {key}")
            prior["candidate_names"].add(str(row["candidate"]))
    rows = list(dedup.values())
    repeat_counts = {
        len(row["outcomes_by_opponent"][response_names[0]]) for row in rows
    }
    if len(repeat_counts) != 1 or next(iter(repeat_counts)) < 4:
        raise ValueError("all action/response rows need at least four chance outcomes")
    repeats = next(iter(repeat_counts))
    midpoint = repeats // 2
    width = len(rows[0]["legal_action_mask"]) // 2
    obs_dim = len(rows[0]["observation"])
    use_logp = all(row.get("policy_log_probability") is not None for row in rows)
    feature_rows = [
        _inputs(row["observation"], row["action"], width,
                row["native_action_features"], row["public_context_features"],
                row.get("policy_log_probability") if use_logp else None, use_logp)
        for row in rows
    ]
    features = np.stack(feature_rows).astype(np.float32)
    outcomes = np.asarray([
        [row["outcomes_by_opponent"][name] for name in response_names]
        for row in rows
    ], dtype=np.float32)  # [action rows, response heads, chance repeats]

    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((int(row["team_cluster"]), int(row["state_repeat"])), []).append(index)
    cluster_ids = np.asarray(sorted({int(row["team_cluster"]) for row in rows}), dtype=np.int64)
    shuffled = rng.permutation(cluster_ids)
    val_count = max(1, round(len(shuffled) * 0.20))
    val_clusters = set(map(int, shuffled[:val_count]))
    train_clusters = set(map(int, shuffled[val_count:]))
    if not train_clusters:
        raise ValueError("not enough team clusters for a separate validation split")

    train_action_idx, train_labels = [], []
    val_action_idx, val_labels = [], []
    for action_index, row in enumerate(rows):
        target_idx = train_action_idx if int(row["team_cluster"]) in train_clusters else val_action_idx
        target_labels = train_labels if int(row["team_cluster"]) in train_clusters else val_labels
        chance_slice = slice(0, midpoint) if target_idx is train_action_idx else slice(midpoint, repeats)
        for r in range(chance_slice.start, chance_slice.stop):
            target_idx.append(action_index)
            target_labels.append(outcomes[action_index, :, r].tolist())
    if not val_action_idx:
        raise ValueError("validation split has no action outcomes")

    pair_a, pair_b, pair_targets, pair_clusters = [], [], [], []
    val_pair_a, val_pair_b, val_pair_targets = [], [], []
    for (cluster, _), indexes in groups.items():
        is_train = cluster in train_clusters
        for pos, a in enumerate(indexes):
            for b in indexes[pos + 1:]:
                a_rates = outcomes[a, :, :midpoint].mean(axis=1) if is_train else outcomes[a, :, midpoint:].mean(axis=1)
                b_rates = outcomes[b, :, :midpoint].mean(axis=1) if is_train else outcomes[b, :, midpoint:].mean(axis=1)
                if is_train:
                    pair_a.append(a); pair_b.append(b)
                    pair_targets.append(0.5 + 0.5 * (a_rates - b_rates))
                    pair_clusters.append(cluster)
                else:
                    val_pair_a.append(a); val_pair_b.append(b)
                    val_pair_targets.append(0.5 + 0.5 * (a_rates - b_rates))
    if not pair_a or not val_pair_a:
        raise ValueError("need within-state candidate pairs in train and validation clusters")

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y_action = torch.as_tensor(np.asarray(train_labels), dtype=torch.float32, device=device)
    i_action = torch.as_tensor(train_action_idx, dtype=torch.long, device=device)
    y_val_action = torch.as_tensor(np.asarray(val_labels), dtype=torch.float32, device=device)
    i_val_action = torch.as_tensor(val_action_idx, dtype=torch.long, device=device)
    i_pair_a = torch.as_tensor(pair_a, dtype=torch.long, device=device)
    i_pair_b = torch.as_tensor(pair_b, dtype=torch.long, device=device)
    y_pair = torch.as_tensor(np.asarray(pair_targets), dtype=torch.float32, device=device)
    i_val_pair_a = torch.as_tensor(val_pair_a, dtype=torch.long, device=device)
    i_val_pair_b = torch.as_tensor(val_pair_b, dtype=torch.long, device=device)
    y_val_pair = torch.as_tensor(np.asarray(val_pair_targets), dtype=torch.float32, device=device)

    action_feature_dim = len(rows[0]["native_action_features"][0][0])
    context_dim = len(rows[0]["public_context_features"])
    extra_dim = features.shape[1] - obs_dim - 2 * width
    model = ActionWinModel(obs_dim, width, extra_dim, response_count=len(response_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        action_perm = rng.permutation(len(train_action_idx))
        pair_perm = rng.permutation(len(pair_a))
        steps = max(
            (len(action_perm) + args.batch_size - 1) // args.batch_size,
            (len(pair_perm) + args.batch_size - 1) // args.batch_size,
        )
        for step in range(steps):
            ab = action_perm[step * args.batch_size:(step + 1) * args.batch_size]
            pb = pair_perm[step * args.batch_size:(step + 1) * args.batch_size]
            losses = []
            if len(ab):
                action_logits = model(x[i_action[ab]])
                action_targets = y_action[ab].reshape(-1, len(response_names))
                losses.append(nn.functional.binary_cross_entropy_with_logits(
                    action_logits, action_targets
                ))
            if len(pb):
                left = model(x[i_pair_a[pb]])
                right = model(x[i_pair_b[pb]])
                pair_logits = left - right
                losses.append(args.pairwise_weight * nn.functional.binary_cross_entropy_with_logits(
                    pair_logits, y_pair[pb]
                ))
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x[i_val_action]).reshape(-1, len(response_names))
            action_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, y_val_action.reshape(-1, len(response_names)))
            val_pair_logits = model(x[i_val_pair_a]) - model(x[i_val_pair_b])
            pair_loss = nn.functional.binary_cross_entropy_with_logits(val_pair_logits, y_val_pair)
            val_loss = float((action_loss + args.pairwise_weight * pair_loss).item())
        epoch_log.append({
            "epoch": epoch, "validation_total_loss": val_loss,
            "validation_action_brier": _brier(val_logits, y_val_action.reshape(-1, len(response_names))),
            "validation_pair_loss": float(pair_loss.item()),
        })
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(x[i_val_action]).reshape(-1, len(response_names))
        val_brier = _brier(val_logits, y_val_action.reshape(-1, len(response_names)))
        val_pair_logits = model(x[i_val_pair_a]) - model(x[i_val_pair_b])
        val_pair_loss = float(nn.functional.binary_cross_entropy_with_logits(val_pair_logits, y_val_pair).item())

    metrics = {
        "validation_action_outcome_brier": val_brier,
        "validation_pairwise_loss": val_pair_loss,
        "validation_total_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_log),
        "train_team_clusters": len(train_clusters),
        "validation_team_clusters": len(val_clusters),
        "train_actions": len(train_action_idx),
        "validation_actions": len(val_action_idx),
        "train_pairs": len(pair_a),
        "validation_pairs": len(val_pair_a),
        "chance_repeats_per_action": repeats,
        "training_chance_half": "first_half",
        "validation_chance_half": "second_half_on_validation_team_clusters",
    }
    checkpoint = {
        "schema_version": 1,
        "model_type": "response_pairwise_calibrated_action_value",
        "opponent_response_names": response_names,
        "heldout_response_names": sorted(heldout),
        "observation_dim": obs_dim,
        "action_width": width,
        "action_feature_dim": action_feature_dim,
        "context_dim": context_dim,
        "extra_feature_dim": extra_dim,
        "use_policy_log_probability": use_logp,
        "state_dict": best_state,
        "metrics": metrics,
        "training_data": str(args.data),
        "seed": args.seed,
        "pairwise_weight": args.pairwise_weight,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "response_pairwise_action_value.pt"
    temp = out.with_suffix(".pt.tmp")
    torch.save(checkpoint, temp)
    temp.replace(out)
    report = {"checkpoint": str(out), "seed": args.seed, "metrics": metrics,
              "response_names": response_names, "heldout_response_names": sorted(heldout),
              "pairwise_weight": args.pairwise_weight, "epoch_log": epoch_log}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
