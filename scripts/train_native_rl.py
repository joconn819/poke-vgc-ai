"""Train the policy on native Showdown trajectory records."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.models.policy import TransformerPolicy
from src.training.native_ppo import (
    NATIVE_ACTION_BASE_FEATURE_VARIANT,
    NATIVE_ACTION_FEATURE_VARIANTS,
    resolve_native_action_feature_variant,
)
from src.training.native_ppo_train import NativePPOTrainer, iter_native_records
from src.training.observability import TrainingTelemetry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-path", type=Path, action="append", required=True,
        help="Native JSONL rollout file; repeat to train on several blocks together.",
    )
    parser.add_argument(
        "--preflight-report", type=Path,
        help="Use the completed rollout audit's counts and content digest to skip an extra full trajectory scan.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/native-rl"))
    parser.add_argument("--init-from-checkpoint", type=Path)
    parser.add_argument(
        "--state-value-checkpoint", type=Path,
        help="Optional closed-sheet evaluator used as a state-only GAE baseline; requires terminal rewards.",
    )
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument(
        "--epochs", type=int, default=4,
        help="Approximate passes over the rollout samples, subject to the KL stop.",
    )
    parser.add_argument("--minibatch-size", type=int, default=512,
                        help="Shuffled PPO minibatch size (default: 512).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--win-probability-loss-weight", type=float, default=0.1)
    parser.add_argument("--hidden-set-loss-weight", type=float, default=0.05)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--counterplay-prior-weight", type=float,
                        help="Train with the same score prior used in prior-aware rollouts.")
    parser.add_argument(
        "--counterplay-teacher-loss-weight", "--teacher-loss-weight",
        dest="counterplay_teacher_loss_weight", type=float, default=0.0,
        help="Add an imitation loss for recorded teacher actions or counterplay score labels.",
    )
    parser.add_argument(
        "--counterplay-imitation-only", "--teacher-imitation-only",
        dest="counterplay_imitation_only", action="store_true",
        help="Train from teacher labels without PPO reward or value losses.",
    )
    parser.add_argument(
        "--exclude-teacher-actions-from-ppo", action="store_true",
        help=("Apply PPO policy loss and KL checks only to policy-sampled actions; "
              "retain terminal value loss and teacher imitation on search-labeled states."),
    )
    parser.add_argument(
        "--training-team", choices=("both", "player1", "player2", "trainable"), default="both",
        help="Which rollout perspective to optimize; trainable uses the explicitly marked learner agent.",
    )
    parser.add_argument(
        "--reward-mode", choices=("mixed", "terminal", "hp_progress"), default="mixed",
        help="Use the environment mix, terminal win/loss only, or HP progress only for PPO advantages.",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE trace parameter controlling how far delayed rewards propagate through actions (0..1).",
    )
    parser.add_argument(
        "--gae-gamma", type=float, default=0.99,
        help="GAE discount; use 1 with lambda 1 for undiscounted win/loss objectives.",
    )
    parser.add_argument(
        "--advantage-normalization", choices=("global", "opponent"), default="global",
        help="Normalize PPO advantages globally or within each frozen opponent group.",
    )
    parser.add_argument(
        "--opponent-balanced-loss", action="store_true",
        help="Reweight PPO loss groups by rollout opponent identity.",
    )
    parser.add_argument(
        "--opponent-balance-power", type=float, default=1.0,
        help="Strength of opponent-balanced loss (0=no reweighting, 1=equal group weight; 0.5 is tempered).",
    )
    parser.add_argument(
        "--max-damage-loss-weight", type=float, default=1.0,
        help="Scale the max-damage opponent group's loss after opponent balancing.",
    )
    parser.add_argument(
        "--kl-target",
        type=float,
        default=0.05,
        help="Stop PPO epochs after the approximate KL exceeds 1.5x this target.",
    )
    parser.add_argument(
        "--initial-log-prob-tolerance", type=float, default=2e-3,
        help=("Maximum absolute rollout/replay log-probability drift allowed before PPO; "
              "small CPU kernel differences can exceed 1e-3 while measured KL remains near zero."),
    )
    parser.add_argument(
        "--off-policy-behavior-log-probs", action="store_true",
        help=("Allow recorded search behavior probabilities to differ from the "
              "current policy's initial probabilities; PPO still uses their ratios."),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--profile-preparation-stages", action="store_true",
        help="Report cumulative loader and per-transition preparation timings during PPO updates.",
    )
    gc_group = parser.add_mutually_exclusive_group()
    gc_group.add_argument(
        "--disable-gc-during-ppo-update",
        dest="disable_gc_during_ppo_update",
        action="store_true",
        help="Disable cyclic GC during each PPO update (the default).",
    )
    gc_group.add_argument(
        "--enable-gc-during-ppo-update",
        dest="disable_gc_during_ppo_update",
        action="store_false",
        help="Keep cyclic GC enabled during PPO updates.",
    )
    parser.set_defaults(disable_gc_during_ppo_update=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="native-ppo")
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument("--native-action-features", action="store_true")
    parser.add_argument(
        "--native-action-feature-variant",
        choices=sorted(NATIVE_ACTION_FEATURE_VARIANTS),
    )
    args = parser.parse_args()
    if args.state_value_checkpoint is not None:
        if not args.state_value_checkpoint.is_file():
            parser.error(f"state-value checkpoint does not exist: {args.state_value_checkpoint}")
        if args.reward_mode != "terminal":
            parser.error("--state-value-checkpoint requires --reward-mode terminal")
        if args.gae_gamma != 1.0 or args.gae_lambda != 1.0:
            parser.error("--state-value-checkpoint requires --gae-gamma 1 and --gae-lambda 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    telemetry = TrainingTelemetry(
        args.telemetry_path or args.checkpoint_dir / "telemetry.jsonl", args.run_id
    )
    checkpoint_state = None
    model_config = {}
    if args.init_from_checkpoint:
        checkpoint_state = torch.load(args.init_from_checkpoint, map_location="cpu", weights_only=True)
        model_config = dict(checkpoint_state.get("model_config") or {})
    if args.counterplay_prior_weight is None:
        args.counterplay_prior_weight = float(
            checkpoint_state.get("counterplay_prior_weight", 0.0)
            if checkpoint_state is not None else 0.0
        )
    requested_variant = args.native_action_feature_variant
    if (
        requested_variant is None
        and args.native_action_features
        and not model_config.get("native_action_feature_variant")
        and not model_config.get("native_action_feature_dim")
    ):
        requested_variant = NATIVE_ACTION_BASE_FEATURE_VARIANT
    if requested_variant is not None:
        resolved_variant, resolved_dim = resolve_native_action_feature_variant(requested_variant)
        model_config["native_action_feature_variant"] = resolved_variant
        model_config["native_action_feature_dim"] = resolved_dim
    # These heads consume no additional observation data and do not alter the
    # action logits. Missing weights in older checkpoints initialize cleanly.
    model_config["predict_opponent_abilities"] = True
    model = TransformerPolicy.from_config_dict(model_config)
    trainer = NativePPOTrainer(model, args.checkpoint_dir, lr=args.lr, grad_clip=args.grad_clip)
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state["model_state_dict"], strict=False)
    rollout_digest = hashlib.sha256()
    valid_record_count = 0
    sample_count = 0
    rollout_id = None
    record_offsets = None
    if args.preflight_report is not None:
        audit = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        expected_paths = [str(path.resolve()) for path in args.trajectory_path]
        audited_paths = [str(Path(path).resolve()) for path in audit.get("trajectory_files", [])]
        if not audit.get("ready_for_ppo") or audited_paths != expected_paths:
            raise ValueError("preflight report is not ready or does not match the rollout files")
        if args.training_team != "trainable":
            raise ValueError("preflight counts can only skip the scan for training_team=trainable")
        audited_metadata = audit.get("file_metadata")
        if not isinstance(audited_metadata, dict):
            raise ValueError("preflight report lacks file metadata; rerun the strict rollout audit")
        for path in args.trajectory_path:
            metadata = audited_metadata.get(str(path))
            stat = path.stat()
            if metadata != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError(f"rollout changed since preflight audit: {path}")
        counts = audit.get("counts", {})
        valid_record_count = int(counts.get("ppo_valid_rows", 0))
        sample_count = int(counts.get("trainable_transitions", 0))
        record_offsets = audit.get("trainable_record_offsets")
        if record_offsets is not None:
            if not isinstance(record_offsets, dict):
                raise ValueError("preflight trainable record offsets are malformed")
            expected_games = int(counts.get("trainable_games", 0))
            indexed_games = sum(len(offsets) for offsets in record_offsets.values())
            if indexed_games != expected_games:
                raise ValueError(
                    "preflight trainable record offsets do not match its trainable-game count"
                )
        rollout_id = str(audit.get("content_digest", ""))
        if valid_record_count < 1 or sample_count < 1 or len(rollout_id) != 64:
            raise ValueError("preflight report is missing valid counts or a SHA-256 rollout digest")
    else:
        for record in iter_native_records(args.trajectory_path, digest=rollout_digest):
            valid_record_count += 1
            first_step = record["trajectory"][0]
            action_keys = [
                player for player in first_step.get("actions", {})
                if player in first_step.get("log_probs", {})
            ]
            if args.training_team == "trainable":
                action_keys = [
                    player for player in action_keys
                    if player == record.get("trainable_agent")
                ]
            elif args.training_team != "both":
                agent_to_team = record.get("agent_to_team") or {}
                player_order = list(first_step.get("observations", {}))
                action_keys = [
                    player for player in action_keys
                    if agent_to_team.get(
                        player,
                        player if player in record["teams"]
                        else f"player{player_order.index(player) + 1}",
                    ) == args.training_team
                ]
            sample_count += len(record["trajectory"]) * len(action_keys)
        rollout_id = rollout_digest.hexdigest()
    if sample_count < 1:
        raise ValueError("trajectory file contains no trainable native transitions")
    effective_batch_size = min(args.minibatch_size or sample_count, sample_count)
    steps_per_sweep = (sample_count + effective_batch_size - 1) // effective_batch_size
    optimizer_steps_per_update = steps_per_sweep * args.epochs
    optimizer_step_budget = optimizer_steps_per_update * args.updates
    telemetry.start(
        trajectory_path=[str(path) for path in args.trajectory_path],
        checkpoint_dir=str(args.checkpoint_dir),
        init_from_checkpoint=str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
        device=args.device,
        seed=args.seed,
        valid_records=valid_record_count,
        trainable_transitions=sample_count,
        sample_count_source="preflight" if args.preflight_report is not None else "trajectory_scan",
        optimizer_steps_per_sweep=steps_per_sweep,
        optimizer_step_budget=optimizer_step_budget,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        entropy_weight=args.entropy_weight,
        policy_temperature=args.policy_temperature,
        counterplay_prior_weight=args.counterplay_prior_weight,
        counterplay_teacher_loss_weight=args.counterplay_teacher_loss_weight,
        training_team=args.training_team,
        reward_mode=args.reward_mode,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.advantage_normalization,
        opponent_balanced_loss=args.opponent_balanced_loss,
        opponent_balance_power=args.opponent_balance_power,
        max_damage_loss_weight=args.max_damage_loss_weight,
        kl_target=args.kl_target,
        grad_clip=args.grad_clip,
        native_action_features=bool(getattr(model, "native_action_feature_dim", 0)),
        native_action_feature_variant=getattr(model, "native_action_feature_variant", None),
    )
    for update in range(args.updates):
        gc_was_enabled = gc.isenabled()
        if args.disable_gc_during_ppo_update and gc_was_enabled:
            gc.disable()
        try:
            metrics = trainer.update(
                iter_native_records(
                    args.trajectory_path,
                    trainable_only=args.training_team == "trainable",
                    record_offsets_by_file=record_offsets,
                ),
                device=args.device,
                # NativePPOTrainer counts optimizer steps. Scale requested passes
                # by minibatch count instead of spending the entire budget on only
                # `epochs` minibatches from a large rollout.
                epochs=optimizer_steps_per_update,
                max_rollout_steps=optimizer_step_budget,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                entropy_weight=args.entropy_weight,
                kl_target=None if args.counterplay_imitation_only else args.kl_target,
                temperature=args.policy_temperature,
                reward_mode=args.reward_mode,
                gae_lambda=args.gae_lambda,
                gae_gamma=args.gae_gamma,
                state_value_checkpoint=args.state_value_checkpoint,
                counterplay_prior_weight=args.counterplay_prior_weight,
                counterplay_teacher_loss_weight=args.counterplay_teacher_loss_weight,
                win_probability_loss_weight=args.win_probability_loss_weight,
                hidden_set_loss_weight=args.hidden_set_loss_weight,
                counterplay_imitation_only=args.counterplay_imitation_only,
                exclude_teacher_actions_from_ppo=args.exclude_teacher_actions_from_ppo,
                initial_log_prob_tolerance=(
                    None
                    if args.counterplay_imitation_only or args.off_policy_behavior_log_probs
                    else args.initial_log_prob_tolerance
                ),
                training_team=args.training_team,
                advantage_normalization=args.advantage_normalization,
                opponent_balanced_loss=args.opponent_balanced_loss,
                opponent_balance_power=args.opponent_balance_power,
                max_damage_loss_weight=args.max_damage_loss_weight,
                rollout_id=rollout_id,
                records_already_validated=True,
                profile_preparation_stages=args.profile_preparation_stages,
            )
        finally:
            if args.disable_gc_during_ppo_update and gc_was_enabled:
                gc.enable()
        actual_transitions = int(metrics.get("transitions", -1))
        if actual_transitions != sample_count:
            raise ValueError(
                "prepared transition count does not match the audited sample count: "
                f"expected {sample_count}, received {actual_transitions}"
            )
        metrics["optimizer_steps_run"] = float(metrics.get("epochs_run", 0.0))
        metrics["optimizer_steps_per_sweep"] = float(steps_per_sweep)
        metrics["passes_requested"] = float(args.epochs)
        metrics["passes_equivalent_run"] = (
            metrics["optimizer_steps_run"] / float(steps_per_sweep)
        )
        metrics["training_team"] = args.training_team
        checkpoint_path = args.checkpoint_dir / f"update_{update + 1:06d}.pt"
        trainer.save(checkpoint_path,
                     counterplay_prior_weight=args.counterplay_prior_weight)
        telemetry.emit(
            "update_completed",
            update=update + 1,
            checkpoint=str(checkpoint_path),
            **metrics,
        )
        print({"update": update + 1, **metrics})
    telemetry.emit("run_completed", updates=args.updates)


if __name__ == "__main__":
    main()
