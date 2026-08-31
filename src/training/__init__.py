"""Training module for synthetic data generation and model training."""

from src.training.synthetic_battles import (
    BattleGenerator,
    BattleRecord,
    TrajectoryStep,
)
from src.training.battle_replayer import BattleReplayer
from src.training.data import SyntheticBattleDataset, build_dataloader, generate_battle_records
from src.training.distributed import (
    TRAJECTORY_FORMAT_VERSION,
    TrajectoryManifest,
    build_shard_name,
    checksum_file,
    compute_checksum,
    deterministic_shard_name,
    sha256_checksum,
    shard_name,
    validate_manifest,
    validate_trajectory_manifest,
)
from src.training.train import SupervisedTrainer, SupervisedTrainingConfig, load_checkpoint

__all__ = [
    "BattleGenerator",
    "BattleRecord",
    "TrajectoryStep",
    "BattleReplayer",
    "SyntheticBattleDataset",
    "build_dataloader",
    "generate_battle_records",
    "SupervisedTrainer",
    "SupervisedTrainingConfig",
    "load_checkpoint",
    "TRAJECTORY_FORMAT_VERSION",
    "TrajectoryManifest",
    "build_shard_name",
    "checksum_file",
    "compute_checksum",
    "deterministic_shard_name",
    "sha256_checksum",
    "shard_name",
    "validate_manifest",
    "validate_trajectory_manifest",
]
