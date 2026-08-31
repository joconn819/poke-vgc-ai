"""Small, dependency-free contracts for distributed trajectory collection.

The collector and learner are intentionally not coupled here.  They exchange a
manifest and content-addressed shard names, which makes retries and resuming
safe even when workers finish in a different order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


TRAJECTORY_FORMAT_VERSION = "trajectory.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_checksum(value: Any) -> str:
    """Return a deterministic SHA-256 digest for bytes, paths, or JSON values."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif isinstance(value, (str, Path)) and Path(value).is_file():
        digest = hashlib.sha256()
        with Path(value).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    else:
        payload = _canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrajectoryManifest:
    """Metadata required to safely consume a collection of trajectory shards."""

    format_version: str = TRAJECTORY_FORMAT_VERSION
    team_size: int = 6
    action_space_size: int = 126
    team_metadata: Mapping[str, Any] = field(default_factory=dict)
    action_metadata: Mapping[str, Any] = field(default_factory=dict)
    trajectory_count: int = 0
    step_count: int = 0
    shard_count: int = 0
    shards: tuple[Mapping[str, Any], ...] = ()

    def validate(self) -> "TrajectoryManifest":
        """Validate this manifest, raising ``ValueError`` with an actionable message."""
        if self.format_version != TRAJECTORY_FORMAT_VERSION:
            raise ValueError(
                f"unsupported trajectory format {self.format_version!r}; "
                f"expected {TRAJECTORY_FORMAT_VERSION!r}"
            )
        if not isinstance(self.team_metadata, Mapping):
            raise ValueError("team_metadata must be a mapping")
        if not isinstance(self.action_metadata, Mapping):
            raise ValueError("action_metadata must be a mapping")
        if not isinstance(self.team_size, int) or self.team_size <= 0:
            raise ValueError("team_size must be a positive integer")
        if not isinstance(self.action_space_size, int) or self.action_space_size <= 0:
            raise ValueError("action_space_size must be a positive integer")
        declared_team_size = self.team_metadata.get("team_size")
        if declared_team_size is not None and declared_team_size != self.team_size:
            raise ValueError("team_metadata team_size does not match team_size")
        declared_action_size = self.action_metadata.get("action_space_size")
        if declared_action_size is not None and declared_action_size != self.action_space_size:
            raise ValueError("action_metadata action_space_size does not match action_space_size")
        action_names = self.action_metadata.get("action_names")
        if action_names is not None:
            if not isinstance(action_names, (list, tuple)) or len(action_names) != self.action_space_size:
                raise ValueError("action_metadata action_names must match action_space_size")
            if any(not isinstance(name, str) or not name for name in action_names):
                raise ValueError("action names must be non-empty strings")
            if len(set(action_names)) != len(action_names):
                raise ValueError("action names must be unique")
        for name, value in (
            ("trajectory_count", self.trajectory_count),
            ("step_count", self.step_count),
            ("shard_count", self.shard_count),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.shard_count and len(self.shards) != self.shard_count:
            raise ValueError("shards length does not match shard_count")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": self.format_version,
            "team_size": self.team_size,
            "action_space_size": self.action_space_size,
            "team_metadata": dict(self.team_metadata),
            "action_metadata": dict(self.action_metadata),
            "trajectory_count": self.trajectory_count,
            "step_count": self.step_count,
            "shard_count": self.shard_count,
            "shards": [dict(shard) for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryManifest":
        manifest = cls(
            format_version=value.get("format_version", ""),
            team_size=value.get("team_size", 0),
            action_space_size=value.get("action_space_size", 0),
            team_metadata=value.get("team_metadata", {}),
            action_metadata=value.get("action_metadata", {}),
            trajectory_count=value.get("trajectory_count", 0),
            step_count=value.get("step_count", 0),
            shard_count=value.get("shard_count", 0),
            shards=tuple(value.get("shards", ())),
        )
        return manifest.validate()


def validate_manifest(manifest: TrajectoryManifest) -> TrajectoryManifest:
    """Validate and return a manifest (convenient for pipeline boundaries)."""
    if not isinstance(manifest, TrajectoryManifest):
        raise TypeError("manifest must be a TrajectoryManifest")
    return manifest.validate()


def validate_trajectory_manifest(manifest: TrajectoryManifest) -> TrajectoryManifest:
    """Backward-compatible descriptive alias for :func:`validate_manifest`."""
    return validate_manifest(manifest)


def shard_name(
    shard_index: int,
    checksum: str,
    *,
    prefix: str = "trajectories",
    extension: str = ".jsonl",
) -> str:
    """Build a stable shard filename from its index and content checksum."""
    if not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    if len(checksum) < 12 or any(char not in "0123456789abcdef" for char in checksum.lower()):
        raise ValueError("checksum must be a hexadecimal digest of at least 12 characters")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename component")
    if not extension.startswith("."):
        raise ValueError("extension must start with '.'")
    return f"{prefix}-{shard_index:06d}-{checksum[:12].lower()}{extension}"


def deterministic_shard_name(
    shard_index: int, payload: Any, *, prefix: str = "trajectories", extension: str = ".jsonl"
) -> str:
    """Name a shard directly from its serialized content."""
    return shard_name(
        shard_index, sha256_checksum(payload), prefix=prefix, extension=extension
    )


def checksum_file(path: str | Path) -> str:
    """Compute the checksum of a shard on disk."""
    return sha256_checksum(Path(path))


compute_checksum = sha256_checksum
build_shard_name = shard_name


__all__ = [
    "TRAJECTORY_FORMAT_VERSION",
    "TrajectoryManifest",
    "checksum_file",
    "build_shard_name",
    "compute_checksum",
    "deterministic_shard_name",
    "sha256_checksum",
    "shard_name",
    "validate_manifest",
    "validate_trajectory_manifest",
]
