"""Tests for the distributed trajectory interchange contract."""

import pytest

from src.training.distributed import (
    TRAJECTORY_FORMAT_VERSION,
    TrajectoryManifest,
    deterministic_shard_name,
    sha256_checksum,
    shard_name,
)


def _manifest() -> TrajectoryManifest:
    return TrajectoryManifest(
        team_size=2,
        action_space_size=2,
        team_metadata={"format": "vgc", "team_size": 2},
        action_metadata={
            "action_space_size": 2,
            "action_names": ["switch", "move"],
        },
        trajectory_count=3,
        step_count=12,
        shard_count=1,
        shards=({"path": "trajectories-000000-abc123456789.jsonl"},),
    )


def test_manifest_validates_and_round_trips() -> None:
    manifest = _manifest().validate()
    assert TrajectoryManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.format_version == TRAJECTORY_FORMAT_VERSION


@pytest.mark.parametrize(
    "changes",
    [
        {"format_version": "trajectory.v0"},
        {"team_metadata": {"team_size": 3}},
        {"action_metadata": {"action_space_size": 2, "action_names": ["move"]}},
        {"action_metadata": {"action_space_size": 2, "action_names": ["move", "move"]}},
    ],
)
def test_manifest_rejects_incompatible_metadata(changes: dict) -> None:
    values = _manifest().to_dict()
    values.update(changes)
    with pytest.raises(ValueError):
        TrajectoryManifest.from_dict(values)


def test_shard_names_are_deterministic_and_content_addressed() -> None:
    payload = {"steps": [{"action": 4}], "worker": 7}
    assert sha256_checksum(payload) == sha256_checksum({"worker": 7, "steps": [{"action": 4}]})
    name = deterministic_shard_name(3, payload)
    assert name == deterministic_shard_name(3, payload)
    assert name.startswith("trajectories-000003-")
    assert name.endswith(".jsonl")
    assert shard_name(3, "ABCDEF0123456789") == "trajectories-000003-abcdef012345.jsonl"
