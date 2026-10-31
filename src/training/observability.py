"""Durable, append-only telemetry for training runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Iterable


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for event records."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TrainingTelemetry:
    """Write one JSON object per line so progress survives interruptions."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def emit(self, event: str, **metrics: Any) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "timestamp": utc_timestamp(),
            "run_id": self.run_id,
            "event": event,
            **metrics,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def start(self, **metadata: Any) -> dict[str, Any]:
        return self.emit(
            "run_started",
            host=socket.gethostname(),
            platform=platform.platform(),
            pid=os.getpid(),
            **metadata,
        )


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read valid telemetry events and fail clearly on malformed JSON."""
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid telemetry JSON at line {line_number}") from exc
            if not isinstance(event, dict) or "event" not in event:
                raise ValueError(f"telemetry line {line_number} is not an event object")
            events.append(event)
    return events


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return the latest durable status and aggregate counters."""
    items = list(events)
    completed = [e for e in items if e.get("event") == "round_completed"]
    episodes = [e for e in items if e.get("event") == "episode_completed"]
    updates = [e for e in items if e.get("event") == "update_completed"]
    latest = items[-1] if items else {}
    return {
        "run_id": latest.get("run_id"),
        "status": latest.get("event", "not_started"),
        "rounds_completed": len(completed),
        "updates_completed": len(updates),
        "games": sum(int(e.get("games", 0)) for e in completed)
        + sum(int(e.get("games", 0)) for e in episodes),
        "transitions": sum(int(e.get("transitions", 0)) for e in updates),
        "latest": latest,
    }
