#!/usr/bin/env python3
"""Build GitHub Pages progress data from the explicit snapshot in PLAN.md."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "PLAN.md"
OUTPUT = ROOT / "site" / "progress.json"
SNAPSHOT = re.compile(
    r"<!-- DASHBOARD_PROGRESS_JSON\s*(\{.*?\})\s*DASHBOARD_PROGRESS_JSON -->",
    re.S,
)


def main() -> None:
    text = PLAN.read_text(encoding="utf-8")
    match = SNAPSHOT.search(text)
    if not match:
        raise SystemExit("PLAN.md is missing its DASHBOARD_PROGRESS_JSON snapshot")

    data = json.loads(match.group(1))
    # PLAN.md owns fields it explicitly records. Preserve the larger set of
    # historical report data already published in progress.json so regenerating
    # the dashboard does not silently erase metrics omitted from that snapshot.
    existing: dict = {}
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}

    def merge_preserving_reports(old: dict, updates: dict) -> dict:
        merged = dict(old)
        for key, value in updates.items():
            previous = merged.get(key)
            if isinstance(previous, dict) and isinstance(value, dict):
                merged[key] = merge_preserving_reports(previous, value)
            else:
                merged[key] = value
        return merged

    data = merge_preserving_reports(existing, data)
    data["repository"] = "joconn819/poke-vgc-ai"
    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
