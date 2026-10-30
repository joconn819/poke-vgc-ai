#!/usr/bin/env python3
"""Build the public dashboard from its curated aggregate data source."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "site" / "progress-source.json"
OUTPUT = ROOT / "site" / "progress.json"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} from {SOURCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
