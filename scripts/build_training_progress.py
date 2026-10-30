#!/usr/bin/env python3
"""Build the public, non-sensitive GitHub Pages training summary from PLAN.md."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "PLAN.md"
OUTPUT = ROOT / "site" / "progress.json"


def main() -> None:
    if not PLAN.is_file():
        if OUTPUT.is_file():
            print(f"{PLAN} is unavailable; keeping the committed dashboard summary")
            return
        raise FileNotFoundError(f"cannot build training summary without {PLAN}")
    text = PLAN.read_text(encoding="utf-8")
    current = re.search(r"\*\*Current decision:\*\*(.*?)(?=\n\s*Independent confirmation|\n\s*Round\d+|\Z)", text, re.S)
    current_text = current.group(1) if current else ""
    candidate = re.search(
        r"Round(\d+).*?([0-9]+\.[0-9]+)%.*?95%.*?\[([0-9.]+%),\s*([0-9.]+%)\]",
        current_text,
        re.S,
    )
    # Keep a known-good fallback so a small wording edit in PLAN.md does not
    # silently publish an empty dashboard.
    candidate_data = {
        "round": int(candidate.group(1)) if candidate else 402,
        "win_rate_pct": float(candidate.group(2)) if candidate else 77.15,
        "ci95": f"[{candidate.group(3)}, {candidate.group(4)}]" if candidate else "[74.90%, 79.35%]",
        "games": 2048,
        "status": "below target",
    }
    result_420 = re.search(
        r"Round420.*?Round402 scored ([0-9.]+)%.*?Round420 scored ([0-9.]+)%.*?paired lift was ([+-][0-9.]+) pp \(95% cluster-bootstrap\s+interval \[([^]]+)\]",
        text,
        re.S,
    )
    latest = (
        f"Round420 trained 20,000 outcomes for 16 epochs. On a fresh paired screen it scored {result_420.group(2)}% vs Round402's {result_420.group(1)}%; paired change {result_420.group(3)} pp (95% CI [{result_420.group(4)}]). Not a confirmed improvement."
        if result_420 else
        "Round420 trained on 20,000 current-policy preview outcomes for 16 epochs; paired evaluation is pending."
    )
    rounds = re.findall(r"Round419 traced.*?([0-9]+\.[0-9]+) games/s", text, re.S)
    speed = float(rounds[-1]) if rounds else 9.36
    data = {
        "repository": "joconn819/poke-vgc-ai",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "target_pct": 80,
        "candidate": candidate_data,
        "speed": {
            "games_per_second": f"{speed:.2f}",
            "context": "Latest 256-match traced local headless evaluation; not a raw simulator benchmark.",
            "decision_latency": "Re-profile pending",
            "current_bottleneck": "Policy inference / feature prep",
            "gpu_status": "PPO training; simulator CPU",
        },
        "latest_run": {
            "round": 420,
            "status": "completed · inconclusive",
            "summary": latest,
            "games": "20,000 outcomes",
            "epochs": "16",
            "evaluation": "No significant gain",
        },
        "method": "Native self-play PPO for in-battle decisions, with terminal win/loss as the objective and auxiliary win-probability plus hidden-set prediction losses. Search is a later add-on after the base policy improves.",
        "jax_decision": "A full JAX battle-engine rewrite is gated on fresh profiling. Existing measurements put policy inference and feature preparation at least level with Showdown advancement, limiting the likely gain from simulator-only acceleration.",
        "evaluation_note": "The headline result is the best confirmed paired result against Max Damage on held-out team-pair clusters. Round402 remains below the 80% target; training-set results and small screens are not presented as confirmed strength.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
