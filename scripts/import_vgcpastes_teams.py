#!/usr/bin/env python3
"""Fetch and prepare event-separated, exact-SP Champions M-C teams.

The raw source is cached locally with a checksum and attribution. Only teams with
six complete sets and exact SP spreads are included. Evaluation splits are by
tournament event; no event occurs in both development and final.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE = "https://raw.githubusercontent.com/PizzaTimeJoshua/munchstats/teams-data/app/champions-mc.json.gz"
INDEX = "https://raw.githubusercontent.com/PizzaTimeJoshua/munchstats/teams-data/app/index.json"
ATTRIBUTION = "Team data from the VGCPastes Repository; fetched via MunchStats' public teams-data branch."
DEV_EVENTS = {
    "Baltimore Regional 2027", "Frankfurt Regional 2027", "Brisbane Regional 2027",
    "Recife Regional 2027", "Louisville Regional 2027",
}
FINAL_EVENTS = {
    "Nice Regional 2027", "Puebla Regional 2027", "Gdańsk Regional 2027",
    "2027 Global Challenge II", "Taiwan PBL 2027",
}
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "poke-learning-research/1.0 (team-source attribution retained)"})
    with urllib.request.urlopen(req, timeout=40) as response:
        return response.read()


def rank_value(rank: str) -> int | None:
    text = rank.strip().lower()
    if text in {"champion", "winner", "1st", "1", "1st place"}:
        return 1
    if text in {"runner up", "runner-up", "2nd", "2", "2nd place"}:
        return 2
    m = re.match(r"^(\d+)(?:st|nd|rd|th)?$", text)
    if m:
        return int(m.group(1))
    return None


def normalize(record: dict[str, Any]) -> list[dict[str, Any]]:
    sets = record["sets"]
    if len(sets) != 6:
        raise ValueError("set_count")
    team = []
    for row in sets:
        if len(row) < 8 or not isinstance(row[5], list) or len(row[5]) != 4:
            raise ValueError("incomplete_set")
        species, item, ability, tera, nature, moves, points, ivs = row[:8]
        if not isinstance(points, list) or len(points) != 6 or any(x is None for x in points):
            raise ValueError("missing_sp")
        evs = dict(zip(STAT_KEYS, map(int, points)))
        iv_values = dict(zip(STAT_KEYS, map(int, ivs))) if isinstance(ivs, list) and len(ivs) == 6 else None
        member: dict[str, Any] = {
            "name": species, "species": species, "item": item or "", "ability": ability,
            "moves": moves, "nature": nature or "Serious", "evs": evs, "level": 50,
        }
        if tera:
            member["teraType"] = tera
        if iv_values is not None:
            member["ivs"] = iv_values
        team.append(member)
    if len({str(p["species"]).casefold() for p in team}) != 6:
        raise ValueError("duplicate_species")
    return team


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--showdown-path", type=Path, required=True)
    ap.add_argument("--format", default="gen9championsdoubles2026mcclosed")
    ap.add_argument("--seed", type=int, default=593202611)
    ap.add_argument("--max-placement", type=int, default=64)
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw = get(SOURCE)
    (args.output_dir / "champions-mc.json.gz").write_bytes(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    index = get(INDEX)
    (args.output_dir / "source-index.json").write_bytes(index)
    catalog = json.loads(gzip.decompress(raw))
    if catalog.get("kind") != "vgcpastes" or catalog.get("reg", "").lower() != "mc":
        raise SystemExit("downloaded catalog is not the expected VGCPastes M-C dataset")
    accepted: dict[str, list[dict[str, Any]]] = {"development": [], "final": []}
    rejected: Counter[str] = Counter()
    source_events: dict[str, set[str]] = {"development": set(), "final": set()}
    roster_hashes: set[str] = set()
    for record in catalog["teams"]:
        event = record.get("event", "")
        split = "development" if event in DEV_EVENTS else "final" if event in FINAL_EVENTS else None
        if split is None:
            continue
        source_events[split].add(event)
        if record.get("evs") is not True:
            rejected[f"{split}:no_exact_sp"] += 1
            continue
        placement = rank_value(str(record.get("rank", "")))
        if placement is None or placement > args.max_placement:
            rejected[f"{split}:not_ranked_within_{args.max_placement}"] += 1
            continue
        try:
            team = normalize(record)
        except ValueError as exc:
            rejected[f"{split}:{exc}"] += 1
            continue
        roster = tuple(sorted(str(p["species"]).casefold() for p in team))
        digest = hashlib.sha256("\n".join(roster).encode()).hexdigest()
        if digest in roster_hashes:
            rejected[f"{split}:duplicate_roster"] += 1
            continue
        roster_hashes.add(digest)
        accepted[split].append({
            "source_team_id": record.get("id"), "event": event,
            "event_date": record.get("date"), "rank": record.get("rank"),
            "placement": placement, "title": record.get("title"),
            "player": record.get("player"), "paste_id": record.get("paste"),
            "source_url": record.get("source"), "team": team,
            "roster_sha256": digest,
        })

    rng = random.Random(args.seed)
    pairs: dict[str, list[dict[str, Any]]] = {}
    for split in ("development", "final"):
        rows = accepted[split]
        rng.shuffle(rows)
        # Pair builds from different events to avoid a single-event match-up bias.
        rows.sort(key=lambda x: x["event"])
        event_buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            event_buckets.setdefault(row["event"], []).append(row)
        events = list(event_buckets)
        rng.shuffle(events)
        ordered: list[dict[str, Any]] = []
        while any(event_buckets.values()):
            for event in events:
                if event_buckets[event]:
                    ordered.append(event_buckets[event].pop())
        pairs[split] = []
        for i in range(0, len(ordered) - 1, 2):
            a, b = ordered[i], ordered[i + 1]
            if a["event"] == b["event"]:
                # Prefer not to pair two teams from the same event; matching is
                # kept deterministic and any unavoidable same-event pairs are tagged.
                other = next((j for j in range(i + 2, len(ordered)) if ordered[j]["event"] != a["event"]), None)
                if other is not None:
                    ordered[i + 1], ordered[other] = ordered[other], ordered[i + 1]
                    b = ordered[i + 1]
            pairs[split].append({
                "pair_id": len(pairs[split]), "team_a": a["team"], "team_b": b["team"],
                "event_a": a["event"], "event_b": b["event"],
                "source_a": {k: a[k] for k in ("source_team_id", "event", "event_date", "rank", "title", "player", "paste_id", "source_url", "roster_sha256")},
                "source_b": {k: b[k] for k in ("source_team_id", "event", "event_date", "rank", "title", "player", "paste_id", "source_url", "roster_sha256")},
                "same_event_pair": a["event"] == b["event"],
            })
    revision = subprocess.run(["git", "-C", str(args.showdown_path.resolve()), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    manifest = {
        "schema": "vgcpastes-event-split-v1", "format": args.format,
        "source_url": SOURCE, "source_index_url": INDEX, "attribution": ATTRIBUTION,
        "source_sha256": source_hash, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalog_metadata": {k: catalog.get(k) for k in ("api_version", "kind", "name", "reg", "format", "attribution")},
        "showdown_path": str(args.showdown_path.resolve()), "showdown_revision": revision,
        "rank_filter": f"numeric/champion placement <= {args.max_placement}; exact SP, six distinct complete sets",
        "seed": args.seed, "event_assignments": {"development": sorted(DEV_EVENTS), "final": sorted(FINAL_EVENTS)},
        "observed_events": {k: sorted(v) for k, v in source_events.items()},
        "team_counts": {k: len(v) for k, v in accepted.items()},
        "pair_counts": {k: len(v) for k, v in pairs.items()}, "rejections": dict(sorted(rejected.items())),
        "same_event_pair_counts": {k: sum(p["same_event_pair"] for p in v) for k, v in pairs.items()},
        "splits": pairs,
    }
    (args.output_dir / "team_pair_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    (args.output_dir / "teams_by_event.json").write_text(json.dumps(accepted, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_sha256", "showdown_revision", "observed_events", "team_counts", "pair_counts", "rejections", "same_event_pair_counts")}, indent=2))


if __name__ == "__main__":
    main()
