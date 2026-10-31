#!/usr/bin/env python3
"""Align Round498 public states and expose each set prior as a response head."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SHARED_FIELDS = (
    "observation",
    "legal_action_mask",
    "native_action_features",
    "public_context_features",
    "policy_log_probability",
)


def _state_rows(payload: dict) -> dict[tuple[int, int], dict[tuple[int, int], dict]]:
    states: dict[tuple[int, int], dict[tuple[int, int], dict]] = defaultdict(dict)
    for row in payload.get("rows", []):
        state_key = (int(row["team_cluster"]), int(row.get("state_repeat", 0)))
        action_key = tuple(map(int, row["action"]))
        existing = states[state_key].get(action_key)
        if existing is None:
            states[state_key][action_key] = dict(row)
            continue
        for field in (*SHARED_FIELDS, "outcomes_by_opponent"):
            if existing.get(field) != row.get(field):
                raise ValueError(f"inconsistent duplicate {field} for {state_key} {action_key}")
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("showdown_prior_data", type=Path)
    parser.add_argument("uniform_prior_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    showdown = json.loads(args.showdown_prior_data.read_text(encoding="utf-8"))
    uniform = json.loads(args.uniform_prior_data.read_text(encoding="utf-8"))
    if showdown.get("hidden_world_prior") != "showdown_random_champions":
        raise ValueError("first input must use the Showdown Random Champions prior")
    if uniform.get("hidden_world_prior") != "uniform_unique_full_sets":
        raise ValueError("second input must use the uniform observed full-set prior")

    showdown_states = _state_rows(showdown)
    uniform_states = _state_rows(uniform)
    showdown_names = sorted(showdown_states)
    uniform_names = sorted(uniform_states)
    common_states = sorted(set(showdown_states) & set(uniform_states))
    if not common_states:
        raise ValueError("the prior datasets have no matched public states")
    first_state = common_states[0]
    first_action = next(iter(showdown_states[first_state]))
    response_names = sorted(
        showdown_states[first_state][first_action]["outcomes_by_opponent"]
    )
    for state_key in common_states:
        a_actions = set(showdown_states[state_key])
        b_actions = set(uniform_states[state_key])
        if a_actions != b_actions:
            raise ValueError(f"candidate action sets differ between priors for {state_key}")

    rows = []
    for state_key in common_states:
        for action_key in sorted(showdown_states[state_key]):
            a = showdown_states[state_key][action_key]
            b = uniform_states[state_key][action_key]
            for field in SHARED_FIELDS:
                if a.get(field) != b.get(field):
                    raise ValueError(f"prior inputs differ for {field}: {state_key} {action_key}")
            a_outcomes = a["outcomes_by_opponent"]
            b_outcomes = b["outcomes_by_opponent"]
            if sorted(a_outcomes) != response_names or sorted(b_outcomes) != response_names:
                raise ValueError(f"response classes differ between priors for {state_key} {action_key}")
            outcomes = {
                **{f"showdown__{name}": values for name, values in a_outcomes.items()},
                **{f"uniform__{name}": values for name, values in b_outcomes.items()},
            }
            if len({len(values) for values in outcomes.values()}) != 1:
                raise ValueError(f"chance repeat counts differ for {state_key} {action_key}")
            rows.append({**a, "outcomes_by_opponent": outcomes})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        "schema_version": 1,
        "experiment": "Round498 matched hidden-prior response-head union",
        "hidden_world_priors": [
            "showdown_random_champions", "uniform_unique_full_sets"
        ],
        "source_data": [str(args.showdown_prior_data), str(args.uniform_prior_data)],
        "response_names": [f"showdown__{name}" for name in response_names]
        + [f"uniform__{name}" for name in response_names],
        "matched_states": len(common_states),
        "showdown_only_states": len(set(showdown_states) - set(uniform_states)),
        "uniform_only_states": len(set(uniform_states) - set(showdown_states)),
        "team_clusters": len({key[0] for key in common_states}),
        "rows": rows,
    }
    args.output.write_text(json.dumps(merged, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in merged.items() if key != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
