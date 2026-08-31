"""Reproducible pairwise benchmark utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List

from src.baselines.base_agent import Agent
from src.evaluation.baseline_runner import run_battles


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome and uncertainty for one directed agent matchup."""

    agent1: str
    agent2: str
    agent1_wins: int
    agent2_wins: int
    total_battles: int
    seed: int
    win_rate_ci_low: float
    win_rate_ci_high: float

    @property
    def agent1_win_rate(self) -> float:
        return self.agent1_wins / self.total_battles if self.total_battles else 0.0

    @property
    def agent2_win_rate(self) -> float:
        return self.agent2_wins / self.total_battles if self.total_battles else 0.0


def benchmark_matchup(
    agent1_factory: Callable[[], Agent],
    agent2_factory: Callable[[], Agent],
    *,
    num_battles: int,
    seed: int,
    agent1_name: str = "agent1",
    agent2_name: str = "agent2",
) -> BenchmarkResult:
    """Run a matchup with fresh agents and compute a Wilson 95% interval."""
    if num_battles < 0:
        raise ValueError("num_battles must be non-negative")

    result = run_battles(
        agent1_factory(),
        agent2_factory(),
        num_battles=num_battles,
        seed=seed,
    )
    low, high = _wilson_interval(result.agent1_wins, result.total_battles)
    return BenchmarkResult(
        agent1=agent1_name,
        agent2=agent2_name,
        agent1_wins=result.agent1_wins,
        agent2_wins=result.agent2_wins,
        total_battles=result.total_battles,
        seed=seed,
        win_rate_ci_low=low,
        win_rate_ci_high=high,
    )


def benchmark_matrix(
    agents: Iterable[tuple[str, Callable[[], Agent]]],
    *,
    num_battles: int,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Run every ordered pair of distinct agents with deterministic seeds."""
    named_agents = list(agents)
    results: List[BenchmarkResult] = []
    matchup_index = 0
    for index, (name1, factory1) in enumerate(named_agents):
        for other_index, (name2, factory2) in enumerate(named_agents):
            if index == other_index:
                continue
            results.append(
                benchmark_matchup(
                    factory1,
                    factory2,
                    num_battles=num_battles,
                    seed=seed + matchup_index,
                    agent1_name=name1,
                    agent2_name=name2,
                )
            )
            matchup_index += 1
    return results


def save_benchmark_results(results: Iterable[BenchmarkResult], path: str | Path) -> None:
    """Persist benchmark results as a JSON array."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _wilson_interval(wins: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    proportion = wins / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * ((proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) ** 0.5)
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)
