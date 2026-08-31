"""Tests for reproducible pairwise benchmarking."""

import json

from src.baselines.random_agent import RandomAgent
from src.evaluation.benchmark import BenchmarkResult, benchmark_matchup, save_benchmark_results


def test_benchmark_matchup_is_reproducible() -> None:
    first = benchmark_matchup(
        lambda: RandomAgent(seed=1),
        lambda: RandomAgent(seed=2),
        num_battles=20,
        seed=123,
    )
    second = benchmark_matchup(
        lambda: RandomAgent(seed=1),
        lambda: RandomAgent(seed=2),
        num_battles=20,
        seed=123,
    )

    assert first == second
    assert first.total_battles == 20
    assert 0.0 <= first.agent1_win_rate <= 1.0
    assert 0.0 <= first.win_rate_ci_low <= first.win_rate_ci_high <= 1.0


def test_benchmark_result_serializes_to_json(tmp_path) -> None:
    result = BenchmarkResult(
        agent1="policy",
        agent2="random",
        agent1_wins=7,
        agent2_wins=3,
        total_battles=10,
        seed=42,
        win_rate_ci_low=0.30,
        win_rate_ci_high=0.90,
    )
    output = tmp_path / "benchmark.json"

    save_benchmark_results([result], output)

    payload = json.loads(output.read_text())
    assert payload[0]["agent1"] == "policy"
    assert payload[0]["agent1_wins"] == 7
    assert payload[0]["win_rate_ci_low"] == 0.30


def test_zero_battles_has_neutral_interval() -> None:
    result = BenchmarkResult(
        agent1="a",
        agent2="b",
        agent1_wins=0,
        agent2_wins=0,
        total_battles=0,
        seed=0,
        win_rate_ci_low=0.0,
        win_rate_ci_high=0.0,
    )
    assert result.agent1_win_rate == 0.0
