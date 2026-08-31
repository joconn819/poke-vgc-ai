"""Evaluation module for benchmarking agents."""

from src.evaluation.baseline_runner import run_battles, BattleResult
from src.evaluation.benchmark import BenchmarkResult, benchmark_matchup, benchmark_matrix, save_benchmark_results

__all__ = [
    "run_battles",
    "BattleResult",
    "BenchmarkResult",
    "benchmark_matchup",
    "benchmark_matrix",
    "save_benchmark_results",
]
