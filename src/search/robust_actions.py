"""Zero-sum mixed strategies for simultaneous action payoff matrices.

This module deliberately accepts a payoff matrix instead of a battle object.
The simulator adapter must construct that matrix from the acting player's
observation and a declared set of hidden-state hypotheses; it must not pass
the live simulator's unrevealed opponent sets into policy/search features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RegretMatchingResult:
    """Approximate row/column strategies and their matrix-game value."""

    row_strategy: np.ndarray
    column_strategy: np.ndarray
    value: float
    exploitability: float
    iterations: int


def _positive_regret_strategy(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    return positive / total if total > 1e-12 else np.full_like(positive, 1.0 / len(positive))


def regret_matching(payoff_matrix: np.ndarray, iterations: int = 500) -> RegretMatchingResult:
    """Approximate a zero-sum equilibrium using regret matching.

    Rows maximize the acting player's win-probability payoff; columns
    minimize it. Returned strategies are time-averaged iterates, not the final
    instantaneous regrets.
    """
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("payoff_matrix must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff_matrix must contain only finite values")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    row_count, column_count = matrix.shape
    row_regrets = np.zeros(row_count, dtype=np.float64)
    column_regrets = np.zeros(column_count, dtype=np.float64)
    row_sum = np.zeros(row_count, dtype=np.float64)
    column_sum = np.zeros(column_count, dtype=np.float64)
    for _ in range(iterations):
        row_strategy = _positive_regret_strategy(row_regrets)
        column_strategy = _positive_regret_strategy(column_regrets)
        value = float(row_strategy @ matrix @ column_strategy)
        row_regrets += matrix @ column_strategy - value
        column_regrets += value - row_strategy @ matrix
        row_sum += row_strategy
        column_sum += column_strategy

    row_average = row_sum / iterations
    column_average = column_sum / iterations
    row_best_response = float(np.max(matrix @ column_average))
    column_best_response = float(np.min(row_average @ matrix))
    return RegretMatchingResult(
        row_strategy=row_average,
        column_strategy=column_average,
        value=float(row_average @ matrix @ column_average),
        exploitability=max(0.0, row_best_response - column_best_response),
        iterations=iterations,
    )


def select_robust_action(
    payoff_matrix: np.ndarray,
    *,
    method: str = "maximin",
    rng: np.random.Generator | None = None,
    iterations: int = 500,
) -> tuple[int, RegretMatchingResult]:
    """Choose a deterministic worst-case or equilibrium-mixed row action."""
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    result = regret_matching(matrix, iterations=iterations)
    if method == "maximin":
        action = int(np.argmax(matrix.min(axis=1)))
    elif method == "equilibrium":
        generator = rng or np.random.default_rng()
        action = int(generator.choice(len(result.row_strategy), p=result.row_strategy))
    else:
        raise ValueError("method must be 'maximin' or 'equilibrium'")
    return action, result


def aggregate_hidden_world_payoffs(
    payoff_matrices: np.ndarray,
    *,
    method: str = "mean",
    lower_quantile: float = 0.1,
) -> np.ndarray:
    """Aggregate equal-shaped action matrices over hidden-set hypotheses.

    ``mean`` uses a uniform belief over sampled worlds. ``lower_quantile``
    retains outcomes in the pessimistic tail, and ``worst_case`` is the strict
    per-cell minimum. None of these methods models opponent behavioral
    tendencies; opponent action columns remain explicit responses in each
    matrix.
    """
    worlds = np.asarray(payoff_matrices, dtype=np.float64)
    if worlds.ndim != 3 or not all(worlds.shape):
        raise ValueError("payoff_matrices must have shape (worlds, rows, columns)")
    if not np.isfinite(worlds).all():
        raise ValueError("payoff_matrices must contain only finite values")
    if method == "mean":
        return worlds.mean(axis=0)
    if method == "worst_case":
        return worlds.min(axis=0)
    if method == "lower_quantile":
        if not 0.0 <= lower_quantile <= 1.0:
            raise ValueError("lower_quantile must be in [0, 1]")
        return np.quantile(worlds, lower_quantile, axis=0)
    raise ValueError("method must be 'mean', 'lower_quantile', or 'worst_case'")


def combine_hidden_world_response_columns(
    payoff_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Treat each hidden-world response as a separate adversarial column.

    Worlds may expose different legal response counts. Keeping each world's
    responses as distinct columns yields a rectangular zero-sum matrix in
    which an adversary can choose any sampled world and any legal response in
    that world; no behavior-frequency assumption is introduced.
    """
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in payoff_matrices]
    if not matrices:
        raise ValueError("at least one hidden-world payoff matrix is required")
    row_count = matrices[0].shape[0] if matrices[0].ndim == 2 else 0
    if row_count < 1:
        raise ValueError("payoff matrices must be non-empty 2D arrays")
    for matrix in matrices:
        if matrix.ndim != 2 or not matrix.shape[1]:
            raise ValueError("payoff matrices must be non-empty 2D arrays")
        if matrix.shape[0] != row_count:
            raise ValueError("all hidden worlds must use the same own-action rows")
        if not np.isfinite(matrix).all():
            raise ValueError("payoff matrices must contain only finite values")
    return np.concatenate(matrices, axis=1)


def prune_candidate_actions(
    action_ids: Sequence[int],
    scores: Sequence[float],
    action_kinds: Mapping[int, Iterable[str]],
    *,
    max_actions: int,
    preserve_kinds: Sequence[str] = ("switch", "protect", "setup", "defensive"),
) -> list[int]:
    """Keep top-scoring candidates while preserving key strategic classes.

    ``scores`` are a policy prior or cheap heuristic only. The returned
    candidates should still receive equal-budget payoff evaluation in search.
    If a class is present among legal actions, at least its highest-scoring
    representative survives pruning.
    """
    if len(action_ids) != len(scores):
        raise ValueError("action_ids and scores must have equal lengths")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise ValueError("scores must be finite")
    unique_ids = list(dict.fromkeys(int(action) for action in action_ids))
    if len(unique_ids) != len(action_ids):
        raise ValueError("action_ids must be unique")
    score_by_id = {int(action): float(score) for action, score in zip(action_ids, scores)}
    ranked = sorted(unique_ids, key=lambda action: (-score_by_id[action], action))
    kept: list[int] = []
    for kind in preserve_kinds:
        eligible = [action for action in unique_ids if kind in set(action_kinds.get(action, ()))]
        if eligible:
            best = max(eligible, key=lambda action: (score_by_id[action], -action))
            if best not in kept:
                kept.append(best)
    for action in ranked:
        if action not in kept and len(kept) < max_actions:
            kept.append(action)
    if len(kept) > max_actions:
        # A search budget smaller than the number of protected categories is
        # a configuration error; silently dropping one would defeat pruning's
        # explicit strategic-preservation guarantee.
        raise ValueError("max_actions is smaller than the present preserved action classes")
    return kept
