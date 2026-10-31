"""Bound poke-env's synchronous queue reads during nonblocking cleanup."""

from __future__ import annotations

from typing import Any


def close_poke_env_nonblocking(env: Any) -> None:
    """Close a ``DoublesEnv`` without hanging on an async queue race.

    poke-env's ``DoublesEnv.close(wait=False)`` checks ``Queue.empty()`` and
    then performs a blocking ``Queue.get()``. Its async player may consume the
    item between those calls, blocking the synchronous caller indefinitely.
    Bound only the queue reads performed during this close; normal battle-time
    operations are unaffected. Call this only when the environment will be
    discarded after cleanup (for example, a timed-out episode).
    """
    queues = (
        env.agent1.order_queue,
        env.agent2.order_queue,
        env.agent1.battle_queue,
        env.agent2.battle_queue,
    )
    for queue in queues:
        original_get = queue.get

        def bounded_get(timeout: float | None = None, *, _get=original_get):
            try:
                return _get(timeout=0.1 if timeout is None else min(timeout, 0.1))
            except TimeoutError:
                return None

        queue.get = bounded_get
    env.close(force=True, wait=False)
