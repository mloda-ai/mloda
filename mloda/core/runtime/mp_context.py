from __future__ import annotations

import multiprocessing
from multiprocessing.context import BaseContext


def mp_spawn_context() -> BaseContext:
    """Return a fresh spawn-based multiprocessing context.

    mloda forks several helper processes (Manager, worker, Flight server)
    at runtime. On Linux, fork inherits the parent's threads' held locks
    but not the threads themselves, which deadlocks the child if the
    parent has any live background threads (pytest-xdist execnet,
    asyncio loops, observability agents). Spawn starts a fresh
    interpreter for each child and avoids this entire class of bug.
    """
    return multiprocessing.get_context("spawn")
