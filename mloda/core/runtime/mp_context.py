from __future__ import annotations

import multiprocessing
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from typing import Any, Callable


def mp_spawn_context() -> SpawnContext:
    """Return a fresh spawn-based multiprocessing context.

    mloda forks several helper processes (Manager, worker, Flight server)
    at runtime. On Linux, fork inherits the parent's threads' held locks
    but not the threads themselves, which deadlocks the child if the
    parent has any live background threads (pytest-xdist execnet,
    asyncio loops, observability agents). Spawn starts a fresh
    interpreter for each child and avoids this entire class of bug.
    """
    return multiprocessing.get_context("spawn")


def spawn_daemon_process(ctx: SpawnContext, target: Callable[..., None], args: tuple[Any, ...] = ()) -> BaseProcess:
    """Creates ctx.Process(target, args, daemon=True): reaped automatically if the parent exits
    cleanly without an explicit join/terminate on this process."""
    return ctx.Process(target=target, args=args, daemon=True)
