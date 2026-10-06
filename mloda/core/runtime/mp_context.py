from __future__ import annotations

import logging
import multiprocessing
import os
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any, Callable

logger = logging.getLogger(__name__)


def mp_start_context() -> BaseContext:
    """Return a multiprocessing context chosen by MLODA_MP_START_METHOD (spawn default, or forkserver).

    Never fork: it deadlocks children when the parent has live threads (xdist, asyncio).
    MLODA_MP_PRELOAD only applies when the forkserver starts (first use in the process).
    """
    method = os.environ.get("MLODA_MP_START_METHOD", "")
    if method in ("", "spawn"):
        return multiprocessing.get_context("spawn")
    if method != "forkserver":
        raise ValueError(f"MLODA_MP_START_METHOD must be one of 'spawn', 'forkserver', got {method!r}")
    if "forkserver" not in multiprocessing.get_all_start_methods():
        logger.warning("forkserver is unavailable on this platform, falling back to spawn")
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context("forkserver")
    preload = [m.strip() for m in os.environ.get("MLODA_MP_PRELOAD", "").split(",") if m.strip()]
    if preload:
        ctx.set_forkserver_preload(preload)  # type: ignore[attr-defined]
    return ctx


def spawn_daemon_process(ctx: BaseContext, target: Callable[..., None], args: tuple[Any, ...] = ()) -> BaseProcess:
    """Creates ctx.Process(target, args, daemon=True): reaped automatically if the parent exits
    cleanly without an explicit join/terminate on this process."""
    return ctx.Process(target=target, args=args, daemon=True)
