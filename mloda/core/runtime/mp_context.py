from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from importlib.util import find_spec
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from multiprocessing.context import ForkServerContext, SpawnContext

logger = logging.getLogger(__name__)
_warned_forkserver_unavailable = False
_checked_preload: set[str] = set()


def mp_start_context() -> SpawnContext | ForkServerContext:
    """Return the spawn (default) or forkserver context chosen by MLODA_MP_START_METHOD.

    Never fork: a child inherits locks held by the parent's live threads and deadlocks.
    Forkserver is safe as it forks from a single-threaded server, so preloads must not start threads."""
    global _warned_forkserver_unavailable
    raw = os.environ.get("MLODA_MP_START_METHOD", "")
    method = raw.strip().lower()
    if method in ("", "spawn"):
        return multiprocessing.get_context("spawn")
    if method != "forkserver":
        raise ValueError(
            f"MLODA_MP_START_METHOD must be one of 'spawn', 'forkserver', got {raw!r}; "
            "'fork' is unsupported because forked children of a threaded parent can deadlock"
        )
    if "forkserver" not in multiprocessing.get_all_start_methods():
        if not _warned_forkserver_unavailable:
            _warned_forkserver_unavailable = True
            logger.warning("forkserver is unavailable on this platform, falling back to spawn")
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context("forkserver")
    preload = [m.strip() for m in os.environ.get("MLODA_MP_PRELOAD", "").split(",") if m.strip()]
    for name in preload:
        top = name.split(".")[0]
        if top in _checked_preload or top in sys.modules:
            continue
        _checked_preload.add(top)
        if find_spec(top) is None:
            logger.warning("MLODA_MP_PRELOAD entry %r: top-level module %r not found", name, top)
    if preload:
        ctx.set_forkserver_preload(preload)
    return ctx


def start_daemon_process(
    ctx: SpawnContext | ForkServerContext, target: Callable[..., None], args: tuple[Any, ...] = ()
) -> BaseProcess:
    """Creates ctx.Process(target, args, daemon=True): reaped automatically if the parent exits
    cleanly without an explicit join/terminate on this process."""
    return ctx.Process(target=target, args=args, daemon=True)
