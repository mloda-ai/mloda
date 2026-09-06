from __future__ import annotations

import multiprocessing
import os
import threading


def start_parent_death_watchdog() -> None:
    """Starts a daemon thread that exits this process once its real parent process is gone."""

    def _watch() -> None:
        parent = multiprocessing.parent_process()
        if parent is None:
            return
        parent.join()
        os._exit(0)

    threading.Thread(target=_watch, daemon=True).start()
