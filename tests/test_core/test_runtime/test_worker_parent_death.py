"""A worker spawned via WorkerManager must exit once the real OS process that spawned it is killed."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.runtime.mp_context import mp_spawn_context
from mloda.core.runtime.worker.multiprocessing_worker import worker
from mloda.core.runtime.worker_manager import WorkerManager
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _FakeCfwManager:
    """Plain, picklable stand-in for CfwManager; Mock objects do not reliably pickle across spawn."""

    def get_location(self) -> str:
        return "grpc://localhost:9999"

    def get_run_context(self) -> RunContext:
        return RunContext()


def _worker_is_gone(pid: int) -> bool:
    """A reparented child can sit as an unreaped zombie, which still answers os.kill(pid, 0);
    /proc state distinguishes that from actually running."""
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return True
    return stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def _fake_parent_main(pid_file: str) -> None:
    """Spawns a real worker() child via WorkerManager, writes its pid to pid_file, then blocks for SIGKILL."""
    manager = WorkerManager()
    cfw_register = _FakeCfwManager()
    cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

    process, _, _ = manager.create_worker_process(
        cfw_uuid=uuid4(),
        target=worker,
        args=(cfw_register, cfw, uuid4()),
    )

    with open(pid_file, "w") as pid_handle:
        pid_handle.write(str(process.pid))

    # Non-daemonic (it has its own child); if the test process dies uncleanly,
    # multiprocessing's exit handling would join this for the full sleep duration.
    time.sleep(20)


class TestWorkerProcessDoesNotOutliveASigkilledParent:
    """A worker spawned via WorkerManager.create_worker_process must exit once its parent dies."""

    @pytest.mark.timeout(30)
    def test_real_worker_process_exits_after_its_real_parent_process_is_sigkilled(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "worker.pid"
        ctx = mp_spawn_context()
        fake_parent = ctx.Process(target=_fake_parent_main, args=(str(pid_file),))
        fake_parent.start()

        worker_pid: int | None = None
        worker_gone = False
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and worker_pid is None:
                if pid_file.exists():
                    content = pid_file.read_text().strip()
                    if content:
                        worker_pid = int(content)
                        break
                time.sleep(0.05)
            assert worker_pid is not None, (
                f"fake parent process never reported the worker's pid (fake_parent.exitcode={fake_parent.exitcode})"
            )

            assert fake_parent.pid is not None, "fake parent process has no pid"
            fake_parent_pid = fake_parent.pid
            os.kill(fake_parent_pid, signal.SIGKILL)
            fake_parent.join(timeout=5)
            assert not fake_parent.is_alive(), "fake parent process survived SIGKILL"

            deadline = time.time() + 10.0
            while time.time() < deadline:
                if _worker_is_gone(worker_pid):
                    worker_gone = True
                    break
                time.sleep(0.1)

            assert worker_gone, "worker process outlived its SIGKILLed parent process"
        finally:
            # Never leak either process, even if an assertion above failed.
            if fake_parent.is_alive():
                fake_parent.kill()
                fake_parent.join(timeout=5)
            if worker_pid is not None and not worker_gone:
                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except OSError:
                    pass
