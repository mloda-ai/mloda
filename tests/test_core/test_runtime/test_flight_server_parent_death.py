"""The Flight server process spawned by ParallelRunnerFlightServer must exit once the real OS
process that spawned it is killed, instead of surviving as a reparented orphan.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.core.runtime.mp_context import mp_spawn_context


def _flight_server_is_gone(pid: int) -> bool:
    """A reparented child can sit as an unreaped zombie, which still answers os.kill(pid, 0);
    /proc state distinguishes that from actually running."""
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return True
    return stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def _fake_parent_main(pid_file: str) -> None:
    server = ParallelRunnerFlightServer()
    server.start_flight_server_process()

    with open(pid_file, "w") as pid_handle:
        pid_handle.write(str(server.flight_server_process.pid))

    # Non-daemonic (it has its own child); if the test process dies uncleanly,
    # multiprocessing's exit handling would join this for the full sleep duration.
    time.sleep(20)


class TestFlightServerProcessDoesNotOutliveASigkilledParent:
    @pytest.mark.timeout(45)
    def test_real_flight_server_process_exits_after_its_real_parent_process_is_sigkilled(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "flight_server.pid"
        ctx = mp_spawn_context()
        fake_parent = ctx.Process(target=_fake_parent_main, args=(str(pid_file),))
        fake_parent.start()

        flight_server_pid: int | None = None
        flight_server_gone = False
        try:
            deadline = time.time() + 15.0
            while time.time() < deadline and flight_server_pid is None:
                if pid_file.exists():
                    content = pid_file.read_text().strip()
                    if content:
                        flight_server_pid = int(content)
                        break
                time.sleep(0.05)
            assert flight_server_pid is not None, (
                f"fake parent process never reported the flight server's pid "
                f"(fake_parent.exitcode={fake_parent.exitcode})"
            )

            assert fake_parent.pid is not None, "fake parent process has no pid"
            fake_parent_pid = fake_parent.pid
            os.kill(fake_parent_pid, signal.SIGKILL)
            fake_parent.join(timeout=5)
            assert not fake_parent.is_alive(), "fake parent process survived SIGKILL"

            deadline = time.time() + 10.0
            while time.time() < deadline:
                if _flight_server_is_gone(flight_server_pid):
                    flight_server_gone = True
                    break
                time.sleep(0.1)

            assert flight_server_gone, "flight server process outlived its SIGKILLed parent process"
        finally:
            # Never leak either process, even if an assertion above failed.
            if fake_parent.is_alive():
                fake_parent.kill()
                fake_parent.join(timeout=5)
            if flight_server_pid is not None and not flight_server_gone:
                try:
                    os.kill(flight_server_pid, signal.SIGKILL)
                except OSError:
                    pass
