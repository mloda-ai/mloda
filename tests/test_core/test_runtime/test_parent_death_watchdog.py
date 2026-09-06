"""start_parent_death_watchdog() must os._exit() once the real parent process's join() returns,
and must do nothing when there is no parent process to watch.
"""

import multiprocessing
import os
import time
from unittest.mock import Mock

import pytest

from mloda.core.runtime.parent_death_watchdog import start_parent_death_watchdog


class TestWatchdogExitsOnceTheParentProcessJoinReturns:
    """A background thread waits on parent_process().join(); once that returns, the process must exit."""

    @pytest.mark.timeout(5)
    def test_os_exit_is_called_once_the_fake_parent_join_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_parent = Mock()
        fake_parent.join.return_value = None
        monkeypatch.setattr(multiprocessing, "parent_process", lambda: fake_parent)
        fake_exit = Mock()
        monkeypatch.setattr(os, "_exit", fake_exit)

        start_parent_death_watchdog()

        deadline = time.time() + 3.0
        while time.time() < deadline and not fake_exit.called:
            time.sleep(0.02)

        fake_exit.assert_called_once_with(0)


class TestWatchdogDoesNothingWithoutARealParentProcess:
    """When multiprocessing.parent_process() is None, the watchdog must never call os._exit."""

    @pytest.mark.timeout(5)
    def test_os_exit_is_never_called_when_there_is_no_parent_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
        fake_exit = Mock()
        monkeypatch.setattr(os, "_exit", fake_exit)

        start_parent_death_watchdog()

        # Give the background thread a moment to call os._exit, if it were going to.
        time.sleep(0.3)

        fake_exit.assert_not_called()
