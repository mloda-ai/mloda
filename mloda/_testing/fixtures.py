"""Pytest fixtures for testing plugins that talk to a service.

Registered as the ``mloda-testing`` pytest11 entry point, so plugin test suites get the fixtures
without importing anything: ``loopback_server`` yields a recording loopback stand-in, and
``no_network_env`` applies the offline env preset to the test process and returns it for
subprocess launches.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mloda._testing import loopback as _loopback
from mloda._testing.loopback import LoopbackServer


@pytest.fixture
def loopback_server() -> Iterator[LoopbackServer]:
    """A LoopbackServer that is closed when the test ends; assert on its ``requests``."""
    with LoopbackServer() as server:
        yield server


@pytest.fixture
def no_network_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Apply the no-network preset to this process and return it for ``subprocess`` env use."""
    env = _loopback.no_network_env()
    for var in _loopback._PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DO_NOT_TRACK", env["DO_NOT_TRACK"])
    monkeypatch.setenv("NO_PROXY", env["NO_PROXY"])
    monkeypatch.setenv("no_proxy", env["no_proxy"])
    return env
