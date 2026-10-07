"""Tests for mp_start_context: env-selected start method and forkserver preload."""

import importlib.util
import logging
import multiprocessing
import multiprocessing.context
import sys
from unittest.mock import Mock

import pytest

from mloda.core.runtime.mp_context import mp_start_context
from tests.conftest import MP_PRELOAD_MODULES

FORKSERVER_AVAILABLE = "forkserver" in multiprocessing.get_all_start_methods()
ONCE_FLAG = "mloda.core.runtime.mp_context._warned_forkserver_unavailable"
CHECKED_PRELOAD = "mloda.core.runtime.mp_context._checked_preload"


@pytest.mark.parametrize("value", [None, "", "spawn", " spawn ", "SPAWN"])
def test_spawn_is_default(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("MLODA_MP_START_METHOD", raising=False)
    else:
        monkeypatch.setenv("MLODA_MP_START_METHOD", value)

    assert mp_start_context().get_start_method() == "spawn"


@pytest.mark.skipif(not FORKSERVER_AVAILABLE, reason="forkserver unavailable")
@pytest.mark.parametrize("value", ["forkserver", " ForkServer "])
def test_forkserver_selected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", value)
    monkeypatch.delenv("MLODA_MP_PRELOAD", raising=False)

    assert mp_start_context().get_start_method() == "forkserver"


def test_forkserver_unavailable_falls_back_to_spawn_with_single_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ONCE_FLAG, False)
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    with caplog.at_level(logging.WARNING, logger="mloda.core.runtime.mp_context"):
        first = mp_start_context()
        second = mp_start_context()

    assert first.get_start_method() == "spawn"
    assert second.get_start_method() == "spawn"
    records = [r for r in caplog.records if r.name == "mloda.core.runtime.mp_context" and r.levelno == logging.WARNING]
    assert len(records) == 1


@pytest.mark.parametrize("value", ["fork", "bogus"])
def test_unsupported_value_raises(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", value)

    with pytest.raises(ValueError, match="spawn") as exc_info:
        mp_start_context()

    assert "forkserver" in str(exc_info.value)
    assert "deadlock" in str(exc_info.value)


@pytest.mark.skipif(not FORKSERVER_AVAILABLE, reason="forkserver unavailable")
@pytest.mark.parametrize(
    "raw, expected, warned, looked_up",
    [
        (
            " mloda_no_such_a, mloda_no_such_b ,,",
            ["mloda_no_such_a", "mloda_no_such_b"],
            [("mloda_no_such_a", "mloda_no_such_a"), ("mloda_no_such_b", "mloda_no_such_b")],
            ["mloda_no_such_a", "mloda_no_such_b"],
        ),
        ("json, os.path", ["json", "os.path"], [], []),
        (
            "json,mloda_no_such_pkg.sub",
            ["json", "mloda_no_such_pkg.sub"],
            [("mloda_no_such_pkg.sub", "mloda_no_such_pkg")],
            ["mloda_no_such_pkg"],
        ),
        ("__main__,json", ["__main__", "json"], [], []),
        (
            "mloda_no_such_c.x,mloda_no_such_c.y",
            ["mloda_no_such_c.x", "mloda_no_such_c.y"],
            [("mloda_no_such_c.x", "mloda_no_such_c")],
            ["mloda_no_such_c"],
        ),
        (
            "mloda_fake_found.sub,mloda_fake_found",
            ["mloda_fake_found.sub", "mloda_fake_found"],
            [],
            ["mloda_fake_found"],
        ),
    ],
)
def test_preload_forwarded_to_forkserver(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: list[str],
    warned: list[tuple[str, str]],
    looked_up: list[str],
) -> None:
    monkeypatch.setattr(CHECKED_PRELOAD, set())
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None)
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setenv("MLODA_MP_PRELOAD", raw)
    preload = Mock()
    monkeypatch.setattr("multiprocessing.forkserver.set_forkserver_preload", preload)
    find_spec_mock = Mock(side_effect=lambda n: object() if n == "mloda_fake_found" else importlib.util.find_spec(n))
    monkeypatch.setattr("mloda.core.runtime.mp_context.find_spec", find_spec_mock)

    with caplog.at_level(logging.WARNING, logger="mloda.core.runtime.mp_context"):
        mp_start_context()
        mp_start_context()

    preload.assert_called_with(expected)
    assert preload.call_count == 2
    records = [r for r in caplog.records if r.name == "mloda.core.runtime.mp_context" and r.levelno == logging.WARNING]
    assert [r.args for r in records] == warned
    assert [c.args[0] for c in find_spec_mock.call_args_list] == looked_up


def test_module_loads_without_forkserver_context_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(multiprocessing.context, "ForkServerContext", raising=False)
    spec = importlib.util.find_spec(mp_start_context.__module__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.mp_start_context)


@pytest.mark.skipif(not FORKSERVER_AVAILABLE, reason="forkserver unavailable")
def test_empty_preload_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setenv("MLODA_MP_PRELOAD", " , ")
    preload = Mock()
    monkeypatch.setattr("multiprocessing.forkserver.set_forkserver_preload", preload)

    mp_start_context()

    preload.assert_not_called()


def test_preload_ignored_under_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", "spawn")
    monkeypatch.setenv("MLODA_MP_PRELOAD", "a,b")
    preload = Mock()
    monkeypatch.setattr("multiprocessing.forkserver.set_forkserver_preload", preload)

    mp_start_context()

    preload.assert_not_called()


@pytest.mark.parametrize("module", MP_PRELOAD_MODULES)
def test_conftest_preload_modules_resolve(module: str) -> None:
    assert importlib.util.find_spec(module) is not None
