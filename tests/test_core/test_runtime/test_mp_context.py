"""Tests for mp_start_context: env-selected start method and forkserver preload."""

import importlib.util
import pathlib
import logging
import multiprocessing
import multiprocessing.context
from unittest.mock import Mock

import pytest

from mloda.core.runtime.mp_context import mp_start_context
from mloda.core.runtime import mp_context
from tests.conftest import MP_PRELOAD_MODULES

FORKSERVER_AVAILABLE = "forkserver" in multiprocessing.get_all_start_methods()
ONCE_FLAG = "mloda.core.runtime.mp_context._warned_forkserver_unavailable"
WARNED_PRELOAD = "mloda.core.runtime.mp_context._warned_missing_preload"


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


def test_forkserver_unavailable_falls_back_to_spawn_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ONCE_FLAG, False, raising=False)
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    with caplog.at_level(logging.WARNING):
        ctx = mp_start_context()

    assert ctx.get_start_method() == "spawn"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_forkserver_unavailable_warns_once_per_process(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ONCE_FLAG, False, raising=False)
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    with caplog.at_level(logging.WARNING):
        mp_start_context()
        mp_start_context()

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize("value", ["fork", "bogus"])
def test_unsupported_value_raises(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MLODA_MP_START_METHOD", value)

    with pytest.raises(ValueError, match="spawn") as exc_info:
        mp_start_context()

    assert "forkserver" in str(exc_info.value)


@pytest.mark.skipif(not FORKSERVER_AVAILABLE, reason="forkserver unavailable")
@pytest.mark.parametrize(
    "raw, expected, missing",
    [
        (" a, b ,,", ["a", "b"], ["a", "b"]),
        ("json, os.path", ["json", "os.path"], []),
        ("json,mloda_no_such_pkg.sub", ["json", "mloda_no_such_pkg.sub"], ["mloda_no_such_pkg.sub"]),
    ],
)
def test_preload_forwarded_to_forkserver(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: list[str],
    missing: list[str],
) -> None:
    monkeypatch.setattr(WARNED_PRELOAD, set())
    monkeypatch.setenv("MLODA_MP_START_METHOD", "forkserver")
    monkeypatch.setenv("MLODA_MP_PRELOAD", raw)
    preload = Mock()
    monkeypatch.setattr("multiprocessing.forkserver.set_forkserver_preload", preload)

    with caplog.at_level(logging.WARNING, logger="mloda.core.runtime.mp_context"):
        mp_start_context()
        mp_start_context()

    preload.assert_called_with(expected)
    assert preload.call_count == 2
    records = [r for r in caplog.records if r.name == "mloda.core.runtime.mp_context" and r.levelno == logging.WARNING]
    assert [r.args[0] for r in records] == missing  # type: ignore[index]


def test_module_loads_without_forkserver_context_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(multiprocessing.context, "ForkServerContext")
    source = pathlib.Path(mp_context.__file__)
    spec = importlib.util.spec_from_file_location("_mp_context_windows_probe", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


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
