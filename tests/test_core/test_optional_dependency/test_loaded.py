"""Contract of mloda.core.optional_dependency.loaded: the module if already imported, else None. Never imports."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from mloda.core.optional_dependency import loaded


def _unique_module_name(prefix: str) -> str:
    return f"mloda_test_{prefix}_{uuid.uuid4().hex}"


@pytest.fixture
def registered_modules() -> Iterator[list[str]]:
    """Names appended here are removed from sys.modules after the test."""
    names: list[str] = []
    yield names
    for name in names:
        sys.modules.pop(name, None)


def test_loaded_returns_none_for_an_unknown_module() -> None:
    assert loaded(_unique_module_name("unknown")) is None


def test_loaded_does_not_import_an_importable_but_unimported_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_modules: list[str]
) -> None:
    """The hot-path answer "absent" must cost nothing: no import is attempted."""
    name = _unique_module_name("probe")
    registered_modules.append(name)
    marker = tmp_path / "body_executed.marker"
    (tmp_path / f"{name}.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert loaded(name) is None
    assert not marker.exists(), "loaded() executed the module body: it must never trigger an import"
    assert name not in sys.modules, "loaded() imported the module: it must never trigger an import"


def test_loaded_returns_none_when_sys_modules_entry_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None entry in sys.modules means absent, and a plain import of it would raise ModuleNotFoundError."""
    name = _unique_module_name("none_entry")
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), name, None)

    assert loaded(name) is None


def test_loaded_returns_an_already_imported_module() -> None:
    import json as json_module

    assert loaded("json") is json_module


def test_loaded_resolves_an_already_imported_dotted_submodule() -> None:
    import email.mime.text

    assert loaded("email.mime.text") is email.mime.text


def test_loaded_returns_pyarrow_once_imported() -> None:
    import pyarrow

    assert loaded("pyarrow") is pyarrow


@pytest.mark.timeout(30)
def test_loaded_never_returns_a_partially_initialized_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_modules: list[str]
) -> None:
    """An importing thread publishes a module to sys.modules BEFORE running its body.

    mloda runs ParallelizationMode.THREADING, so loaded() must resolve through an import (which waits on the
    per-module import lock) rather than handing back the raw sys.modules entry, which can be half-built.
    """
    name = _unique_module_name("slow")
    registered_modules.append(name)
    (tmp_path / f"{name}.py").write_text("import time\ntime.sleep(1.0)\nREADY = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def _import_in_thread() -> None:
        importlib.import_module(name)

    thread = threading.Thread(target=_import_in_thread)
    thread.start()

    # Wait until the module is published to sys.modules, which happens before its body runs.
    deadline = time.monotonic() + 5.0
    while name not in sys.modules and time.monotonic() < deadline:
        time.sleep(0.001)
    assert name in sys.modules, "the module was never published to sys.modules"

    module = loaded(name)
    # Snapshot now: the body is still running, and after the join the same module object would look complete.
    ready = getattr(module, "READY", False)
    thread.join(timeout=10.0)

    assert isinstance(module, ModuleType)
    assert ready is True, (
        "loaded() handed back a partially initialized module. It must resolve the module through an import, "
        "which blocks on the per-module import lock until a concurrent first import has finished."
    )
