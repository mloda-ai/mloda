"""Contract of mloda.core.optional_dependency.require: import at point of use, or name the extra to install.

The install hint is only correct for an ABSENT module. A module that is installed but fails to load (broken
.so, ABI mismatch, missing transitive dependency) must report its own error: telling the user to install what
they already have hides the real fault.
"""

from __future__ import annotations

import json
import subprocess  # nosec
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from mloda.core.optional_dependency import require

# A distribution that is guaranteed not to be installed.
_ABSENT = "mloda_absent_backend_for_tests"


def _expected_message(distribution: str, reason: str) -> str:
    return f"{distribution} is required for {reason}. Install it with: pip install 'mloda[{distribution}]'"


def _unique_module_name(prefix: str) -> str:
    return f"mloda_test_{prefix}_{uuid.uuid4().hex}"


@pytest.fixture
def registered_modules() -> Iterator[list[str]]:
    """Names appended here are removed from sys.modules after the test."""
    names: list[str] = []
    yield names
    for name in names:
        sys.modules.pop(name, None)


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """Run body in a fresh interpreter (pyarrow present)."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_require_returns_the_module() -> None:
    """require returns the module object itself, not a proxy."""
    import json as json_module

    assert require("json", "JSON encoding") is json_module


def test_require_resolves_dotted_submodules() -> None:
    """require("a.b") returns the submodule a.b, not the root package."""
    import email.mime.text

    assert require("email.mime.text", "MIME encoding") is email.mime.text


def test_require_returns_pyarrow_when_installed() -> None:
    import pyarrow

    assert require("pyarrow", "Arrow type conversions") is pyarrow


def test_require_returns_pyarrow_flight_when_installed() -> None:
    import pyarrow.flight

    assert require("pyarrow.flight", "Flight-based (multiprocessing/distributed) data transport") is pyarrow.flight


def test_require_absent_module_raises_import_error_naming_the_extra() -> None:
    with pytest.raises(ImportError) as exc_info:
        require(_ABSENT, "some reason")

    assert str(exc_info.value) == _expected_message(_ABSENT, "some reason")


def test_require_absent_dotted_module_names_the_root_distribution() -> None:
    """The install hint is the root package, so pyarrow.flight resolves to mloda[pyarrow]."""
    with pytest.raises(ImportError) as exc_info:
        require(f"{_ABSENT}.sub.module", "some reason")

    assert str(exc_info.value) == _expected_message(_ABSENT, "some reason")


def test_require_preserves_the_original_import_error() -> None:
    with pytest.raises(ImportError) as exc_info:
        require(_ABSENT, "some reason")

    original = exc_info.value.__cause__ or exc_info.value.__context__
    assert isinstance(original, ImportError), f"original ImportError not preserved, got {original!r}"
    assert original is not exc_info.value


def test_require_treats_a_none_entry_in_sys_modules_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """sys.modules[name] = None is CPython's absence sentinel: importing it raises ModuleNotFoundError(name=name).

    Narrowing the rewrite to genuine absences must not lose this case, which is how the pyarrow blocker and
    the "not installed" test rigs across this tree simulate a missing backend.
    """
    name = _unique_module_name("none_entry")
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), name, None)

    with pytest.raises(ImportError) as exc_info:
        require(name, "some reason")

    assert str(exc_info.value) == _expected_message(name, "some reason")


def test_require_propagates_the_error_of_a_broken_but_installed_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed library whose body fails to load (missing .so, ABI mismatch) reports its OWN error.

    Rewriting this into "install it with pip install 'mloda[x]'" tells the user to install what they already
    have, and buries the real cause in __cause__ where nobody reads it.
    """
    name = _unique_module_name("broken")
    (tmp_path / f"{name}.py").write_text('raise ImportError("libbroken.so.1: cannot open shared object file")\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ImportError) as exc_info:
        require(name, "some reason")

    assert str(exc_info.value) == "libbroken.so.1: cannot open shared object file"
    assert "Install it with" not in str(exc_info.value), (
        f"{name} IS installed, it is broken. Reporting it as missing is wrong advice."
    )


def test_require_propagates_a_missing_transitive_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend is present, but ITS dependency is not: the fix is the dependency, not the mloda extra.

    Mirrors pyarrow being installed while its body raises ModuleNotFoundError for numpy.
    """
    name = _unique_module_name("present_backend")
    dependency = _unique_module_name("transitive_dep")
    (tmp_path / f"{name}.py").write_text(f"import {dependency}\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require(name, "some reason")

    assert exc_info.value.name == dependency
    assert dependency in str(exc_info.value)
    assert "Install it with" not in str(exc_info.value), (
        f"the module that is missing is {dependency}, not {name}. An install hint for {name} points the user "
        "at a package that is already installed."
    )


def test_require_missing_submodule_of_an_installed_package_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_modules: list[str]
) -> None:
    """pyarrow built without Flight: the root imports, the submodule does not. Here the extra IS the remedy.

    CPython reports ModuleNotFoundError(name="pyarrow.flight"), i.e. an absence under the distribution's own
    namespace, so it must still be rewritten to the friendly message naming the root distribution.
    """
    package = _unique_module_name("pkg")
    registered_modules.append(package)
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ImportError) as exc_info:
        require(f"{package}.flight", "Flight-based transport")

    assert str(exc_info.value) == _expected_message(package, "Flight-based transport")
    original = exc_info.value.__cause__
    assert isinstance(original, ModuleNotFoundError)
    assert original.name == f"{package}.flight"


# Reproduces the three messages the hand-rolled helpers raised before the refactor, byte for byte.
_LEGACY_CASES: list[tuple[str, str, str]] = [
    (
        "pyarrow",
        "Arrow type conversions",
        "pyarrow is required for Arrow type conversions. Install it with: pip install 'mloda[pyarrow]'",
    ),
    (
        "pyarrow",
        "this operation",
        "pyarrow is required for this operation. Install it with: pip install 'mloda[pyarrow]'",
    ),
    (
        "pyarrow.flight",
        "Flight-based (multiprocessing/distributed) data transport",
        "pyarrow is required for Flight-based (multiprocessing/distributed) data transport. "
        "Install it with: pip install 'mloda[pyarrow]'",
    ),
]

_LEGACY_BODY: str = """
import json

from mloda.core.optional_dependency import require

cases = [
    ("pyarrow", "Arrow type conversions"),
    ("pyarrow", "this operation"),
    ("pyarrow.flight", "Flight-based (multiprocessing/distributed) data transport"),
]

messages = []
for module, reason in cases:
    try:
        require(module, reason)
        messages.append("NO_RAISE")
    except ImportError as e:
        messages.append(str(e))

print("MESSAGES:" + json.dumps(messages))
"""


@pytest.mark.timeout(30)
def test_require_reproduces_the_legacy_messages_byte_for_byte() -> None:
    """require must emit the exact strings the three hand-rolled helpers used, so no caller-facing text changes."""
    from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

    result = run_blocked(_LEGACY_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    prefix = "MESSAGES:"
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert lines, f"Expected {prefix} sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"

    messages = json.loads(lines[0][len(prefix) :])
    expected = [message for _, _, message in _LEGACY_CASES]
    assert messages == expected


_NO_EAGER_IMPORT_BODY: str = """
import sys

import mloda.core.optional_dependency  # noqa: F401

backends = ("pyarrow", "pandas", "polars", "duckdb", "pyspark", "pyiceberg")
leaked = [name for name in backends if name in sys.modules]
print("LEAKED:" + ",".join(leaked))
"""


@pytest.mark.timeout(30)
def test_importing_optional_dependency_imports_no_backend() -> None:
    """Importing the module must not pull in any optional backend, even where they are installed."""
    result = _run(_NO_EAGER_IMPORT_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    prefix = "LEAKED:"
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert lines, f"Expected {prefix} sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"

    leaked = [name for name in lines[0][len(prefix) :].split(",") if name]
    assert leaked == [], f"mloda.core.optional_dependency eagerly imported: {leaked}"
