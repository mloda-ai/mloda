"""Uniform import policy for the mloda.user backend modules (issue #736).

Policy: importing ``mloda.user.<backend>`` NEVER raises ModuleNotFoundError because an optional
backend library is missing. The framework class always imports; a framework whose library is
missing reports itself unavailable through ``is_available()``, and core discovery
(``PreFilterPlugins.get_cfw_subclasses``) excludes it. This is what
``docs/docs/chapter1/compute-frameworks.md`` already promises and what the registry guide
documents (guarded module-level import plus ``is_available()``).

The test lives next to the other mloda.user backend-module contracts (``test_compute_framework_exports``)
rather than in ``tests/test_core/test_optional_pyarrow/``: the policy spans pandas, polars, duckdb,
pyspark and pyiceberg, so it is not pyarrow-scoped, and it must be enforced by the default tox env.

Each case runs in a subprocess with the library blocked by a ``sys.meta_path`` finder, so the
result does not depend on which extras the current env happens to install.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from tests.test_core.test_abstract_plugins.test_plugin_registry.test_compute_framework_exports import (
    _package_directory,
)
from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked


class Backend(NamedTuple):
    """A mloda.user backend module, the library to block, and the availability that must follow."""

    module: str
    library: str
    classes: tuple[str, ...]
    available: bool


# python_dict has no optional backend, so nothing of its own can be blocked: pyarrow stands in as an
# unrelated library, and blocking it must leave PythonDictFramework available.
# sqlite is backed by the stdlib sqlite3, but its extra declares pyarrow (the relation and merge
# engine need it), so pyarrow is the library that decides its availability.
# duckdb gets two rows: data only reaches the framework through Arrow, so pyarrow decides its
# availability as much as duckdb does, and both must be blockable to False.
# Every mloda.user backend module needs at least one row here; the set is pinned against the
# directory below, so a new backend cannot land without a policy row.
BACKENDS: list[Backend] = [
    Backend("mloda.user.pandas", "pandas", ("PandasDataFrame",), False),
    Backend("mloda.user.pyarrow", "pyarrow", ("PyArrowTable",), False),
    Backend("mloda.user.polars", "polars", ("PolarsDataFrame", "PolarsLazyDataFrame"), False),
    Backend("mloda.user.duckdb", "duckdb", ("DuckDBFramework",), False),
    Backend("mloda.user.duckdb", "pyarrow", ("DuckDBFramework",), False),
    Backend("mloda.user.spark", "pyspark", ("SparkFramework",), False),
    Backend("mloda.user.iceberg", "pyiceberg", ("IcebergFramework",), False),
    Backend("mloda.user.sqlite", "pyarrow", ("SqliteFramework",), False),
    Backend("mloda.user.python_dict", "pyarrow", ("PythonDictFramework",), True),
]

_AVAILABILITY_IDS = [f"{backend.module}:no-{backend.library}" for backend in BACKENDS]

# One import case per module: the pyarrow-blocked import check does not depend on which library a row blocks.
_IMPORT_CASES: list[Backend] = list({backend.module: backend for backend in BACKENDS}.values())

_PYARROW_IDS = [backend.module for backend in _IMPORT_CASES]

# Imports the backend module, then pins is_available() and core discovery membership for every
# class it exports. A missing library must surface here, never as an import error.
_AVAILABILITY_BODY: str = """
import importlib

from mloda.core.prepare.accessible_plugins import PreFilterPlugins

module = importlib.import_module("{module}")

discovered = PreFilterPlugins.get_cfw_subclasses()

for name in {classes}:
    cls = getattr(module, name, None)
    assert cls is not None, "{module} must export " + name

    available = cls.is_available()
    assert available is {available}, (
        name + ".is_available() must be {available} while {library} is blocked, got " + repr(available)
    )

    assert (cls in discovered) is {available}, (
        "core discovery must " + ("include " if {available} else "exclude ") + name
        + " while {library} is blocked"
    )

print("OK")
"""

_IMPORT_BODY: str = """
import importlib

module = importlib.import_module("{module}")

for name in {classes}:
    assert getattr(module, name, None) is not None, "{module} must export " + name

print("OK")
"""


@pytest.mark.timeout(30)
@pytest.mark.parametrize("backend", BACKENDS, ids=_AVAILABILITY_IDS)
def test_backend_module_imports_and_reports_availability_without_its_library(backend: Backend) -> None:
    """With the library blocked: the module imports, exports its classes, and is_available() decides."""
    body = _AVAILABILITY_BODY.format(
        module=backend.module,
        classes=repr(list(backend.classes)),
        library=backend.library,
        available=backend.available,
    )
    result = run_blocked(body, module=backend.library)

    assert result.returncode == 0, (
        f"import {backend.module} must not raise while '{backend.library}' is missing: a framework whose "
        f"library is absent stays importable and reports is_available() == {backend.available}; core "
        f"discovery filters it out.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout.splitlines(), f"Expected OK sentinel. Got stdout:\n{result.stdout}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("backend", _IMPORT_CASES, ids=_PYARROW_IDS)
def test_backend_module_imports_with_pyarrow_blocked(backend: Backend) -> None:
    """pyarrow is the library that leaks across backends (duckdb, sqlite, iceberg and spark all
    reference it), so every backend module must import with pyarrow blocked, not just with its own
    library blocked.
    """
    body = _IMPORT_BODY.format(module=backend.module, classes=repr(list(backend.classes)))
    result = run_blocked(body, module="pyarrow")

    assert result.returncode == 0, (
        f"import {backend.module} must not raise while pyarrow is missing: pyarrow is an optional extra, "
        f"so no backend module may require it at import time.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout.splitlines(), f"Expected OK sentinel. Got stdout:\n{result.stdout}"


def _shipped_backend_modules() -> set[str]:
    """The mloda.user.<backend> modules that exist on disk, read from the package directory."""
    sources: list[Path] = sorted(_package_directory("mloda.user").glob("*.py"))
    return {f"mloda.user.{source.stem}" for source in sources if source.name != "__init__.py"}


def test_every_shipped_backend_module_is_covered_by_the_policy() -> None:
    """The parametrization is derived, not written down: a new backend without a row fails here.

    Without this, adding mloda/user/<new>.py with an eager backend import passes the suite: the policy
    test has no row for it, and the completeness test in test_compute_framework_exports cannot catch it
    either, because PluginLoader silently skips a plugin module whose optional root import fails.
    """
    covered = {backend.module for backend in BACKENDS}
    shipped = _shipped_backend_modules()

    assert shipped == covered, (
        f"Every mloda.user backend module needs a BACKENDS row stating which library decides its "
        f"availability. Missing rows: {sorted(shipped - covered)}. Rows for modules that do not exist: "
        f"{sorted(covered - shipped)}."
    )
