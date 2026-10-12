"""Tests that pyproject.toml declares the correct optional-dependency extras.

These tests parse pyproject.toml directly (no subprocess needed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_PYPROJECT = Path(__file__).parent.parent.parent.parent / "pyproject.toml"


def _load_optional_deps() -> dict[str, list[str]]:
    with open(_PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    result: dict[str, list[str]] = data["project"]["optional-dependencies"]
    return result


def test_duckdb_extra_contains_pyarrow() -> None:
    """The duckdb optional-dependency extra must list pyarrow as a dependency."""
    optional = _load_optional_deps()
    assert "duckdb" in optional, "No duckdb extra found in pyproject.toml"
    duckdb_deps = optional["duckdb"]
    assert any("pyarrow" in dep for dep in duckdb_deps), f"Expected pyarrow in duckdb extra. Got: {duckdb_deps}"


def test_sqlite_extra_exists_and_contains_pyarrow() -> None:
    """A sqlite optional-dependency extra must exist and include pyarrow."""
    optional = _load_optional_deps()
    assert "sqlite" in optional, f"No sqlite extra found in pyproject.toml. Available extras: {list(optional.keys())}"
    sqlite_deps = optional["sqlite"]
    assert any("pyarrow" in dep for dep in sqlite_deps), f"Expected pyarrow in sqlite extra. Got: {sqlite_deps}"


@pytest.mark.parametrize("referenced_extra", ["sqlite", "yaml"])
def test_all_extra_references(referenced_extra: str) -> None:
    """The all extra must reference the sqlite and yaml extras."""
    optional = _load_optional_deps()
    assert "all" in optional, "No all extra found in pyproject.toml"
    all_deps = optional["all"]
    assert any(f"mloda[{referenced_extra}]" in dep for dep in all_deps), (
        f"Expected mloda[{referenced_extra}] in all extra. Got: {all_deps}"
    )


def _load_project() -> dict[str, Any]:
    with open(_PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    result: dict[str, Any] = data["project"]
    return result


def test_core_dependencies_stay_empty() -> None:
    """Core installs without pyarrow, so project.dependencies must stay empty."""
    assert _load_project()["dependencies"] == []


def test_every_pyarrow_requirement_has_floor() -> None:
    """Every pyarrow requirement in any extra must carry the >=14.0.1 floor."""
    bare: dict[str, str] = {}
    found = 0
    for group, deps in _load_optional_deps().items():
        for dep in deps:
            if dep.startswith("mloda["):
                continue
            req = Requirement(dep)
            if req.name != "pyarrow":
                continue
            found += 1
            if ">=14.0.1" not in {str(spec) for spec in req.specifier}:
                bare[group] = dep
    assert found > 0, "No pyarrow requirement found in any extra"
    assert not bare, f"pyarrow requirements missing the >=14.0.1 floor: {bare}"


@pytest.mark.parametrize(
    ("extra", "dependency"),
    [
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("test", "pyyaml"),
        ("test", "pyarrow>=14.0.1"),
        ("pyarrow", "pyarrow>=14.0.1"),
        ("duckdb", "pyarrow>=14.0.1"),
        ("sqlite", "pyarrow>=14.0.1"),
        ("iceberg", "pyarrow>=14.0.1"),
        ("spark", "pyarrow>=14.0.1"),
    ],
)
def test_extra_exists_and_contains_dependency(extra: str, dependency: str) -> None:
    """The extras that install hints and tests rely on must exist and list their distribution."""
    optional = _load_optional_deps()
    assert extra in optional, f"No {extra} extra found in pyproject.toml. Available extras: {list(optional.keys())}"
    deps = optional[extra]
    assert any(dependency in dep for dep in deps), f"Expected {dependency} in {extra} extra. Got: {deps}"
