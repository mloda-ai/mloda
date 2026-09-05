"""Tests that pyproject.toml declares the correct optional-dependency extras.

These tests parse pyproject.toml directly (no subprocess needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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


def test_otel_extra_exists_and_aliases_registry_package() -> None:
    """An otel optional-dependency extra must exist and be a pure alias of mloda-community-otel."""
    optional = _load_optional_deps()
    assert "otel" in optional, f"No otel extra found in pyproject.toml. Available extras: {list(optional.keys())}"
    otel_deps = optional["otel"]
    assert len(otel_deps) == 1, f"Expected otel extra to be a single-package alias. Got: {otel_deps}"
    canonical_names = {canonicalize_name(Requirement(dep).name) for dep in otel_deps}
    assert canonicalize_name("mloda-community-otel") in canonical_names, (
        f"Expected mloda-community-otel in otel extra. Got: {otel_deps}"
    )


def test_all_extra_references_sqlite() -> None:
    """The all extra must reference mloda[sqlite]."""
    optional = _load_optional_deps()
    assert "all" in optional, "No all extra found in pyproject.toml"
    all_deps = optional["all"]
    assert any("mloda[sqlite]" in dep for dep in all_deps), f"Expected mloda[sqlite] in all extra. Got: {all_deps}"
