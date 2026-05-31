"""Tests that pyproject.toml declares the correct optional-dependency extras.

These tests parse pyproject.toml directly (no subprocess needed).

Current (red) failures expected:
- duckdb extra does not list pyarrow
- no sqlite extra exists
- all extra does not reference mloda[sqlite]
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    """The duckdb optional-dependency extra must list pyarrow as a dependency.

    Current (red): duckdb = ["duckdb"] with no pyarrow entry.
    """
    optional = _load_optional_deps()
    assert "duckdb" in optional, "No duckdb extra found in pyproject.toml"
    duckdb_deps = optional["duckdb"]
    assert any("pyarrow" in dep for dep in duckdb_deps), f"Expected pyarrow in duckdb extra. Got: {duckdb_deps}"


def test_sqlite_extra_exists_and_contains_pyarrow() -> None:
    """A sqlite optional-dependency extra must exist and include pyarrow.

    Current (red): no sqlite extra exists in pyproject.toml.
    """
    optional = _load_optional_deps()
    assert "sqlite" in optional, f"No sqlite extra found in pyproject.toml. Available extras: {list(optional.keys())}"
    sqlite_deps = optional["sqlite"]
    assert any("pyarrow" in dep for dep in sqlite_deps), f"Expected pyarrow in sqlite extra. Got: {sqlite_deps}"


def test_all_extra_references_sqlite() -> None:
    """The all extra must reference mloda[sqlite].

    Current (red): all does not include mloda[sqlite].
    """
    optional = _load_optional_deps()
    assert "all" in optional, "No all extra found in pyproject.toml"
    all_deps = optional["all"]
    assert any("mloda[sqlite]" in dep for dep in all_deps), f"Expected mloda[sqlite] in all extra. Got: {all_deps}"
