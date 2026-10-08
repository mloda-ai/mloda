"""Regression tests for positional column appends on re-executed lazy plans.

Covers the DuckDB ``append_column`` implementation, which must keep values
aligned with the original row order even when the underlying relation is
re-executed (e.g. after a filter or a second materialization).
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import duckdb  # noqa: E402

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation  # noqa: E402


def _make_relation() -> DuckdbRelation:
    connection = duckdb.connect()
    return DuckdbRelation.from_dict(connection, {"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_append_column_aligns_values() -> None:
    relation = _make_relation()
    result = relation.append_column("c", [10, 20, 30])

    assert result.columns == ["a", "b", "c"]
    assert result.df().to_dict(orient="list") == {
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [10, 20, 30],
    }


def test_append_column_re_executed_plan_stays_aligned() -> None:
    relation = _make_relation()
    appended = relation.append_column("c", [10, 20, 30])

    # Re-execute the lazy plan multiple times; positional alignment must hold.
    first = appended.df().to_dict(orient="list")
    second = appended.df().to_dict(orient="list")

    assert first == second
    assert first["c"] == [10, 20, 30]


def test_append_column_after_filter_keeps_alignment() -> None:
    relation = _make_relation()
    filtered = relation.filter("a > 1")
    result = filtered.append_column("c", [20, 30])

    assert result.df().to_dict(orient="list") == {
        "a": [2, 3],
        "b": ["y", "z"],
        "c": [20, 30],
    }


def test_append_column_rejects_duplicate_name() -> None:
    relation = _make_relation()
    with pytest.raises(ValueError):
        relation.append_column("a", [1, 2, 3])
