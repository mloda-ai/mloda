"""Tests for ComputeFramework._row_count, the per-framework row-counting hook used by
HookContext's rows_in/rows_out.

Pins that PythonDictFramework counts ROWS (not columns), that DuckDB/SQLite relations
never trigger their expensive __len__ query, and that the default _row_count still
delegates to HookContext.row_count unchanged.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FeatureGroup
from mloda.user import Feature, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class TestPythonDictRowCount:
    """PythonDictFramework._row_count counts ROWS (value-list length), not COLUMNS (dict keys)."""

    def test_counts_rows_not_columns(self) -> None:
        fw = PythonDictFramework()
        assert fw._row_count({"a": [1, 2, 3], "b": [4, 5, 6]}) == 3

    def test_single_column_zero_rows(self) -> None:
        fw = PythonDictFramework()
        assert fw._row_count({"a": []}) == 0

    def test_schemaless_empty_dict(self) -> None:
        """``{}`` (zero columns, PythonDictFramework's schema-less value) has no data at all: 0 rows."""
        fw = PythonDictFramework()
        assert fw._row_count({}) == 0


class _RowCountFeatureGroup(FeatureGroup):
    """Root feature group returning a 2-column, 5-row dict."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}


class _ContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


def _build_feature_set() -> FeatureSet:
    return FeatureSet([Feature("my_feature")])


def _build_framework(extenders: set[Extender]) -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender=extenders)


class TestRowsOutEndToEnd:
    """Regression: rows_out through the real hook wiring must count rows, not columns."""

    def test_rows_out_counts_rows_not_columns(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = _build_framework({extender})

        cfw.run_calculate_feature(_RowCountFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.rows_out == 5


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")
class TestDuckDBRowCountStaysLazy:
    """DuckDBFramework._row_count must never trigger the relation's expensive count_star() query."""

    def test_row_count_does_not_call_dunder_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})
        relation = DuckdbRelation.from_arrow(conn, arrow_table)

        def _raise_if_called(self: Any) -> int:
            raise AssertionError("DuckdbRelation.__len__ must not be called by _row_count")

        monkeypatch.setattr(DuckdbRelation, "__len__", _raise_if_called)

        result = DuckDBFramework()._row_count(relation)

        assert result is None


class TestSqliteRowCountStaysLazy:
    """SqliteFramework._row_count must never trigger the relation's expensive SELECT COUNT(*) query."""

    def test_row_count_does_not_call_dunder_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})
        relation = SqliteRelation.from_arrow(conn, arrow_table)

        def _raise_if_called(self: Any) -> int:
            raise AssertionError("SqliteRelation.__len__ must not be called by _row_count")

        monkeypatch.setattr(SqliteRelation, "__len__", _raise_if_called)

        result = SqliteFramework()._row_count(relation)

        assert result is None


class TestDefaultRowCountDelegatesUnchanged:
    """The base (unoverridden) ComputeFramework._row_count behaves like today's HookContext.row_count."""

    def test_default_row_count_on_sized_and_unsized_objects(self) -> None:
        cf = ComputeFramework()

        assert cf._row_count([1, 2, 3]) == 3
        assert cf._row_count(object()) is None
