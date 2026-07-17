"""
Tests for SqliteMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. SQLite native ASOF cannot express
'nearest', so a negative test pins that 'nearest' raises a ValueError. Conversion
into a SqliteRelation routes through PyArrow, so PyArrow must be available.
"""

import sqlite3
from datetime import timedelta
from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for SqliteMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return SqliteMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        return SqliteRelation

    def get_connection(self) -> Optional[Any]:
        if not hasattr(self, "_connection"):
            self._connection = sqlite3.connect(":memory:")
            self._connection.create_function("REGEXP", 2, _regexp)
        return self._connection

    def test_coerce_one_sided_string_vs_numeric_raises(self) -> None:
        """sqlite coercion turns ISO TEXT into julianday() day numbers. If only ONE side is a
        string column and the other side is already numeric (unknown unit, e.g. epoch seconds),
        the join would silently compare julian day numbers against epoch values. With
        coerce_time_columns=True, when exactly one side's time column needs coercion and the
        other is already ordered, the engine must raise ValueError naming the coerced column
        and advising to cast manually."""
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )
        engine = SqliteMergeEngine(self.get_connection())

        left_string = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-03T00:00:00", "lv": 1}])
        right_numeric = self.convert_dict_to_framework(
            [{"k": "a", "t": 1.0, "rv": 1.0}, {"k": "a", "t": 2.0, "rv": 2.0}]
        )
        with pytest.raises(ValueError, match=r"'t'.*cast"):
            engine.merge_asof(left_string, right_numeric, Index(("k",)), Index(("k",)), cfg)

        # Symmetric case: left numeric, right string must raise as well.
        left_numeric = self.convert_dict_to_framework([{"k": "a", "t": 1.0, "lv": 1}, {"k": "a", "t": 2.0, "lv": 2}])
        right_string = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-03T00:00:00", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*cast"):
            engine.merge_asof(left_numeric, right_string, Index(("k",)), Index(("k",)), cfg)

    def test_nearest_raises_value_error(self) -> None:
        """Vector F: SQLite native ASOF cannot express 'nearest' in v1 -> ValueError."""
        left = self.convert_dict_to_framework([{"k": 1, "t": 10, "lv": 100}])
        right = self.convert_dict_to_framework([{"k": 1, "t": 8, "rv": 7}])

        engine = SqliteMergeEngine(self.get_connection())
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError, match="nearest"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_timedelta_tolerance_raises_value_error(self) -> None:
        """A timedelta tolerance is unsupported on the SQLite SQL backend (which needs a numeric
        gap). It must raise a clear ValueError mentioning 'timedelta', not the confusing
        TypeError from float(timedelta)."""
        left = self.convert_dict_to_framework([{"k": 1, "t": 10, "lv": 100}])
        right = self.convert_dict_to_framework([{"k": 1, "t": 8, "rv": 7}])

        engine = SqliteMergeEngine(self.get_connection())
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="timedelta"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tie_yields_single_row(self) -> None:
        """Tie contract: when two right rows share the identical boundary time within the same
        by-key, the ASOF join must still yield exactly ONE row per left row (parity with
        duckdb/pandas/python_dict/pyarrow). Ties are broken deterministically by ordering the
        candidate right rows on their non-key columns ascending, so the smaller rv (100) wins.
        """
        left = self.convert_dict_to_framework([{"k": 1, "t": 10, "lv": 1}])
        right = self.convert_dict_to_framework([{"k": 1, "t": 5, "rv": 100}, {"k": 1, "t": 5, "rv": 200}])

        engine = SqliteMergeEngine(self.get_connection())
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_dicts = self.convert_framework_to_dict(result)
        assert len(result_dicts) == 1, f"expected 1 row, got {len(result_dicts)}: {result_dicts}"
        assert self._normalize_value(result_dicts[0]["rv"]) == 100
