"""
Tests for SqliteMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. SQLite native ASOF cannot express
'nearest', so a negative test pins that 'nearest' raises a ValueError. Conversion
into a SqliteRelation routes through PyArrow, so PyArrow must be available.
"""

import sqlite3
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
    pa = None  # type: ignore[assignment]


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

    def test_nearest_raises_value_error(self) -> None:
        """Vector F: SQLite native ASOF cannot express 'nearest' in v1 -> ValueError."""
        left = self.convert_dict_to_framework([{"k": 1, "t": 10, "lv": 100}])
        right = self.convert_dict_to_framework([{"k": 1, "t": 8, "rv": 7}])

        engine = SqliteMergeEngine(self.get_connection())
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError, match="nearest"):
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
