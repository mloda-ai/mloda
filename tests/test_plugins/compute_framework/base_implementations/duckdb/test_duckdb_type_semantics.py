"""Column-semantics introspection for duckdb relations (epic #518, Fix 3).

A duckdb ``INTERVAL`` type must be treated as temporal but NOT numeric. The
reader currently reports ``is_numeric=True`` because the substring ``"int"``
appears inside ``"interval"``. This test pins the correct expectation and
fails until the numeric-token matching is tightened.
"""

from typing import Any
import logging

import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_type_semantics import column_semantics

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckdbIntervalSemantics:
    def test_interval_is_temporal_not_numeric(self, connection: Any) -> None:
        relation = connection.sql("SELECT INTERVAL 1 HOUR AS c")
        sem = column_semantics(relation, "c")
        assert sem.is_temporal is True
        assert sem.is_numeric is False
