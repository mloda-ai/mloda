"""Unit tests for the DuckDBFilterEngine class."""

from typing import Any
import logging

import pytest

from mloda._testing.compute_framework.filter_suite import NULLABLE_DATA, FilterEngineSuite
from mloda._testing.framework_adapter import DuckDBAdapter, FrameworkAdapter
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


class TestDuckDBFilterEngine(FilterEngineSuite):
    """Unit tests for the DuckDBFilterEngine class using the shared suite."""

    adapter_class = DuckDBAdapter

    @pytest.fixture(params=["arrow", "native"])
    def nullable_data(self, request: Any, adapter: FrameworkAdapter) -> Any:
        """Runs on an Arrow scan (the production path) and a materialized native table (DuckDB's own NaN semantics)."""
        relation = self.frame(adapter, "nullable", NULLABLE_DATA)
        if request.param == "arrow":
            return relation

        connection = adapter.connection
        connection.register("nullable_data_src", relation.to_arrow_table())
        connection.execute("CREATE TABLE nullable_data_tbl AS SELECT * FROM nullable_data_src")
        return DuckdbRelation(connection, connection.table("nullable_data_tbl"))

    # Framework-specific tests below

    def test_filter_with_null_values(self, sample_data: Any) -> None:
        """Test filtering with null values in data."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "age": [25, 30, 35, 40, 45, None],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
                "category": ["A", "B", "A", "C", "B", "A"],
            }
        )
        extended_data = DuckdbRelation.from_arrow(conn, arrow_table)

        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_min_filter(extended_data, single_filter)

        result_df = result.df()
        assert len(result_df) == 4
        ages = result_df["age"].tolist()
        assert None not in ages
        assert sorted(ages) == [30, 35, 40, 45]

    def test_filter_with_empty_data(self, adapter: FrameworkAdapter) -> None:
        """Test filtering with empty DuckDB relation."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [],
                "age": [],
                "name": [],
                "category": [],
            }
        )
        empty_data = DuckdbRelation.from_arrow(conn, arrow_table)

        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_min_filter(empty_data, single_filter)
        result_df = result.df()
        assert len(result_df) == 0

    def test_filter_with_string_data(self, adapter: FrameworkAdapter) -> None:
        """Test filtering with string data types."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4],
                "status": ["active", "inactive", "pending", "active"],
                "priority": ["high", "low", "medium", "high"],
            }
        )
        data = DuckdbRelation.from_arrow(conn, arrow_table)

        feature = Feature("status")
        filter_type = FilterType.EQUAL
        parameter = {"value": "active"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_equal_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        statuses = result_df["status"].tolist()
        assert all(status == "active" for status in statuses)

    def test_filter_with_boolean_data(self, adapter: FrameworkAdapter) -> None:
        """Test filtering with boolean data types."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4],
                "is_active": [True, False, True, False],
                "is_premium": [False, True, True, False],
            }
        )
        data = DuckdbRelation.from_arrow(conn, arrow_table)

        feature = Feature("is_active")
        filter_type = FilterType.EQUAL
        parameter = {"value": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_equal_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        active_flags = result_df["is_active"].tolist()
        assert all(flag is True for flag in active_flags)

    def test_complex_regex_patterns(self, adapter: FrameworkAdapter) -> None:
        """Test complex regex patterns."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "email": [
                    "alice@test.com",
                    "bob@example.org",
                    "charlie@test.com",
                    "david@company.net",
                    "eve@test.org",
                ],
            }
        )
        data = DuckdbRelation.from_arrow(conn, arrow_table)

        feature = Feature("email")
        filter_type = FilterType.REGEX
        parameter = {"value": r"\.com$"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_regex_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        emails = result_df["email"].tolist()
        assert all(email.endswith(".com") for email in emails)
