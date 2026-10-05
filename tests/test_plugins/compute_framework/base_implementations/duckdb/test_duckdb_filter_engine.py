"""Unit tests for the DuckDBFilterEngine class."""

from decimal import Decimal
from typing import Any
import logging

import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.time_range_filter_engine_test_mixin import (
    SAMPLE_IDS,
    SAMPLE_TIMESTAMPS,
    TimeRangeFilterEngineTestMixin,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBFilterEngine(FilterEngineTestMixin, TimeRangeFilterEngineTestMixin):
    """Unit tests for the DuckDBFilterEngine class using shared mixins."""

    filter_engine_class = DuckDBFilterEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        """Create a sample DuckDB relation for testing."""
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )
        return DuckdbRelation.from_arrow(connection, arrow_table)

    @pytest.fixture(params=["arrow", "native"])
    def nullable_category_sample_data(self, request: Any, connection: Any) -> Any:
        """Runs on an Arrow scan (the production path) and a materialized native table (DuckDB's own NaN semantics)."""
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", None, "B", None, "C"],
                "score": [1, None, 2, None, 3],
                "ratio": pa.array([1.0, float("nan"), 2.0, None, 3.0], type=pa.float64()),
            }
        )
        if request.param == "arrow":
            return DuckdbRelation.from_arrow(connection, arrow_table)

        connection.register("nullable_category_sample_data_src", arrow_table)
        connection.execute(
            "CREATE TABLE nullable_category_sample_data_tbl AS SELECT * FROM nullable_category_sample_data_src"
        )
        return DuckdbRelation(connection, connection.table("nullable_category_sample_data_tbl"))

    @pytest.fixture
    def decimal_sample_data(self, connection: Any) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None]
        return DuckdbRelation.from_arrow(connection, pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))}))

    def get_decimal_column_dtype(self, data: Any) -> Any:
        return data.types[data.columns.index("d")]

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from DuckDB relation via Arrow."""
        return result.to_arrow_table()[column].to_pylist()  # type: ignore[no-any-return]

    @pytest.fixture
    def sample_time_data(self, connection: Any) -> Any:
        # pa.array with a tz-typed timestamp interprets naive datetimes as already in that tz;
        # tz-aware datetimes are rejected on some PyArrow versions.
        naive_ts = [ts.replace(tzinfo=None) for ts in SAMPLE_TIMESTAMPS]
        arrow_table = pa.table(
            {"id": pa.array(SAMPLE_IDS), "ts": pa.array(naive_ts, type=pa.timestamp("us", tz="UTC"))}
        )
        return DuckdbRelation.from_arrow(connection, arrow_table)

    def get_id_column_values(self, result: Any) -> list[int]:
        return list(result.df()["id"].tolist())

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

    def test_filter_with_empty_data(self) -> None:
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

    def test_filter_with_string_data(self) -> None:
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

    def test_filter_with_boolean_data(self) -> None:
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

    def test_complex_regex_patterns(self) -> None:
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
