"""Unit tests for the DuckDBFilterEngine class."""

from typing import Any, List
import logging

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBFilterEngine(FilterEngineTestMixin):
    """Unit tests for the DuckDBFilterEngine class using shared mixin."""

    # DuckDB doesn't guarantee row order in results
    preserves_order = False

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the DuckDBFilterEngine class."""
        return DuckDBFilterEngine

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
        return connection.from_arrow(arrow_table)

    def get_column_values(self, result: Any, column: str) -> List[Any]:
        """Extract column values from DuckDB relation via pandas DataFrame."""
        return result.df()[column].tolist()  # type: ignore[no-any-return]

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
        extended_data = conn.from_arrow(arrow_table)

        feature = Feature("age")
        filter_type = FilterTypeEnum.min
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
        empty_data = conn.from_arrow(arrow_table)

        feature = Feature("age")
        filter_type = FilterTypeEnum.min
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
        data = conn.from_arrow(arrow_table)

        feature = Feature("status")
        filter_type = FilterTypeEnum.equal
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
        data = conn.from_arrow(arrow_table)

        feature = Feature("is_active")
        filter_type = FilterTypeEnum.equal
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
        data = conn.from_arrow(arrow_table)

        feature = Feature("email")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": r"\.com$"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_regex_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        emails = result_df["email"].tolist()
        assert all(email.endswith(".com") for email in emails)
