import sqlite3
from typing import Any, List

import pyarrow as pa
import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_engine import SqliteFilterEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)


class TestSqliteFilterEngine(FilterEngineTestMixin):
    @pytest.fixture
    def filter_engine(self) -> Any:
        return SqliteFilterEngine

    @pytest.fixture
    def sample_data(self, connection: sqlite3.Connection) -> Any:
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )
        return SqliteRelation.from_arrow(connection, arrow_table)

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        values: list[Any] = result.df()[column].tolist()
        return values

    def test_filter_with_null_values(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "age": [25, 30, 35, 40, 45, None],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
                "category": ["A", "B", "A", "C", "B", "A"],
            }
        )
        data = SqliteRelation.from_arrow(connection, arrow_table)

        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SqliteFilterEngine.do_min_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 4
        ages = sorted(result_df["age"].tolist())
        assert ages == [30, 35, 40, 45]

    def test_filter_with_empty_data(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict(
            {
                "id": pa.array([], type=pa.int64()),
                "age": pa.array([], type=pa.int64()),
                "name": pa.array([], type=pa.string()),
                "category": pa.array([], type=pa.string()),
            }
        )
        empty_data = SqliteRelation.from_arrow(connection, arrow_table)

        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SqliteFilterEngine.do_min_filter(empty_data, single_filter)
        assert len(result) == 0

    def test_filter_with_string_data(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4],
                "status": ["active", "inactive", "pending", "active"],
                "priority": ["high", "low", "medium", "high"],
            }
        )
        data = SqliteRelation.from_arrow(connection, arrow_table)

        feature = Feature("status")
        filter_type = FilterType.equal
        parameter = {"value": "active"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SqliteFilterEngine.do_equal_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        statuses = result_df["status"].tolist()
        assert all(status == "active" for status in statuses)

    def test_complex_regex_patterns(self, connection: sqlite3.Connection) -> None:
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
        data = SqliteRelation.from_arrow(connection, arrow_table)

        feature = Feature("email")
        filter_type = FilterType.regex
        parameter = {"value": r"\.com$"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SqliteFilterEngine.do_regex_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        emails = result_df["email"].tolist()
        assert all(email.endswith(".com") for email in emails)


class TestAdversarialInputs:
    """Verify that adversarial column names and filter values do not cause SQL injection."""

    @pytest.fixture
    def base_data(self, connection: sqlite3.Connection) -> SqliteRelation:
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
                "name": ["alice", "bob", "charlie"],
                "category": ["A", "B", "A"],
            }
        )
        return SqliteRelation.from_arrow(connection, arrow_table)

    def test_filter_value_with_sql_injection_attempt(
        self, base_data: SqliteRelation, connection: sqlite3.Connection
    ) -> None:
        """Filter value containing SQL injection payload must not execute injected SQL."""
        feature = Feature("name")
        single_filter = SingleFilter(feature, FilterType.equal, {"value": "'; DROP TABLE users; --"})
        result = SqliteFilterEngine.do_equal_filter(base_data, single_filter)
        # Should return 0 rows (no row has that exact value), not crash
        assert len(result) == 0

    def test_filter_value_with_or_injection(self, base_data: SqliteRelation, connection: sqlite3.Connection) -> None:
        """Filter value '1 OR 1=1' must be treated as a literal string, not SQL."""
        feature = Feature("name")
        single_filter = SingleFilter(feature, FilterType.equal, {"value": "1 OR 1=1"})
        result = SqliteFilterEngine.do_equal_filter(base_data, single_filter)
        # Should return 0 rows, not all rows
        assert len(result) == 0

    def test_categorical_inclusion_with_single_quote_value(
        self, base_data: SqliteRelation, connection: sqlite3.Connection
    ) -> None:
        """Categorical values containing single quotes must be handled safely."""
        feature = Feature("name")
        single_filter = SingleFilter(
            feature, FilterType.categorical_inclusion, {"values": ["alice", "bob's value", "'; DROP TABLE --"]}
        )
        result = SqliteFilterEngine.do_categorical_inclusion_filter(base_data, single_filter)
        # Only 'alice' matches; the others are literal strings with no match
        assert len(result) == 1
        assert result.df()["name"].tolist() == ["alice"]

    def test_min_filter_with_injection_in_numeric_context(
        self, base_data: SqliteRelation, connection: sqlite3.Connection
    ) -> None:
        """Numeric filter with a legitimate numeric value should work correctly."""
        feature = Feature("value")
        single_filter = SingleFilter(feature, FilterType.min, {"value": 20})
        result = SqliteFilterEngine.do_min_filter(base_data, single_filter)
        assert len(result) == 2
        values = sorted(result.df()["value"].tolist())
        assert values == [20, 30]
