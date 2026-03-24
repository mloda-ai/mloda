"""Tests for the DuckdbRelation class."""

from typing import Any, List
import logging

import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from tests.test_plugins.compute_framework.base_implementations.relation_test_mixin import (
    RelationTestMixin,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed.")
class TestDuckdbRelation(RelationTestMixin):
    @pytest.fixture
    def sample_relation(self, connection: Any) -> "DuckdbRelation":
        return DuckdbRelation.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            },
        )

    @pytest.fixture
    def relation_class(self) -> Any:
        return DuckdbRelation

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        return result.df()[column].tolist()  # type: ignore[no-any-return]

    # DuckDB-specific tests

    def test_stores_connection_and_relation(self, connection: Any) -> None:
        native_rel = connection.from_arrow(pa.table({"x": [1, 2]}))
        rel = DuckdbRelation(connection, native_rel)
        assert rel.connection is connection
        assert rel.columns == ["x"]

    def test_drop_is_noop(self, sample_relation: "DuckdbRelation") -> None:
        sample_relation.drop()
        assert len(sample_relation) == 5

    def test_right_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="right")
        assert len(result) == 3

    def test_outer_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="outer")
        assert len(result) == 4

    def test_join_with_sql_injection_alias(self, connection: Any) -> None:
        """A crafted alias must not be interpreted as SQL."""
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "val": ["a", "b"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "score": [10, 20]})

        malicious_alias = 'a" JOIN duckdb_tables() --'
        left_aliased = left.set_alias(malicious_alias)
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, f"{malicious_alias}.idx = right_rel.idx", how="inner")
        assert len(result) == 2
        assert "val" in result.columns
        assert "score" in result.columns
