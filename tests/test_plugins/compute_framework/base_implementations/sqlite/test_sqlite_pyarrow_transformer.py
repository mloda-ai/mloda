import re
import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_pyarrow_transformer import (
    SqlBasePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_pyarrow_transformer import (
    SqlitePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


def _regexp(pattern: str, string: str) -> bool:
    return bool(re.search(pattern, string))


@pytest.fixture
def connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp)
    return conn


class TestSqlitePyArrowTransformer:
    def test_framework_types(self) -> None:
        assert SqlitePyArrowTransformer.framework() == SqliteRelation
        assert SqlitePyArrowTransformer.other_framework() == pa.Table

    def test_check_imports(self) -> None:
        assert SqlitePyArrowTransformer.check_imports() is True

    def test_sqlite_to_pyarrow(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict(
            {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]}
        )
        relation = SqliteRelation.from_arrow(connection, arrow_table)

        result = SqlitePyArrowTransformer.transform_fw_to_other_fw(relation)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
        assert set(result.column_names) == {"int_col", "str_col", "float_col"}

    def test_pyarrow_to_sqlite(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict(
            {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]}
        )

        result = SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        assert isinstance(result, SqliteRelation)
        assert len(result) == 3
        assert set(result.columns) == {"int_col", "str_col", "float_col"}

    def test_roundtrip(self, connection: sqlite3.Connection) -> None:
        original_arrow = pa.Table.from_pydict(
            {
                "int_col": [1, 2, 3, 4],
                "str_col": ["hello", "world", "test", "data"],
                "float_col": [1.1, 2.2, 3.3, 4.4],
            }
        )
        relation = SqliteRelation.from_arrow(connection, original_arrow)

        arrow_table = SqlitePyArrowTransformer.transform_fw_to_other_fw(relation)
        restored = SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        original_df = relation.df()
        restored_df = restored.df()

        assert set(original_df.columns) == set(restored_df.columns)
        assert len(original_df) == len(restored_df)

    def test_empty_relation(self, connection: sqlite3.Connection) -> None:
        empty_arrow = pa.Table.from_pydict(
            {"int_col": pa.array([], type=pa.int64()), "str_col": pa.array([], type=pa.string())}
        )
        relation = SqliteRelation.from_arrow(connection, empty_arrow)

        arrow_table = SqlitePyArrowTransformer.transform_fw_to_other_fw(relation)
        restored = SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        assert len(restored) == 0
        assert set(restored.columns) == {"int_col", "str_col"}

    def test_null_values(self, connection: sqlite3.Connection) -> None:
        arrow_with_nulls = pa.Table.from_pydict({"int_col": [1, None, 3], "str_col": ["a", None, "c"]})
        relation = SqliteRelation.from_arrow(connection, arrow_with_nulls)

        arrow_table = SqlitePyArrowTransformer.transform_fw_to_other_fw(relation)
        restored = SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        original_df = relation.df()
        restored_df = restored.df()

        for col in original_df.columns:
            assert original_df[col].isna().tolist() == restored_df[col].isna().tolist()

    def test_transform_with_correct_orientation(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
        relation = SqliteRelation.from_arrow(connection, arrow_table)

        arrow_result = SqlitePyArrowTransformer.transform(SqliteRelation, pa.Table, relation, None)
        assert isinstance(arrow_result, pa.Table)

        sqlite_result = SqlitePyArrowTransformer.transform(pa.Table, SqliteRelation, arrow_result, connection)
        assert isinstance(sqlite_result, SqliteRelation)

    def test_transform_with_unsupported_frameworks(self, connection: sqlite3.Connection) -> None:
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})
        relation = SqliteRelation.from_arrow(connection, arrow_table)

        with pytest.raises(ValueError):
            SqlitePyArrowTransformer.transform(list, dict, relation, None)

    def test_connection_required(self) -> None:
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})

        with pytest.raises(ValueError, match="connection object is required"):
            SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, None)

    def test_invalid_connection_type(self) -> None:
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})

        with pytest.raises(ValueError, match="Expected a sqlite3.Connection"):
            SqlitePyArrowTransformer.transform_other_fw_to_fw(arrow_table, "not_a_connection")


class TestSqlBasePyArrowTransformerImportError:
    def test_other_framework_returns_not_implemented_when_pyarrow_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """other_framework() must return NotImplementedError when pyarrow is None so check_imports() works correctly."""
        monkeypatch.setattr(
            "mloda_plugins.compute_framework.base_implementations.sql.sql_base_pyarrow_transformer.pa",
            None,
        )
        assert SqlBasePyArrowTransformer.other_framework() == NotImplementedError
