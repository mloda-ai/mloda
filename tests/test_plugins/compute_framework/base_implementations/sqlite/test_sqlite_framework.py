import sqlite3
from typing import Any, Optional, Type

import pyarrow as pa
import pytest

from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework, _regexp
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase


class TestSqliteFrameworkBasics:
    def test_is_available(self) -> None:
        assert SqliteFramework.is_available() is True

    def test_expected_data_framework(self) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert fw.expected_data_framework() == SqliteRelation

    def test_set_framework_connection_object(self, connection: sqlite3.Connection) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)
        assert fw.framework_connection_object is connection

    def test_set_framework_connection_object_invalid(self) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError, match="Expected a sqlite3.Connection"):
            fw.set_framework_connection_object("not_a_connection")

    def test_transform_dict(self, connection: sqlite3.Connection) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        result = fw.transform(dict_data, set())
        assert isinstance(result, SqliteRelation)
        assert len(result) == 3
        assert set(result.columns) == {"column1", "column2"}

    def test_transform_invalid_data(self) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError):
            fw.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, connection: sqlite3.Connection) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)

        arrow = pa.Table.from_pydict({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        data = SqliteRelation.from_arrow(connection, arrow)

        result = fw.select_data_by_column_names(data, {FeatureName("column1")})
        assert "column1" in result.columns

    def test_set_framework_connection_object_none_raises(self) -> None:
        """Passing None when no connection is set should raise ValueError."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError):
            fw.set_framework_connection_object(None)

    def test_set_framework_connection_object_different_conn_raises(self, connection: sqlite3.Connection) -> None:
        """Passing a different connection when one is already set should raise ValueError."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)
        other_conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError):
            fw.set_framework_connection_object(other_conn)
        other_conn.close()

    def test_set_framework_connection_object_same_conn_is_safe(self, connection: sqlite3.Connection) -> None:
        """Passing the same connection object again should not raise."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)
        fw.set_framework_connection_object(connection)  # should not raise
        assert fw.framework_connection_object is connection

    def test_set_column_names(self, connection: sqlite3.Connection) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        arrow = pa.Table.from_pydict({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        fw.data = SqliteRelation.from_arrow(connection, arrow)
        fw.set_column_names()
        assert "column1" in fw.column_names
        assert "column2" in fw.column_names

    def test_transform_unsupported_type_raises(self) -> None:
        """transform() with a plain int raises ValueError."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError):
            fw.transform(data=42, feature_names=set())

    def test_transform_dict_no_connection_raises(self) -> None:
        """transform() with dict but no connection set raises ValueError."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError, match="not set"):
            fw.transform(data={"col": [1, 2]}, feature_names=set())

    def test_transform_add_column_preserves_existing(self, connection: sqlite3.Connection) -> None:
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(connection)
        dict_data = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
        fw.data = fw.transform(dict_data, set())
        result = fw.transform(data=[7, 8, 9], feature_names={"col_c"})
        assert set(result.columns) == {"col_a", "col_b", "col_c"}
        assert len(result) == 3
        arrow = result.to_arrow_table()
        assert arrow.column("col_a").to_pylist() == [1, 2, 3]
        assert arrow.column("col_c").to_pylist() == [7, 8, 9]

    def test_set_framework_connection_wrong_type_raises(self) -> None:
        """set_framework_connection_object() with wrong type raises ValueError."""
        fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(ValueError):
            fw.set_framework_connection_object(12345)

    def test_from_dict_mismatched_column_lengths(self, connection: sqlite3.Connection) -> None:
        """from_dict() should raise ValueError when columns have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            SqliteRelation.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5]})


class TestSqliteFrameworkMerge(DataFrameTestBase):
    @classmethod
    def framework_class(cls) -> Type[Any]:
        return SqliteFramework

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.create_function("REGEXP", 2, _regexp)
        super().setup_method()

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        arrow_table = pa.Table.from_pydict(data)
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def get_connection(self) -> Optional[Any]:
        return self.conn

    def _create_test_framework(self) -> Any:
        framework = super()._create_test_framework()
        framework.set_framework_connection_object(self.conn)
        return framework

    def _get_merge_engine(self, framework: Any) -> Any:
        merge_engine_class = framework.merge_engine()
        framework_connection = framework.get_framework_connection_object()

        class MergeEngineFactory:
            def __call__(self) -> Any:
                return merge_engine_class(framework_connection)

        return MergeEngineFactory()
