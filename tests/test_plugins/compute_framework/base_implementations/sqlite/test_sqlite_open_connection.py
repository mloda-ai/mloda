"""Connection-seam tests for SqliteFramework.open_connection() and ensure_connection()."""

import pickle  # nosec B403
import sqlite3
from unittest.mock import Mock, patch

from mloda.user import ConnectionSpec, DataAccessCollection, ParallelizationMode
from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
import pytest

_SQLITE_CONNECT_TARGET = "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework.sqlite3.connect"


def test_open_connection_none_returns_none() -> None:
    assert SqliteFramework.open_connection(None) is None


def test_open_connection_with_spec_returns_sqlite_connection() -> None:
    conn = SqliteFramework.open_connection(ConnectionSpec(SqliteFramework, database=":memory:"))
    assert isinstance(conn, sqlite3.Connection)


def test_open_connection_with_spec_opens_distinct_connections_on_each_call() -> None:
    spec = ConnectionSpec(SqliteFramework, database=":memory:")
    first = SqliteFramework.open_connection(spec)
    second = SqliteFramework.open_connection(spec)
    assert first is not second


def test_open_connection_forwards_spec_params_to_sqlite3_connect() -> None:
    spec = ConnectionSpec(SqliteFramework, database=":memory:")
    mock_connect = Mock(return_value=Mock())
    with patch(_SQLITE_CONNECT_TARGET, mock_connect):
        SqliteFramework.open_connection(spec)
    mock_connect.assert_called_once_with(database=":memory:")


def test_ensure_connection_from_spec_source_is_a_working_and_cached_connection() -> None:
    fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    fw.connection_source = ConnectionSource(spec=ConnectionSpec(SqliteFramework, database=":memory:"))

    first = fw.ensure_connection()
    assert first is not None
    assert first.execute("select 1").fetchone() == (1,)

    second = fw.ensure_connection()
    assert second is first


def test_pickle_round_trip_reopens_a_fresh_connection() -> None:
    fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    fw.connection_source = ConnectionSource(spec=ConnectionSpec(SqliteFramework, database=":memory:"))
    fw.ensure_connection()

    restored: SqliteFramework = pickle.loads(pickle.dumps(fw))  # nosec B301

    assert restored.framework_connection_object is None
    conn = restored.ensure_connection()
    assert conn is not None
    assert conn.execute("select 1").fetchone() == (1,)


def test_transform_dict_without_connection_or_source_mentions_spec() -> None:
    fw = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    with pytest.raises(ValueError) as excinfo:
        fw.transform({"a": [1]}, [])
    assert "ConnectionSpec(SqliteFramework" in str(excinfo.value)


def test_pick_connection_from_dac_returns_matching_spec() -> None:
    spec = ConnectionSpec(SqliteFramework)
    dac = DataAccessCollection(connections={spec})
    assert SqliteFramework.pick_connection_from_dac(dac) is spec
