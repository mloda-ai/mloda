"""Connection-seam tests for DuckDBFramework.open_connection() and ensure_connection()."""

import pickle  # nosec B403
from unittest.mock import Mock, patch

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.user import ConnectionSpec, DataAccessCollection, ParallelizationMode
from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

pytestmark = pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")

_DUCKDB_CONNECT_TARGET = "mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework.duckdb.connect"


def test_open_connection_none_returns_none() -> None:
    assert DuckDBFramework.open_connection(None) is None


def test_open_connection_with_spec_returns_duckdb_connection() -> None:
    conn = DuckDBFramework.open_connection(ConnectionSpec(DuckDBFramework))
    assert isinstance(conn, duckdb.DuckDBPyConnection)


def test_open_connection_with_spec_opens_distinct_connections_on_each_call() -> None:
    spec = ConnectionSpec(DuckDBFramework)
    first = DuckDBFramework.open_connection(spec)
    second = DuckDBFramework.open_connection(spec)
    assert first is not second


def test_open_connection_forwards_spec_params_to_duckdb_connect() -> None:
    spec = ConnectionSpec(DuckDBFramework, database=":memory:")
    mock_connect = Mock(return_value=Mock())
    with patch(_DUCKDB_CONNECT_TARGET, mock_connect):
        DuckDBFramework.open_connection(spec)
    mock_connect.assert_called_once_with(database=":memory:")


def test_ensure_connection_from_spec_source_is_a_working_and_cached_connection() -> None:
    fw = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    fw.connection_source = ConnectionSource(spec=ConnectionSpec(DuckDBFramework))

    first = fw.ensure_connection()
    assert first is not None
    assert first.execute("select 1").fetchone() == (1,)

    second = fw.ensure_connection()
    assert second is first


def test_pickle_round_trip_reopens_a_fresh_connection() -> None:
    fw = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    fw.connection_source = ConnectionSource(spec=ConnectionSpec(DuckDBFramework))
    fw.ensure_connection()

    restored: DuckDBFramework = pickle.loads(pickle.dumps(fw))  # nosec B301

    assert restored.framework_connection_object is None
    conn = restored.ensure_connection()
    assert conn is not None
    assert conn.execute("select 1").fetchone() == (1,)


def test_transform_dict_without_connection_or_source_mentions_spec_and_dac() -> None:
    fw = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    with pytest.raises(ValueError) as excinfo:
        fw.transform({"a": [1]}, [])
    message = str(excinfo.value)
    assert "ConnectionSpec(DuckDBFramework" in message
    assert "DataAccessCollection" in message


def test_pick_connection_from_dac_returns_matching_spec() -> None:
    spec = ConnectionSpec(DuckDBFramework)
    dac = DataAccessCollection(connections={spec})
    assert DuckDBFramework.pick_connection_from_dac(dac) is spec
