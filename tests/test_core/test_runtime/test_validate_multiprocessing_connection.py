"""Unit tests for raise_on_multiprocessing_connection_conflict."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.runtime.validate_multiprocessing_link import raise_on_multiprocessing_connection_conflict
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework


class _ConnectionConflictUnitCFW(ComputeFramework):
    """Supports MULTIPROCESSING by default; reports unavailable to avoid leaking into other tests."""

    @staticmethod
    def is_available() -> bool:
        return False


def test_a_framework_supporting_multiprocessing_with_a_resolved_connection_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        raise_on_multiprocessing_connection_conflict({_ConnectionConflictUnitCFW: object()})

    message = str(excinfo.value)
    assert _ConnectionConflictUnitCFW.__name__ in message, f"the offending framework must be named; got: {message}"
    assert "DataAccessCollection" in message, f"the connection source must be named; got: {message}"
    assert "supported_parallelization_modes" in message, (
        f"resolution (a), excluding MULTIPROCESSING from supported_parallelization_modes, must be offered; "
        f"got: {message}"
    )
    assert "or the caller runs without" in message, (
        f"resolution (b), running without MULTIPROCESSING, must be offered; got: {message}"
    )
    assert "omits that connection" in message, (
        f"resolution (c), omitting the connection from the DataAccessCollection, must be offered; got: {message}"
    )


@pytest.mark.parametrize("cfw_class", [DuckDBFramework, SqliteFramework, SparkFramework, IcebergFramework])
def test_a_framework_excluding_multiprocessing_with_a_resolved_connection_does_not_raise(
    cfw_class: type[ComputeFramework],
) -> None:
    raise_on_multiprocessing_connection_conflict({cfw_class: object()})


def test_an_empty_connection_map_does_not_raise() -> None:
    raise_on_multiprocessing_connection_conflict({})
