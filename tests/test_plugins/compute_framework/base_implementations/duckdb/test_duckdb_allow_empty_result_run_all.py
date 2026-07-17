"""End-to-end ``allow_empty_result`` policy test for DuckDB.

Consumes the shared EmptyResultRunAllTestBase. A live DuckDB connection is threaded
through ``Feature.options`` + a ``DataAccessCollection`` by the base.
"""

from typing import Any, Optional

import pytest

from mloda.user import ParallelizationMode

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on DuckDB."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "DuckDBFramework"

    def get_connection(self) -> Optional[Any]:
        """DuckDB requires a connection object."""
        if not hasattr(self, "_connection"):
            self._connection = duckdb.connect()
        return self._connection

    def teardown_method(self) -> None:
        """Close the DuckDB connection opened by get_connection, if any."""
        if hasattr(self, "_connection"):
            self._connection.close()

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC])
    def test_empty_result_default(self, mode: ParallelizationMode, flight_server: Any) -> None:
        """DuckDBFramework only supports SYNC; run the inherited default case SYNC only."""
        self._run_default_case(mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC])
    def test_empty_result_allowed_succeeds(self, mode: ParallelizationMode, flight_server: Any) -> None:
        """DuckDBFramework only supports SYNC; run the inherited allowed case SYNC only."""
        self._run_allowed_case(mode, flight_server)
