"""End-to-end ASOF (point-in-time) join integration test for DuckDB.

Consumes the shared AsofRunAllTestBase. A live DuckDB connection is threaded
through ``Feature.options`` + a ``DataAccessCollection``. DuckDBFramework only
supports ``ParallelizationMode.SYNC``, so the inherited test method is overridden
to run SYNC only.
"""

from typing import Any, Optional

import pytest

from mloda.user import ParallelizationMode

from tests.test_plugins.compute_framework.test_tooling.asof.asof_run_all_test_base import (
    AsofRunAllTestBase,
    _EXPECTED_ENCODED,
)

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on DuckDB."""

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

    @pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}])
    def test_backward_single_key(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        assert self._run(modes, flight_server) == _EXPECTED_ENCODED
