"""End-to-end ASOF (point-in-time) join integration test for SQLite.

Consumes the shared AsofRunAllTestBase. A live sqlite3 connection is threaded
through ``Feature.options`` + a ``DataAccessCollection``. SqliteFramework only
supports ``ParallelizationMode.SYNC``, so the inherited test method is overridden
to run SYNC only. The sqlite pyarrow transformer requires pyarrow, so the class is
guarded with a skipif on ``pa``.
"""

import sqlite3
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
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on SQLite."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "SqliteFramework"

    def get_connection(self) -> Optional[Any]:
        """SQLite requires a connection object."""
        if not hasattr(self, "_connection"):
            self._connection = sqlite3.connect(":memory:")
        return self._connection

    def teardown_method(self) -> None:
        """Close the sqlite3 connection opened by get_connection, if any."""
        if hasattr(self, "_connection"):
            self._connection.close()

    @pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}])
    def test_backward_single_key(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        assert self._run(modes, flight_server) == _EXPECTED_ENCODED
