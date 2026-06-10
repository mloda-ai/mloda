"""End-to-end ``allow_empty_result`` policy test for SQLite.

Consumes the shared EmptyResultRunAllTestBase. A live sqlite3 connection is threaded
through ``Feature.options`` + a ``DataAccessCollection`` by the base. The sqlite pyarrow
transformer requires pyarrow, so the class is guarded with a skipif on ``pa``.
"""

import sqlite3
from typing import Any, Optional

import pytest

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on SQLite."""

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
