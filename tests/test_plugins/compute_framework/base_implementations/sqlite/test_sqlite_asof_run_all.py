"""End-to-end ASOF (point-in-time) join integration test for SQLite, on the shipped AsofRunAllSuite."""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from mloda._testing.framework_adapter import SqliteAdapter


class TestSqliteAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on SQLite."""

    adapter_class = SqliteAdapter
