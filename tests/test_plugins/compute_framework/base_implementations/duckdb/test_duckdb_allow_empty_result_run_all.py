"""End-to-end empty-result policy test for DuckDB.

Consumes the shipped EmptyResultRunAllSuite. The adapter owns the live DuckDB connection.
"""

import logging
from collections.abc import Mapping
from typing import Any

from mloda._testing.compute_framework.run_all_suite import (
    UNDECLARED_COLUMN,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    PolicyRaises,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    select_read,
)
from mloda._testing.framework_adapter import DuckDBAdapter
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import MissingRelationColumnError

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]


class TestDuckDBAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on DuckDB."""

    adapter_class = DuckDBAdapter

    def zero_column_expectation(self) -> PolicyRaises:
        return PolicyRaises(match_substring="must have at least one column", error_type=duckdb.InvalidInputException)


class TestDuckDBSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all (SYNC only)."""

    adapter_class = DuckDBAdapter


def _project_read(data: Any) -> Any:
    return data.project(f"{UNDECLARED_COLUMN} * 2 AS mc_doubled")


def _filter_read(data: Any) -> Any:
    return data.filter(f"{UNDECLARED_COLUMN} > 0")


def _binder_exception() -> type[BaseException]:
    import duckdb

    return duckdb.BinderException


class TestDuckDBMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on DuckDB raise an explanatory error (SYNC only)."""

    adapter_class = DuckDBAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "SelectReadConsumer": (select_read, lambda: MissingRelationColumnError),
            "DuckDBProjectReadConsumer": (_project_read, _binder_exception),
            "DuckDBFilterReadConsumer": (_filter_read, _binder_exception),
        }
