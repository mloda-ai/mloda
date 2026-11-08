"""End-to-end empty-result policy test for SQLite.

Consumes the shipped EmptyResultRunAllSuite. The sqlite pyarrow transformer requires pyarrow; the adapter
skips the host when it is unavailable.
"""

import sqlite3
import sys
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
from mloda._testing.framework_adapter import SqliteAdapter
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import MissingRelationColumnError


class TestSqliteAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on SQLite."""

    adapter_class = SqliteAdapter

    def zero_column_expectation(self) -> PolicyRaises:
        return PolicyRaises(match_substring="Cannot create relation from empty dictionary", error_type=ValueError)


class TestSqliteSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all (SYNC only)."""

    adapter_class = SqliteAdapter


def _filter_read(data: Any) -> Any:
    return data.filter(f"{UNDECLARED_COLUMN} > 0")


def _quoted_filter_read(data: Any) -> Any:
    return data.filter(f'"{UNDECLARED_COLUMN}" > 0')


class TestSqliteMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on SQLite raise an explanatory error (SYNC only)."""

    adapter_class = SqliteAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        reads: dict[str, tuple[UndeclaredRead, CauseType]] = {
            "SelectReadConsumer": (select_read, lambda: MissingRelationColumnError),
            "SqliteFilterReadConsumer": (_filter_read, lambda: sqlite3.OperationalError),
        }
        if sys.version_info >= (3, 12):
            reads["SqliteQuotedFilterReadConsumer"] = (_quoted_filter_read, lambda: sqlite3.OperationalError)
        return reads
