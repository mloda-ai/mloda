"""
Shared fixtures for DuckDB compute framework tests.

This module provides common fixtures used across multiple DuckDB-specific tests
to reduce duplication and ensure consistent connection handling.
"""

from typing import Any

import pytest

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB is not installed. Some fixtures will not be available.")
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False


@pytest.fixture
def connection() -> Any:
    """Create a DuckDB connection for testing.

    Returns a fresh DuckDB in-memory connection for each test.
    This fixture is shared across DuckDB filter engine, merge engine,
    and framework tests.
    """
    if not DUCKDB_AVAILABLE:
        pytest.skip("DuckDB is not installed")
    conn = duckdb.connect()
    yield conn
    conn.close()


# Non-UTC session timezones used to surface DATE_TRUNC / time_bucket drift (issues #522/#523).
# - America/New_York: whole-hour offset (UTC-5), catches day-granularity DATE_TRUNC drift.
# - Asia/Kolkata: fractional offset (UTC+5:30), catches sub-hour drift whole-hour zones miss.
NON_UTC_ZONES = ["America/New_York", "Asia/Kolkata"]


@pytest.fixture(params=NON_UTC_ZONES)
def non_utc_zone(request: Any) -> str:
    """Expose the parametrized non-UTC session timezone string."""
    return str(request.param)


@pytest.fixture
def non_utc_connection(non_utc_zone: str) -> Any:
    """Create a DuckDB connection whose session timezone is preset to a non-UTC zone.

    The zone is set via ``SET TimeZone`` BEFORE yielding so that any framework
    consuming this connection inherits a non-UTC session timezone, mirroring a
    user-supplied connection that was configured for a local timezone.
    """
    if not DUCKDB_AVAILABLE:
        pytest.skip("DuckDB is not installed")
    conn = duckdb.connect()
    conn.execute(f"SET TimeZone='{non_utc_zone}'")
    yield conn
    conn.close()
