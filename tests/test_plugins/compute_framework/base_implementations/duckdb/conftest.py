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
    return duckdb.connect()
