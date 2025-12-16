"""
Shared fixtures for compute framework base implementation tests.

This module provides common fixtures used across multiple compute framework tests
to reduce duplication and ensure consistency.
"""

from typing import Any

import pytest

from mloda.user import Index


@pytest.fixture
def index_obj() -> Any:
    """Create index object for joins.

    Returns a standard Index with ("idx",) tuple for use in merge engine tests.
    This fixture is shared across DuckDB, Polars, and PythonDict merge engine tests.
    """
    return Index(("idx",))


@pytest.fixture
def dict_data() -> dict[str, list[int]]:
    """Create fresh test dictionary data for each test.

    Returns a simple dictionary with two columns for use in dataframe tests.
    This fixture is shared across Pandas, Polars, PyArrow, and DuckDB dataframe tests.
    """
    return {"column1": [1, 2, 3], "column2": [4, 5, 6]}
