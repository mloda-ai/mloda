import sqlite3
from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp


@pytest.fixture
def connection() -> Any:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp, deterministic=True)
    yield conn
    conn.close()
