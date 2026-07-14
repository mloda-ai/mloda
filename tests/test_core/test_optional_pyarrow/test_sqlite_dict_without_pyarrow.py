"""Test that SqliteRelation.from_dict raises ImportError when pyarrow is absent.

sqlite3 is stdlib, so the relation is reachable without any extra installed, but it types its columns
through Arrow. Without the guard the failure is an AttributeError on None, which names nothing.
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
import sqlite3
import sys

try:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__ + ":" + str(e))
    sys.exit(1)

conn = sqlite3.connect(":memory:")

try:
    SqliteRelation.from_dict(conn, {"a": [1, 2, 3]})
    print("NO_RAISE")
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__)
"""


@pytest.mark.timeout(30)
def test_sqlite_from_dict_raises_import_error_without_pyarrow() -> None:
    """SqliteRelation.from_dict must raise ImportError mentioning pyarrow when pyarrow is absent."""
    result = run_blocked(_BODY)

    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
