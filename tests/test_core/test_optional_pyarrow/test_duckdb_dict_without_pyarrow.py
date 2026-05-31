"""Test that DuckdbRelation.from_dict raises ImportError when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
import sys

try:
    import duckdb
except ImportError:
    print("DUCKDB_MISSING")
    sys.exit(0)

try:
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__ + ":" + str(e))
    sys.exit(1)

conn = duckdb.connect()

try:
    DuckdbRelation.from_dict(conn, {"a": [1, 2, 3]})
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
def test_duckdb_from_dict_raises_import_error_without_pyarrow() -> None:
    """DuckdbRelation.from_dict must raise ImportError mentioning pyarrow when pyarrow absent."""
    result = run_blocked(_BODY)

    if "DUCKDB_MISSING" in result.stdout:
        pytest.skip("duckdb not installed in this environment")

    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
