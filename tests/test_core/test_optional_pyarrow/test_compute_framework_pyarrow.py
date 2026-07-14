"""Test that _pyarrow() raises ImportError mentioning pyarrow when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
import sys

try:
    from mloda.core.abstract_plugins.compute_framework import _pyarrow
except ImportError:
    print("NO_FUNCTION")
    sys.exit(0)

try:
    _pyarrow()
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
def test_pyarrow_raises_import_error_without_pyarrow() -> None:
    """_pyarrow() must raise ImportError mentioning pyarrow when pyarrow is absent."""
    result = run_blocked(_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
