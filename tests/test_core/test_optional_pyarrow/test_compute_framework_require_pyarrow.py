"""Test that _require_pyarrow() raises ImportError mentioning pyarrow when pyarrow is absent.

The function _require_pyarrow does not yet exist in
mloda.core.abstract_plugins.compute_framework, so today this test FAILS because
the body prints NO_FUNCTION (import fails with ImportError on the name).
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
import sys

try:
    from mloda.core.abstract_plugins.compute_framework import _require_pyarrow
except ImportError:
    print("NO_FUNCTION")
    sys.exit(0)

try:
    _require_pyarrow()
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
def test_require_pyarrow_raises_import_error_without_pyarrow() -> None:
    """_require_pyarrow() must raise ImportError mentioning pyarrow when pyarrow is absent.

    Current (red): function does not exist yet; body prints NO_FUNCTION.
    """
    result = run_blocked(_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
