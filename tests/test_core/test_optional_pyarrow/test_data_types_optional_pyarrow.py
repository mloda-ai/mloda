"""Tests that DataType arrow-conversion methods raise ImportError when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY_TO_ARROW: str = """
import sys
from mloda.core.abstract_plugins.components.data_types import DataType

try:
    DataType.to_arrow_type(DataType.INT32)
    print("NO_RAISE")
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__)
"""

_BODY_FROM_ARROW: str = """
import sys
from mloda.core.abstract_plugins.components.data_types import DataType

try:
    DataType.from_arrow_type(object())
    print("NO_RAISE")
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__)
"""

_BODY_INFER_ARROW: str = """
import sys
from mloda.core.abstract_plugins.components.data_types import DataType

try:
    DataType.infer_arrow_type(5)
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
def test_to_arrow_type_raises_import_error_without_pyarrow() -> None:
    """DataType.to_arrow_type must raise ImportError mentioning pyarrow when pyarrow absent."""
    result = run_blocked(_BODY_TO_ARROW)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_from_arrow_type_raises_import_error_without_pyarrow() -> None:
    """DataType.from_arrow_type must raise ImportError mentioning pyarrow when pyarrow absent."""
    result = run_blocked(_BODY_FROM_ARROW)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_infer_arrow_type_raises_import_error_without_pyarrow() -> None:
    """DataType.infer_arrow_type must raise ImportError mentioning pyarrow when pyarrow absent."""
    result = run_blocked(_BODY_INFER_ARROW)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
