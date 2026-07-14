"""Test that ComputeFramework.upload_table raises ImportError mentioning pyarrow when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class _MinimalFramework(ComputeFramework):
    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict


framework = _MinimalFramework()
framework.set_data({"a": [1]})

try:
    framework.upload_table("grpc://127.0.0.1:0")
    print("NO_RAISE")
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__ + ":" + str(e))
"""


@pytest.mark.timeout(30)
def test_upload_table_raises_import_error_without_pyarrow() -> None:
    """upload_table needs pyarrow to build the table it ships: it must say so, not fail with AttributeError."""
    result = run_blocked(_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
