"""End-to-end ``allow_empty_result`` policy test for PyArrow.

Consumes the shared EmptyResultRunAllTestBase. PyArrow is selected by name; no
connection is required.
"""

import pytest

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on PyArrow."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PyArrowTable"
