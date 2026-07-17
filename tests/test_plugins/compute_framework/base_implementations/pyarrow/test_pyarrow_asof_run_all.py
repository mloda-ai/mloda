"""End-to-end ASOF (point-in-time) join integration test for PyArrow.

Consumes the shared AsofRunAllTestBase. PyArrowTable supports SYNC + THREADING,
so the subclass is hooks-only and inherits the base SYNC+THREADING parametrization.
"""

import pytest

from tests.test_plugins.compute_framework.test_tooling.asof.asof_run_all_test_base import AsofRunAllTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on PyArrow."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PyArrowTable"
