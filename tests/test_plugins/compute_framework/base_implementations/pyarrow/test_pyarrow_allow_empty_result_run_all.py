"""End-to-end ``allow_empty_result`` policy test for PyArrow.

Consumes the shared EmptyResultRunAllTestBase. PyArrow is selected by name; no
connection is required.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import mloda
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
    _ENABLED_SCHEMALESS_ALLOWED,
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


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
def test_zero_column_raises_pyarrow(flight_server: Any) -> None:
    """A FG returning ``{}`` (zero columns) raises ``EmptyResultError`` end-to-end on PyArrow.

    With ``allow_empty_result`` retired, a schema-less result is never legitimate: the
    output guard raises for the zero-column table. ``run_all`` re-wraps the framework error,
    so the assertion targets the type NAME in the wrapped message.
    """
    feature = Feature(name="empty_result_schemaless_col")

    with pytest.raises(Exception) as excinfo:
        mloda.run_all(
            [feature],
            compute_frameworks=["PyArrowTable"],
            plugin_collector=_ENABLED_SCHEMALESS_ALLOWED,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )
    assert isinstance(excinfo.value, EmptyResultError)
