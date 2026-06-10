"""End-to-end ``allow_empty_result`` policy test for PyArrow.

Consumes the shared EmptyResultRunAllTestBase. PyArrow is selected by name; no
connection is required.
"""

from typing import Any

import pytest

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
def test_zero_column_allowed_succeeds_pyarrow(flight_server: Any) -> None:
    """An opted-in FG returning ``{}`` (zero columns) must succeed end-to-end on PyArrow.

    The output guard accepts the schema-less result because ``allow_empty_result()`` is
    True; result selection must then pass the zero-column table through unchanged instead
    of trying (and failing) to match feature names against an empty column set. The caller
    receives exactly one result: a ``pa.Table`` with zero columns.
    """
    feature = Feature(name="empty_result_schemaless_col")

    result = mloda.run_all(
        [feature],
        compute_frameworks=["PyArrowTable"],
        plugin_collector=_ENABLED_SCHEMALESS_ALLOWED,
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
    )

    assert len(result) == 1
    assert result[0].num_columns == 0
