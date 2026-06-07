"""End-to-end ``allow_empty_result`` policy test for Iceberg.

Consumes the shared EmptyResultRunAllTestBase. ``IcebergFramework.transform`` emits a
PyArrow table for dict input (see ``test_iceberg_framework.test_transform_dict_to_arrow``),
so no catalog connection is needed for this dict-producing FG: at validation ``self.data``
is a zero-row PyArrow table. This pins the PyArrow working-data branch of Iceberg's
``_is_empty``; the native IcebergTable (scan) branch is covered by the unit mixin test.
"""

import pytest

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on Iceberg."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "IcebergFramework"
