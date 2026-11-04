"""End-to-end ``allow_empty_result`` policy test for Iceberg.

Consumes the shared EmptyResultRunAllTestBase. ``IcebergFramework.transform`` emits a
PyArrow table for dict input (see ``test_iceberg_framework.test_transform_dict_to_arrow``),
so no catalog connection is needed for this dict-producing FG: at validation ``self.data``
is a zero-row PyArrow table that still carries its column (state C -> success). The native
IcebergTable (schema) branch is covered by the unit mixin test.
"""

from typing import Any

import pytest

from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from tests.test_plugins.compute_framework.base_implementations.iceberg.iceberg_test_table import real_iceberg_table
from tests.test_plugins.compute_framework.test_tooling.missing_column_run_all_test_base import (
    MissingColumnConsumer,
    MissingColumnRoot,
    MissingColumnRunAllTestBase,
    UNDECLARED_COLUMN,
    PyArrowFilterReadConsumer,
    PyArrowKeyReadConsumer,
)
from tests.test_plugins.compute_framework.test_tooling.sibling_dict_run_all_test_base import (
    SiblingDictRunAllTestBase,
)
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
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on Iceberg."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "IcebergFramework"


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergSiblingDictRunAll(SiblingDictRunAllTestBase):
    """Dict-returning sibling consumers end-to-end through run_all."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "IcebergFramework"


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergMissingColumnRunAll(MissingColumnRunAllTestBase):
    """Undeclared column reads on the PyArrow table Iceberg hands to consumers raise an explanatory error."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "IcebergFramework"

    @classmethod
    def consumer_fgs(cls) -> list[type[MissingColumnConsumer]]:
        return [PyArrowKeyReadConsumer, PyArrowFilterReadConsumer]


class IcebergNativeRoot(MissingColumnRoot):
    """Root producing a real native IcebergTable; projecting an unknown column raises in its scan."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return real_iceberg_table(pa.table({"mc_declared": [1, 2, 3]}))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}


class IcebergNativeScanReadConsumer(MissingColumnConsumer):
    """Projects the undeclared column in a native Iceberg scan."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return data.scan(selected_fields=(UNDECLARED_COLUMN,)).to_arrow()

    @classmethod
    def cause_type(cls) -> Any:
        return ValueError


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
@pytest.mark.usefixtures("iceberg_table_source")
class TestIcebergNativeMissingColumnRunAll(TestIcebergMissingColumnRunAll):
    """Undeclared column reads on a native Iceberg scan raise an explanatory error."""

    @classmethod
    def root_fg(cls) -> type[FeatureGroup]:
        return IcebergNativeRoot

    @classmethod
    def consumer_fgs(cls) -> list[type[MissingColumnConsumer]]:
        return [IcebergNativeScanReadConsumer]
