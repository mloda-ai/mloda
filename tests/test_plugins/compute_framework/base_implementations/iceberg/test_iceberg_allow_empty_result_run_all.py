"""End-to-end empty-result policy test for Iceberg.

Consumes the shipped EmptyResultRunAllSuite. ``IcebergFramework.transform`` emits a PyArrow table for dict input, so
no catalog connection is needed: run_all gets no connection. The native IcebergTable branch is covered elsewhere.
"""

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda._testing.compute_framework.run_all_suite import (
    UNDECLARED_COLUMN,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    pyarrow_filter_read,
    pyarrow_key_read,
)
from mloda._testing.framework_adapter import FrameworkAdapter, IcebergAdapter
from tests.test_plugins.compute_framework.base_implementations.iceberg.iceberg_test_table import real_iceberg_table

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyiceberg
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


def _arrow_invalid() -> type[BaseException]:
    import pyarrow as pa

    cause: type[BaseException] = pa.ArrowInvalid
    return cause


class TestIcebergAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on Iceberg."""

    adapter_class = IcebergAdapter

    def run_all_connection(self, adapter: FrameworkAdapter) -> Any | None:
        return None


class TestIcebergSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all."""

    adapter_class = IcebergAdapter

    def run_all_connection(self, adapter: FrameworkAdapter) -> Any | None:
        return None


class TestIcebergMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on the PyArrow table Iceberg hands to consumers raise an explanatory error."""

    adapter_class = IcebergAdapter

    def run_all_connection(self, adapter: FrameworkAdapter) -> Any | None:
        return None

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "PyArrowKeyReadConsumer": (pyarrow_key_read, lambda: KeyError),
            "PyArrowFilterReadConsumer": (pyarrow_filter_read, _arrow_invalid),
        }


def _native_scan_read(data: Any) -> Any:
    return data.scan(selected_fields=(UNDECLARED_COLUMN,)).to_arrow()


@pytest.mark.usefixtures("iceberg_table_source")
class TestIcebergNativeMissingColumnRunAll(TestIcebergMissingColumnRunAll):
    """Undeclared column reads on a native Iceberg scan raise an explanatory error."""

    def root_values(self, adapter: FrameworkAdapter) -> Any:
        """A real native IcebergTable; projecting an unknown column raises in its scan."""
        return real_iceberg_table(pa.table({"mc_declared": [1, 2, 3]}))

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {"IcebergNativeScanReadConsumer": (_native_scan_read, lambda: ValueError)}
