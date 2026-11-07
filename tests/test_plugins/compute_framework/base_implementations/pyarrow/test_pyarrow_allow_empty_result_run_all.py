"""End-to-end empty-result policy test for PyArrow.

Consumes the shipped EmptyResultRunAllSuite. No connection is required.
"""

from collections.abc import Mapping

from mloda._testing.compute_framework.run_all_suite import (
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    pyarrow_filter_read,
    pyarrow_key_read,
)
from mloda._testing.framework_adapter import PyArrowAdapter


def _arrow_invalid() -> type[BaseException]:
    import pyarrow as pa

    cause: type[BaseException] = pa.ArrowInvalid
    return cause


class TestPyArrowAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on PyArrow."""

    adapter_class = PyArrowAdapter


class TestPyArrowSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all."""

    adapter_class = PyArrowAdapter


class TestPyArrowMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on PyArrow raise an explanatory error."""

    adapter_class = PyArrowAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "PyArrowKeyReadConsumer": (pyarrow_key_read, lambda: KeyError),
            "PyArrowFilterReadConsumer": (pyarrow_filter_read, _arrow_invalid),
        }
