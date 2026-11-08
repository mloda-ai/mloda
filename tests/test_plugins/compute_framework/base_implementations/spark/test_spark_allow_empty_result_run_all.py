"""End-to-end empty-result policy test for Spark.

Consumes the shipped EmptyResultRunAllSuite. The session-scoped ``spark_session`` fixture must exist before the
adapter borrows the active session.
"""

from collections.abc import Mapping
from typing import Any

from mloda._testing.compute_framework.run_all_suite import (
    UNDECLARED_COLUMN,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    attribute_read,
)
from tests.test_plugins.compute_framework.base_implementations.spark.spark_session_adapter import SparkSessionAdapter


class TestSparkAllowEmptyResultRunAll(SparkSessionAdapter, EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on Spark."""


class TestSparkSiblingDictRunAll(SparkSessionAdapter, SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all on Spark."""


def _col_read(data: Any) -> Any:
    from pyspark.sql import functions as F

    return data.select(F.col(UNDECLARED_COLUMN))


def _analysis_exception() -> type[BaseException]:
    from pyspark.errors import AnalysisException

    cause: type[BaseException] = AnalysisException
    return cause


class TestSparkMissingColumnRunAll(SparkSessionAdapter, MissingColumnRunAllSuite):
    """Undeclared column reads on Spark raise an explanatory error."""

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "SparkColReadConsumer": (_col_read, _analysis_exception),
            "AttributeReadConsumer": (attribute_read, lambda: AttributeError),
        }
