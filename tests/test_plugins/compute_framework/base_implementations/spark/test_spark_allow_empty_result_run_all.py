"""End-to-end ``allow_empty_result`` policy test for Spark.

Consumes the shared EmptyResultRunAllTestBase. The session-scoped ``spark_session`` fixture
is captured into the instance via an autouse fixture, so the base's ``get_connection`` can
thread it through ``Feature.options`` + a ``DataAccessCollection``, mirroring how the Spark
integration tests pass the SparkSession to ``run_all``.
"""

from typing import Any, Optional

import pytest

from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on Spark."""

    @pytest.fixture(autouse=True)
    def _bind_spark_session(self, spark_session: Any) -> None:
        self._spark_session = spark_session

    @classmethod
    def compute_framework_name(cls) -> str:
        return "SparkFramework"

    def get_connection(self) -> Optional[Any]:
        """Spark requires the shared SparkSession as its connection object."""
        return self._spark_session
