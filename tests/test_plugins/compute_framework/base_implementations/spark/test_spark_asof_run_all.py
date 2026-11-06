"""End-to-end ASOF (point-in-time) join integration test for Spark, on the shipped AsofRunAllSuite.

The session-scoped ``spark_session`` fixture must exist before the adapter borrows the active session.
"""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from tests.test_plugins.compute_framework.base_implementations.spark.spark_session_adapter import SparkSessionAdapter


class TestSparkAsofRunAll(SparkSessionAdapter, AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on Spark."""
