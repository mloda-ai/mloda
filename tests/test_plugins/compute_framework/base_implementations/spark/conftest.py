"""
Shared fixtures for Spark compute framework tests.

This module provides centralized fixtures for all Spark-related tests to avoid
JVM conflicts and ensure proper resource management across test sessions.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

The shared SparkSession fixture ensures that only one Spark context exists
across all test files, preventing JVM conflicts and resource issues.
"""

import os
import pytest
from typing import Any

import logging

logger = logging.getLogger(__name__)

# Check PySpark availability and Java environment
try:
    from pyspark.sql import SparkSession

    # Check if JAVA_HOME is set
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        logger.warning("JAVA_HOME environment variable is not set. Spark tests will be skipped.")
        PYSPARK_AVAILABLE = False
        SKIP_REASON = "JAVA_HOME environment variable is not set"
    else:
        logger.info(f"JAVA_HOME found: {java_home}")
        PYSPARK_AVAILABLE = True
        SKIP_REASON = None  # type: ignore

except ImportError as e:
    logger.warning(f"PySpark is not installed: {e}. Spark tests will be skipped.")
    SparkSession = None
    PYSPARK_AVAILABLE = False
    SKIP_REASON = "PySpark is not installed"


@pytest.fixture(scope="session")
def spark_session() -> Any:
    """
    Create a single SparkSession for all Spark tests in this directory.

    This fixture is session-scoped to ensure that only one SparkSession exists
    across all test files, preventing JVM conflicts and resource issues.

    The fixture configures Spark with settings suitable for testing:
    - Local mode with single thread to avoid resource conflicts
    - Disabled adaptive query execution for predictable behavior
    - Arrow integration enabled for PyArrow compatibility
    - Reduced memory settings for test environments
    - Unique app name to avoid conflicts

    The fixture ensures proper cleanup by stopping the SparkSession after all tests complete.
    """
    if not PYSPARK_AVAILABLE:
        pytest.skip(SKIP_REASON or "PySpark is not available")

    # Configure Spark for testing with unique app name
    spark = (
        SparkSession.builder.appName("mloda-spark-tests-session")
        .master("local[1]")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse-tests")  # nosec
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.sql.shuffle.partitions", "4")  # Reduce partitions for tests
        .config("spark.default.parallelism", "2")  # Reduce parallelism for tests
        .getOrCreate()
    )

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Created shared SparkSession for all Spark tests")

    yield spark

    # Cleanup - this runs after all tests in the session complete
    try:
        logger.info("Stopping shared SparkSession")
        spark.stop()

        # Additional cleanup to ensure JVM resources are released
        from pyspark import SparkContext

        if SparkContext._active_spark_context is not None:
            SparkContext._active_spark_context.stop()
            SparkContext._active_spark_context = None

    except Exception as e:
        logger.warning(f"Error during SparkSession cleanup: {e}")


# Export availability flags for use in test files
__all__ = ["spark_session", "PYSPARK_AVAILABLE", "SKIP_REASON"]
