"""SparkFramework passes the compute framework shape check."""

import pytest

from mloda._testing.compute_framework_shape import applicable_aspects, check_compute_framework_shape
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark not available")
def test_spark_framework_shape() -> None:
    assert check_compute_framework_shape(SparkFramework).failures == ()
    assert applicable_aspects(SparkFramework) == ("filter", "mask", "merge")
