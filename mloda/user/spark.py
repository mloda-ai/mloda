# Spark compute framework. Importing this module works without pyspark installed;
# the framework then reports itself unavailable via is_available().
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import (
    SparkFramework as SparkFramework,
)

__all__ = ["SparkFramework"]
