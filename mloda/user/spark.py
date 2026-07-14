# Spark compute framework. Backend: pyspark (install: mloda[spark]).
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import (
    SparkFramework as SparkFramework,
)

__all__ = ["SparkFramework"]
