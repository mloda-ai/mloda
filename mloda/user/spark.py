# Spark compute framework. Backend: pyspark (install: mloda[spark]; the extra also pulls pyarrow, which
# only the pyarrow transformer needs, so is_available() checks pyspark alone).
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import (
    SparkFramework as SparkFramework,
)

__all__ = ["SparkFramework"]
