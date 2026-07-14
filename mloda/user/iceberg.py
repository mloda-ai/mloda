# Iceberg compute framework. Backend: pyiceberg plus pyarrow (install: mloda[iceberg]).
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import (
    IcebergFramework as IcebergFramework,
)

__all__ = ["IcebergFramework"]
