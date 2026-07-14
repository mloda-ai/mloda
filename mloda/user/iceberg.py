# Iceberg compute framework. Importing this module works without pyiceberg installed;
# the framework then reports itself unavailable via is_available().
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import (
    IcebergFramework as IcebergFramework,
)

__all__ = ["IcebergFramework"]
