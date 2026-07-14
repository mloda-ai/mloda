# DuckDB compute framework. Importing this module works without duckdb installed;
# the framework then reports itself unavailable via is_available().
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
    DuckDBFramework as DuckDBFramework,
)

__all__ = ["DuckDBFramework"]
