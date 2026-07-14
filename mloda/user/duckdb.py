# DuckDB compute framework. Backend: duckdb (install: mloda[duckdb]).
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
    DuckDBFramework as DuckDBFramework,
)

__all__ = ["DuckDBFramework"]
