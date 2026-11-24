# DuckDB compute framework. Backend: duckdb plus pyarrow (install: mloda[duckdb]).
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
    DuckDBFramework as DuckDBFramework,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
    DuckdbRelation as DuckdbRelation,
)

__all__ = ["DuckDBFramework", "DuckdbRelation"]
