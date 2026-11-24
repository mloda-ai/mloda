# DuckDB compute framework and its SQL relation helpers. Backend: duckdb plus pyarrow (install: mloda[duckdb]).
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
    DuckDBFramework as DuckDBFramework,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
    DuckdbRelation as DuckdbRelation,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    pick_helper_column_name as pick_helper_column_name,
    quote_ident as quote_ident,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow as CurrentRow,
    FrameBound as FrameBound,
    Following as Following,
    OrderBy as OrderBy,
    Preceding as Preceding,
    Unbounded as Unbounded,
    WindowFrame as WindowFrame,
)

__all__ = [
    "CurrentRow",
    "DuckDBFramework",
    "DuckdbRelation",
    "Following",
    "FrameBound",
    "OrderBy",
    "Preceding",
    "Unbounded",
    "WindowFrame",
    "pick_helper_column_name",
    "quote_ident",
]
