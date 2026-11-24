# SQLite compute framework. Backend: stdlib sqlite3 plus pyarrow (install: mloda[sqlite]).
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import (
    SqliteFramework as SqliteFramework,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
    SqliteRelation as SqliteRelation,
)

__all__ = ["SqliteFramework", "SqliteRelation"]
