"""Structured file suffix constants shared by the file readers and ReadDocument."""

CSV_SUFFIXES = (".csv", ".CSV")
PARQUET_SUFFIXES = (".parquet", ".PARQUET", ".pqt", ".PQT")
JSON_SUFFIXES = (".json", ".JSON")
FEATHER_SUFFIXES = (".feather",)
ORC_SUFFIXES = (".orc", ".ORC")

STRUCTURED_SUFFIXES: frozenset[str] = frozenset(
    CSV_SUFFIXES + PARQUET_SUFFIXES + JSON_SUFFIXES + FEATHER_SUFFIXES + ORC_SUFFIXES
)
