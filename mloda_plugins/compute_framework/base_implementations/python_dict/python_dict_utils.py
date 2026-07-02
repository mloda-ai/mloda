"""Shared helpers for the PythonDict framework's columnar ``dict[str, list[Any]]`` data."""

from typing import Any


def row_count(data: dict[str, list[Any]]) -> int:
    """Number of rows in a columnar dict: length of the first column, 0 for the empty dict."""
    for column in data.values():
        return len(column)
    return 0


def rows_to_columnar(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Pivot row-wise ``list[dict]`` to columnar. Keys must be homogeneous across rows."""
    if not rows:
        return {}

    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Expected list of dictionaries, but item at index {i} is {type(item)}")

    first_keys = list(rows[0].keys())
    first_keys_set = set(first_keys)
    for i, item in enumerate(rows):
        if set(item.keys()) != first_keys_set:
            raise ValueError(f"Inconsistent row keys at index {i}: expected {first_keys_set}, got {set(item.keys())}.")

    return {key: [row[key] for row in rows] for key in first_keys}
