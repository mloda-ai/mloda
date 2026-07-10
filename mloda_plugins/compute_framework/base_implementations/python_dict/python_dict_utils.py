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


def is_columnar(data: Any) -> bool:
    """True iff data is a dict whose values are all lists of one shared length."""
    if not isinstance(data, dict):
        return False
    if not all(isinstance(column, list) for column in data.values()):
        return False
    return len({len(column) for column in data.values()}) <= 1


def validate_columnar_dict(data: dict[str, Any]) -> None:
    """Raise ValueError unless every value is a list and all value-lists share one length."""
    lengths = set()
    for key, column in data.items():
        if not isinstance(column, list):
            raise ValueError(f"Expected list column values, but column {key!r} is {type(column)}")
        lengths.add(len(column))
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent column lengths: {sorted(lengths)}.")


def columnar_to_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pivot a columnar dict to row-wise ``list[dict]``. Anything else raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected columnar dict (row-wise lists are no longer accepted), got {type(data)}")

    validate_columnar_dict(data)

    return [{key: data[key][i] for key in data} for i in range(row_count(data))]


def homogenize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return new row dicts carrying the union of keys in first-occurrence order, missing keys as ``None``."""
    keys: list[str] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Expected list of dictionaries, but item at index {i} is {type(item)}")
        for key in item:
            if key not in keys:
                keys.append(key)

    return [{key: row.get(key) for key in keys} for row in rows]
