import math
from collections.abc import Iterable
from typing import Any


def is_null_or_nan(value: Any) -> bool:
    """Return True for None or a float NaN (numpy float64 included, a float subclass)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def split_null_or_nan(values: Iterable[Any]) -> tuple[list[Any], bool]:
    """Split values into the ones that are neither None nor NaN, and whether any were."""
    present = []
    has_null_or_nan = False
    for value in values:
        if is_null_or_nan(value):
            has_null_or_nan = True
        else:
            present.append(value)
    return present, has_null_or_nan
