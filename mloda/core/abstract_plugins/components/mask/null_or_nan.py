import math
import numbers
from collections.abc import Iterable
from typing import Any


def is_null_or_nan(value: Any) -> bool:
    """Return True for None or a non-integral real NaN (float, numpy float64, ...)."""
    return value is None or (
        isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral) and math.isnan(value)
    )


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
