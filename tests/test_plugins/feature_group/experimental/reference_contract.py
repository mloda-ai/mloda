"""Shared values and comparison for the cross-framework reference contract tests."""

import math
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

NAN_LIKE_VALUES = [
    pytest.param(np.float32("nan"), id="np_float32_nan"),
    pytest.param(pd.NA, id="pd_na"),
    pytest.param(pd.NaT, id="pd_nat"),
]


class NAType:
    """A user class that only shares its name with the pandas NA type."""


class _ArrayNe:
    def __eq__(self, other: object) -> npt.NDArray[np.bool_]:  # type: ignore[override]
        return np.array([True, False])

    def __ne__(self, other: object) -> npt.NDArray[np.bool_]:  # type: ignore[override]
        return np.array([False, True])

    def __hash__(self) -> int:
        return 1


class _RaisingNe:
    def __ne__(self, other: object) -> bool:
        raise RuntimeError("no comparison")

    def __hash__(self) -> int:
        return 2


class NeverEqual:
    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 3


ODD_CELL_KINDS = [
    pytest.param(lambda i: np.array([float(i), 2.0]), id="numpy_array"),
    pytest.param(lambda i: _ArrayNe(), id="array_ne"),
    pytest.param(lambda i: _RaisingNe(), id="raising_ne"),
    pytest.param(lambda i: NeverEqual(), id="never_equal"),
    pytest.param(lambda i: NAType(), id="user_class_named_natype"),
]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def same(actual: Any, expected: Any) -> bool:
    """None equals only None, NaN only NaN, numbers within a tight tolerance, anything else by equality."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if _is_nan(actual) or _is_nan(expected):
        return _is_nan(actual) and _is_nan(expected)
    if isinstance(actual, Real) and isinstance(expected, Real):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    return bool(actual == expected)


def same_on_pandas(actual: Any, expected: Any) -> bool:
    """Like same, but None, NaN and pd.NA are interchangeable (a pandas float column cannot hold null)."""
    missing_actual = actual is None or _is_nan(actual) or actual is pd.NA
    missing_expected = expected is None or _is_nan(expected) or expected is pd.NA
    if missing_actual or missing_expected:
        return missing_actual and missing_expected
    return same(actual, expected)
