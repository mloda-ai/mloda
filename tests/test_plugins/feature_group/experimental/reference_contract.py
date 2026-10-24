"""Shared value comparison for the cross-framework reference contract tests."""

import math
from numbers import Real
from typing import Any

import pandas as pd


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


def is_missing(value: Any) -> bool:
    """None, NaN and pd.NA all count as missing."""
    return value is None or _is_nan(value) or value is pd.NA


def same_missing_agnostic(actual: Any, expected: Any) -> bool:
    """A missing expected value matches any missing representation; otherwise strict `same`."""
    if is_missing(expected):
        return is_missing(actual)
    return not is_missing(actual) and same(actual, expected)


def same_on_pandas(actual: Any, expected: Any) -> bool:
    """Like same, but None and NaN are interchangeable (a pandas float column cannot hold null)."""
    missing_actual = actual is None or _is_nan(actual)
    missing_expected = expected is None or _is_nan(expected)
    if missing_actual or missing_expected:
        return missing_actual and missing_expected
    return same(actual, expected)
