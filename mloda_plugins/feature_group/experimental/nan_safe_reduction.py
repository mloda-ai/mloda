"""NaN-skipping row-wise reductions shared by the PyArrow feature groups; numpy is required lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mloda.core.optional_dependency import require

if TYPE_CHECKING:
    import numpy as np

_NUMPY_REASON = "NaN-skipping row-wise reduction"


def reduce_without_nan_warning(
    stacked: np.ndarray[Any, Any],
    degenerate_rows: np.ndarray[Any, Any],
    reducer: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
) -> np.ndarray[Any, Any]:
    """Run a np.nan* row-wise reducer without ever triggering its RuntimeWarning.

    Rows flagged as degenerate (e.g. all-NaN, or too few valid values for ddof=1) have their
    NaN cells temporarily patched with 0.0 in a copy so the reducer never sees a degenerate
    row; those rows' results are then overwritten back to NaN, matching the unpatched result.
    """
    if not degenerate_rows.any():
        return reducer(stacked)
    numpy = require("numpy", _NUMPY_REASON)
    patched = stacked.copy()
    nan_mask = numpy.isnan(patched)
    patched[degenerate_rows[:, numpy.newaxis] & nan_mask] = 0.0
    result = reducer(patched)
    restored: np.ndarray[Any, Any] = numpy.where(degenerate_rows, numpy.nan, result)
    return restored


def sample_variance_skipping_nan(stacked: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Row-wise ddof=1 variance over non-NaN cells; NaN for rows with fewer than 2 values."""
    numpy = require("numpy", _NUMPY_REASON)
    degenerate = numpy.sum(~numpy.isnan(stacked), axis=1) < 2
    return reduce_without_nan_warning(stacked, degenerate, lambda a: numpy.nanvar(a, axis=1, ddof=1))
