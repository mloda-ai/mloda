"""
PyArrow implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.optional_dependency import loaded, require
from mloda.provider import FeatureSet, Implementation
from mloda.user.pyarrow import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

if TYPE_CHECKING:
    import numpy as np

_NUMPY_REASON = "pyarrow row-wise aggregation across multiple columns"


def _reduce_without_nan_warning(
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


class PyArrowAggregation(Implementation):
    """
    PyArrow implementation of aggregated feature group.

    Supports multiple aggregation types in a single class.
    """

    feature_group = AggregatedFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(cls, group: type[AggregatedFeatureGroup], data: Any, features: FeatureSet) -> Any:
        return group.aggregate(
            data,
            features,
            perform=cls._perform_aggregation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
        )

    @classmethod
    def _add_result_to_data(cls, data: pa.Table, feature_name: str, result: Any) -> pa.Table:
        """Add the result to the Table. Also accepts plain Python values from overrides."""
        numpy = loaded("numpy")
        if numpy is not None and isinstance(result, numpy.ndarray):
            # Multi-column (row-wise) aggregation: one value per row already.
            result_array = pa.array(result)
        else:
            # Single-column (vertical) aggregation: a scalar broadcast to every row.
            result_array = pa.repeat(result, data.num_rows)

        if feature_name in data.schema.names:
            column_index = data.schema.names.index(feature_name)
            data = data.remove_column(column_index)
            return data.append_column(feature_name, result_array)
        else:
            return data.append_column(feature_name, result_array)

    @classmethod
    def _perform_aggregation(cls, data: pa.Table, aggregation_type: str, in_features: list[str]) -> Any:
        """
        Perform the aggregation using PyArrow compute functions.

        Supports both single-column and multi-column aggregation:
        - Single column: aggregates values within the column (returns scalar)
        - Multi-column: aggregates across columns row-wise (returns array)

        Args:
            data: The PyArrow Table
            aggregation_type: The type of aggregation to perform
            in_features: List of source feature names (may be single or multiple columns)

        Returns:
            The result of the aggregation (pa.Scalar for single-column, preserving type on zero-row and
            all-null input; array for multi-column)
        """
        if len(in_features) > 1:
            np = require("numpy", _NUMPY_REASON)

            # Multi-column: aggregate across columns row-wise
            # PyArrow doesn't have direct horizontal operations, need to implement manually
            columns = [data.column(name) for name in in_features]

            # Cast to float64 only when a null is present, so it can be represented as NaN for the
            # np.nan* reducers to skip. Casting unconditionally would both force every result to
            # float64 and silently lose precision for int64 values beyond +/-2**53.
            if any(col.null_count > 0 for col in columns):
                arrays = [pc.cast(col, pa.float64()).to_numpy() for col in columns]
            else:
                arrays = [col.to_numpy() for col in columns]
            stacked = np.column_stack(arrays)

            # Reducers run on the null mask below; degenerate rows are patched so numpy never warns.
            # sum/avg/mean/std/var/count skip only null cells, so a genuine NaN propagates (count
            # counts it); the null mask comes from Arrow, not from the NaN-cast values.
            null_mask = np.column_stack([col.is_null().to_numpy(zero_copy_only=False) for col in columns])
            valid_counts = np.sum(~null_mask, axis=1)
            zeroed = np.where(null_mask, 0, stacked)

            if aggregation_type == "sum":
                result = zeroed.sum(axis=1)
            elif aggregation_type == "count":
                result = valid_counts
            elif aggregation_type in ["avg", "mean", "std", "var"]:
                row_sums = zeroed.sum(axis=1)
                means = np.full(row_sums.shape, np.nan)
                np.divide(row_sums, valid_counts, out=means, where=valid_counts > 0)
                if aggregation_type in ["avg", "mean"]:
                    result = means
                else:
                    squares = np.where(null_mask, 0.0, stacked - means[:, np.newaxis]) ** 2
                    variances = np.full(row_sums.shape, np.nan)
                    np.divide(squares.sum(axis=1), valid_counts - 1, out=variances, where=valid_counts >= 2)
                    result = variances if aggregation_type == "var" else np.sqrt(variances)
            elif aggregation_type in ["min", "max", "median"]:
                all_nan_rows = np.sum(~np.isnan(stacked), axis=1) == 0
                if aggregation_type == "min":
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmin(a, axis=1))
                elif aggregation_type == "max":
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmax(a, axis=1))
                else:
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmedian(a, axis=1))
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            # Convert back to PyArrow array (will be added as column)
            return result
        else:
            # Single column: vertical aggregation (returns scalar)
            column = data.column(in_features[0])

            if aggregation_type == "sum":
                return pc.sum(column)
            elif aggregation_type == "min":
                return pc.min(column)
            elif aggregation_type == "max":
                return pc.max(column)
            elif aggregation_type in ["avg", "mean"]:
                return pc.mean(column)
            elif aggregation_type == "count":
                return pc.count(column)
            elif aggregation_type == "std":
                return pc.stddev(column, ddof=1)
            elif aggregation_type == "var":
                return pc.variance(column, ddof=1)
            elif aggregation_type == "median":
                # PyArrow doesn't have a direct median function
                # We can approximate it using quantile with q=0.5
                # quantile returns an array, so we need to extract the first value
                result = pc.quantile(column, q=0.5)
                return result[0]
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
