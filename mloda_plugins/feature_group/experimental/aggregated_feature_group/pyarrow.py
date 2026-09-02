"""
PyArrow implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework

from mloda.user.pyarrow import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


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
    patched = stacked.copy()
    nan_mask = np.isnan(patched)
    patched[degenerate_rows[:, np.newaxis] & nan_mask] = 0.0
    result = reducer(patched)
    return np.where(degenerate_rows, np.nan, result)


class PyArrowAggregatedFeatureGroup(AggregatedFeatureGroup):
    """
    PyArrow implementation of aggregated feature group.

    Supports multiple aggregation types in a single class.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        """Specify that this feature group works with PyArrow."""
        return {PyArrowTable}

    @classmethod
    def _get_available_columns(cls, data: pa.Table) -> set[str]:
        """Get the set of available column names from the Table schema."""
        return set(data.schema.names)

    @classmethod
    def _check_source_features_exist(cls, data: pa.Table, feature_names: list[str]) -> None:
        """
        Check if the resolved features exist in the Table.

        Args:
            data: The PyArrow Table
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the resolved features exist in the data
        """
        schema_names = set(data.schema.names)
        missing_features = [name for name in feature_names if name not in schema_names]
        if len(missing_features) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data. Available columns: {list(schema_names)}"
            )

    @classmethod
    def _add_result_to_data(cls, data: pa.Table, feature_name: str, result: Any) -> pa.Table:
        """Add the result to the Table."""
        if isinstance(result, np.ndarray):
            # Multi-column (row-wise) aggregation: one value per row already.
            result_array = pa.array(result)
        else:
            # Single-column (vertical) aggregation: a scalar broadcast to every row.
            result_array = pa.array([result] * data.num_rows)

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
            The result of the aggregation (scalar for single-column, array for multi-column)
        """
        if len(in_features) > 1:
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

            # np.nansum and the count below never warn on an all-NaN row (nansum returns 0,
            # count is not nan-aware). The other reducers warn on rows they can't skipna
            # over: an all-NaN row for min/max/mean/median, or a row with fewer than 2 valid
            # values for std/var (ddof=1 needs >=2). Route those through the helper, which
            # patches just the degenerate rows so the warning never fires, instead of
            # suppressing it process-wide via warnings.catch_warnings().
            if aggregation_type == "sum":
                result = np.nansum(stacked, axis=1)
            elif aggregation_type == "count":
                result = np.sum(~np.isnan(stacked), axis=1)
            elif aggregation_type in ["min", "max", "avg", "mean", "median"]:
                all_nan_rows = np.sum(~np.isnan(stacked), axis=1) == 0
                if aggregation_type == "min":
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmin(a, axis=1))
                elif aggregation_type == "max":
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmax(a, axis=1))
                elif aggregation_type in ["avg", "mean"]:
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmean(a, axis=1))
                else:
                    result = _reduce_without_nan_warning(stacked, all_nan_rows, lambda a: np.nanmedian(a, axis=1))
            elif aggregation_type in ["std", "var"]:
                under_two_valid_rows = np.sum(~np.isnan(stacked), axis=1) < 2
                if aggregation_type == "std":
                    result = _reduce_without_nan_warning(
                        stacked, under_two_valid_rows, lambda a: np.nanstd(a, axis=1, ddof=1)
                    )
                else:
                    result = _reduce_without_nan_warning(
                        stacked, under_two_valid_rows, lambda a: np.nanvar(a, axis=1, ddof=1)
                    )
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            # Convert back to PyArrow array (will be added as column)
            return result
        else:
            # Single column: vertical aggregation (returns scalar)
            column = data.column(in_features[0])

            if aggregation_type == "sum":
                return pc.sum(column).as_py()
            elif aggregation_type == "min":
                return pc.min(column).as_py()
            elif aggregation_type == "max":
                return pc.max(column).as_py()
            elif aggregation_type in ["avg", "mean"]:
                return pc.mean(column).as_py()
            elif aggregation_type == "count":
                return pc.count(column).as_py()
            elif aggregation_type == "std":
                return pc.stddev(column, ddof=1).as_py()
            elif aggregation_type == "var":
                return pc.variance(column, ddof=1).as_py()
            elif aggregation_type == "median":
                # PyArrow doesn't have a direct median function
                # We can approximate it using quantile with q=0.5
                # quantile returns an array, so we need to extract the first value
                result = pc.quantile(column, q=0.5)
                return result[0].as_py()
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
