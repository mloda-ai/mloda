"""
PyArrow implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, List, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class PyArrowAggregatedFeatureGroup(AggregatedFeatureGroup):
    """
    PyArrow implementation of aggregated feature group.

    Supports multiple aggregation types in a single class.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with PyArrow."""
        return {PyArrowTable}

    @classmethod
    def _get_available_columns(cls, data: pa.Table) -> Set[str]:
        """Get the set of available column names from the Table schema."""
        return set(data.schema.names)

    @classmethod
    def _check_source_features_exist(cls, data: pa.Table, feature_names: List[str]) -> None:
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
        # Create an array with the aggregated result repeated for each row
        repeat_count = data.num_rows
        repeated_result = pa.array([result] * repeat_count)

        # Add the new column to the table
        return data.append_column(feature_name, repeated_result)

    @classmethod
    def _perform_aggregation(cls, data: pa.Table, aggregation_type: str, in_features: List[str]) -> Any:
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

            # Convert columns to numpy for easier row-wise operations
            import numpy as np

            arrays = [col.to_numpy() for col in columns]
            stacked = np.column_stack(arrays)

            if aggregation_type == "sum":
                result = np.sum(stacked, axis=1)
            elif aggregation_type == "min":
                result = np.min(stacked, axis=1)
            elif aggregation_type == "max":
                result = np.max(stacked, axis=1)
            elif aggregation_type in ["avg", "mean"]:
                result = np.mean(stacked, axis=1)
            elif aggregation_type == "count":
                result = np.sum(~np.isnan(stacked), axis=1)
            elif aggregation_type == "std":
                result = np.std(stacked, axis=1)
            elif aggregation_type == "var":
                result = np.var(stacked, axis=1)
            elif aggregation_type == "median":
                result = np.median(stacked, axis=1)
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
                return pc.stddev(column).as_py()
            elif aggregation_type == "var":
                return pc.variance(column).as_py()
            elif aggregation_type == "median":
                # PyArrow doesn't have a direct median function
                # We can approximate it using quantile with q=0.5
                # quantile returns an array, so we need to extract the first value
                result = pc.quantile(column, q=0.5)
                return result[0].as_py()
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
