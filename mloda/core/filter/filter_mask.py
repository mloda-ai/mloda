import sys
from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.filter.single_filter import SingleFilter


class FilterMask:
    """Builds a combined boolean mask from SingleFilters for inline use in calculate_feature()."""

    @classmethod
    def build(
        cls,
        data: Any,
        filters: Optional[set[SingleFilter]],
        column: str,
    ) -> Any:
        """Build a combined boolean mask for all filters matching the given column.

        Returns a framework-native boolean array/series. If no filters match,
        returns an all-True mask.
        """
        framework = cls._detect_framework(data)
        matching = cls._matching_filters(filters, column)
        if not matching:
            return cls._all_true(data, framework)

        mask = cls._all_true(data, framework)
        for sf in matching:
            single_mask = cls._single_mask(data, sf, column, framework)
            mask = cls._combine(mask, single_mask, framework)
        return mask

    @classmethod
    def _detect_framework(cls, data: Any) -> str:
        if isinstance(data, pa.Table):
            return "pyarrow"
        type_module = type(data).__module__
        type_name = type(data).__name__
        if type_module.startswith("pandas") and type_name == "DataFrame":
            return "pandas"
        if type_module.startswith("polars") and type_name == "DataFrame":
            return "polars"
        if isinstance(data, list):
            return "python_dict"
        raise ValueError(f"Unsupported data type for FilterMask: {type(data)}")

    @classmethod
    def _matching_filters(cls, filters: Optional[set[SingleFilter]], column: str) -> list[SingleFilter]:
        if filters is None:
            return []
        return [sf for sf in filters if sf.name == column]

    @classmethod
    def _all_true(cls, data: Any, framework: str) -> Any:
        if framework == "pyarrow":
            return pa.array([True] * data.num_rows)
        if framework == "pandas":
            pd = sys.modules["pandas"]
            return pd.Series([True] * len(data))
        if framework == "polars":
            pl = sys.modules["polars"]
            return pl.Series([True] * len(data))
        # python_dict
        return [True] * len(data)

    @classmethod
    def _single_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        filter_type = sf.filter_type
        if filter_type == "equal":
            return cls._equal_mask(data, sf, column, framework)
        if filter_type == "min":
            return cls._min_mask(data, sf, column, framework)
        if filter_type == "max":
            return cls._max_mask(data, sf, column, framework)
        if filter_type == "range":
            return cls._range_mask(data, sf, column, framework)
        if filter_type == "categorical_inclusion":
            return cls._categorical_inclusion_mask(data, sf, column, framework)
        raise ValueError(
            f"Unsupported filter type for FilterMask: '{filter_type}'. "
            f"Supported types: equal, min, max, range, categorical_inclusion."
        )

    @classmethod
    def _equal_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        value = sf.parameter.value
        if framework == "pyarrow":
            return pc.equal(data[column], value)
        if framework == "pandas":
            return data[column] == value
        if framework == "polars":
            return data[column] == value
        # python_dict
        return [row.get(column) == value for row in data]

    @classmethod
    def _min_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        value = sf.parameter.value
        if framework == "pyarrow":
            return pc.greater_equal(data[column], value)
        if framework == "pandas":
            return data[column] >= value
        if framework == "polars":
            return data[column] >= value
        # python_dict
        return [row.get(column) is not None and row.get(column) >= value for row in data]

    @classmethod
    def _max_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        # Complex parameter (max + max_exclusive) takes priority over simple (value)
        max_val = sf.parameter.max_value
        if max_val is not None:
            if sf.parameter.max_exclusive:
                return cls._compare_lt(data, column, max_val, framework)
            return cls._compare_lte(data, column, max_val, framework)
        value = sf.parameter.value
        return cls._compare_lte(data, column, value, framework)

    @classmethod
    def _range_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        min_val = sf.parameter.min_value
        max_val = sf.parameter.max_value
        min_mask = cls._compare_gte(data, column, min_val, framework)
        if sf.parameter.max_exclusive:
            max_mask = cls._compare_lt(data, column, max_val, framework)
        else:
            max_mask = cls._compare_lte(data, column, max_val, framework)
        return cls._combine(min_mask, max_mask, framework)

    @classmethod
    def _categorical_inclusion_mask(cls, data: Any, sf: SingleFilter, column: str, framework: str) -> Any:
        values = sf.parameter.values
        if framework == "pyarrow":
            return pc.is_in(data[column], pa.array(values))
        if framework == "pandas":
            return data[column].isin(values)
        if framework == "polars":
            return data[column].is_in(values)
        # python_dict
        values_set = set(values) if values else set()
        return [row.get(column) in values_set for row in data]

    # --- Comparison helpers ---

    @classmethod
    def _compare_gte(cls, data: Any, column: str, value: Any, framework: str) -> Any:
        if framework == "pyarrow":
            return pc.greater_equal(data[column], value)
        if framework == "pandas":
            return data[column] >= value
        if framework == "polars":
            return data[column] >= value
        return [row.get(column) is not None and row.get(column) >= value for row in data]

    @classmethod
    def _compare_lte(cls, data: Any, column: str, value: Any, framework: str) -> Any:
        if framework == "pyarrow":
            return pc.less_equal(data[column], value)
        if framework == "pandas":
            return data[column] <= value
        if framework == "polars":
            return data[column] <= value
        return [row.get(column) is not None and row.get(column) <= value for row in data]

    @classmethod
    def _compare_lt(cls, data: Any, column: str, value: Any, framework: str) -> Any:
        if framework == "pyarrow":
            return pc.less(data[column], value)
        if framework == "pandas":
            return data[column] < value
        if framework == "polars":
            return data[column] < value
        return [row.get(column) is not None and row.get(column) < value for row in data]

    @classmethod
    def _combine(cls, mask1: Any, mask2: Any, framework: str) -> Any:
        if framework == "pyarrow":
            return pc.and_(mask1, mask2)
        if framework in ("pandas", "polars"):
            return mask1 & mask2
        # python_dict
        return [a and b for a, b in zip(mask1, mask2)]
