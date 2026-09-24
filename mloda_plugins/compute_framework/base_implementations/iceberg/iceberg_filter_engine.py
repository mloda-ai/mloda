from typing import Any
from mloda.user import SingleFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine

try:
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.expressions import (
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        EqualTo,
        And,
        Reference,
    )
except ImportError:
    IcebergTable: type[Any] | None = None  # type: ignore[no-redef]
    LessThan: type[Any] | None = None  # type: ignore[no-redef]
    GreaterThanOrEqual: type[Any] | None = None  # type: ignore[no-redef]
    LessThanOrEqual: type[Any] | None = None  # type: ignore[no-redef]
    EqualTo: type[Any] | None = None  # type: ignore[no-redef]
    And: type[Any] | None = None  # type: ignore[no-redef]
    Reference: type[Any] | None = None  # type: ignore[no-redef]


class IcebergFilterEngine(PyArrowFilterEngine):
    """Filters Iceberg tables by scan pushdown and pa.Table data with the PyArrow filters."""

    @classmethod
    def final_filters(cls) -> bool:
        """Iceberg filters are applied during scan, not after feature calculation."""
        return False

    @classmethod
    def apply_filters(cls, data: Any, features: Any) -> Any:
        """
        Push filters into an Iceberg table scan; other data goes through the PyArrow filters.

        Returns the filtered scan as a pa.Table, or the input unchanged when no filter applies.
        """
        if not isinstance(data, IcebergTable):
            return super().apply_filters(data, features)

        # Build Iceberg filter expressions
        filter_expressions = []
        for single_filter in cls.applicable_filters(features):
            iceberg_expr = cls._build_iceberg_expression(single_filter)
            if iceberg_expr is not None:
                filter_expressions.append(iceberg_expr)

        if not filter_expressions:
            return data

        # Combine multiple filters with AND
        combined_filter = filter_expressions[0]
        for expr in filter_expressions[1:]:
            if And is not None:
                combined_filter = And(combined_filter, expr)

        # A bare scan has no schema the framework can read
        return data.scan(row_filter=combined_filter).to_arrow()

    @classmethod
    def _build_iceberg_expression(cls, filter_feature: SingleFilter) -> Any:
        """Build an Iceberg filter expression from a SingleFilter."""
        if any(expr is None for expr in [EqualTo, And, LessThan, GreaterThanOrEqual, LessThanOrEqual, Reference]):
            return None

        column_name = str(filter_feature.filter_feature.name)
        filter_type = filter_feature.filter_type

        if filter_type == "equal":
            value = cls._extract_parameter_value(filter_feature, "value")
            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
            return EqualTo(Reference(column_name), value)

        elif filter_type == "min":
            value = cls._extract_parameter_value(filter_feature, "value")
            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
            return GreaterThanOrEqual(Reference(column_name), value)

        elif filter_type == "max":
            has_max = cls._has_parameter(filter_feature, "max")
            has_value = cls._extract_parameter_value(filter_feature, "value") is not None

            if has_max:
                min_param, max_param, is_max_exclusive = cls.get_min_max_operator(filter_feature)
                if min_param is not None:
                    raise ValueError(
                        f"Filter parameter {filter_feature.parameter} not supported as max filter: "
                        f"{filter_feature.name}"
                    )
                if is_max_exclusive is True:
                    return LessThan(Reference(column_name), max_param)
                return LessThanOrEqual(Reference(column_name), max_param)
            elif has_value:
                value = cls._extract_parameter_value(filter_feature, "value")
                return LessThanOrEqual(Reference(column_name), value)
            else:
                raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

        elif filter_type == "range":
            min_param, max_param, is_max_exclusive = cls.get_min_max_operator(filter_feature)
            if min_param is None or max_param is None:
                raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

            expr_min = GreaterThanOrEqual(Reference(column_name), min_param)
            expr_max: Any
            if is_max_exclusive is True:
                expr_max = LessThan(Reference(column_name), max_param)
            else:
                expr_max = LessThanOrEqual(Reference(column_name), max_param)
            return And(expr_min, expr_max)

        else:
            raise NotImplementedError(f"Unsupported Iceberg filter type: {filter_type!r}")

    @classmethod
    def _extract_parameter_value(cls, filter_feature: SingleFilter, param_name: str) -> Any:
        """Extract a parameter value from filter feature."""
        if param_name == "value":
            return filter_feature.parameter.value
        elif param_name == "values":
            return filter_feature.parameter.values
        elif param_name == "min":
            return filter_feature.parameter.min_value
        elif param_name == "max":
            return filter_feature.parameter.max_value
        elif param_name == "max_exclusive":
            return filter_feature.parameter.max_exclusive
        return None

    @classmethod
    def _has_parameter(cls, filter_feature: SingleFilter, param_name: str) -> bool:
        """Check if filter feature has a specific parameter."""
        value = cls._extract_parameter_value(filter_feature, param_name)
        if param_name == "max_exclusive":
            return True
        return value is not None

    @classmethod
    def do_custom_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Custom filtering is not supported for Iceberg tables")
