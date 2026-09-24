from functools import reduce
from typing import Any
from mloda.user import SingleFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine

try:
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.expressions import (
        AlwaysTrue,
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        EqualTo,
        And,
        Reference,
    )
except ImportError:
    IcebergTable: type[Any] | None = None  # type: ignore[no-redef]
    AlwaysTrue: type[Any] | None = None  # type: ignore[no-redef]
    LessThan: type[Any] | None = None  # type: ignore[no-redef]
    GreaterThanOrEqual: type[Any] | None = None  # type: ignore[no-redef]
    LessThanOrEqual: type[Any] | None = None  # type: ignore[no-redef]
    EqualTo: type[Any] | None = None  # type: ignore[no-redef]
    And: type[Any] | None = None  # type: ignore[no-redef]
    Reference: type[Any] | None = None  # type: ignore[no-redef]

_PUSHDOWN_FILTER_TYPES = frozenset({"range", "min", "max", "equal"})


class IcebergFilterEngine(PyArrowFilterEngine):
    """Filters Iceberg tables by scan pushdown plus the PyArrow filters; other data via the PyArrow filters."""

    @classmethod
    def apply_filters(cls, data: Any, features: Any) -> Any:
        """Push range, min, max and equal into the Iceberg scan, then apply every filter to the pa.Table it returns."""
        if not isinstance(data, IcebergTable):
            return super().apply_filters(data, features)

        applicable = cls.applicable_filters(features)
        if not applicable:
            return data

        # Build pushable expressions before scanning so a malformed filter raises without a scan.
        expressions: list[Any] = [
            cls._build_iceberg_expression(f) for f in applicable if f.filter_type in _PUSHDOWN_FILTER_TYPES
        ]
        row_filter = reduce(And, expressions, AlwaysTrue())
        # The PyArrow pass applies the filters Iceberg cannot push and re-checks the pushed ones.
        return super().apply_filters(data.scan(row_filter=row_filter).to_arrow(), features)

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
