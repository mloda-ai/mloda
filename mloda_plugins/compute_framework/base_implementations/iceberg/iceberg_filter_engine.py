from typing import Any, Optional, Type
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter

try:
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.expressions import GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual, EqualTo, And
except ImportError:
    IcebergTable: Optional[Type] = None  # type: ignore
    GreaterThan: Optional[Type] = None  # type: ignore
    LessThan: Optional[Type] = None  # type: ignore
    GreaterThanOrEqual: Optional[Type] = None  # type: ignore
    LessThanOrEqual: Optional[Type] = None  # type: ignore
    EqualTo: Optional[Type] = None  # type: ignore
    And: Optional[Type] = None  # type: ignore


class IcebergFilterEngine(BaseFilterEngine):
    """
    Filter engine for Iceberg tables using predicate pushdown.

    This engine translates mloda filter operations to Iceberg expressions
    for optimal performance through predicate pushdown.
    """

    @classmethod
    def final_filters(cls) -> bool:
        """Iceberg filters are applied during scan, not after feature calculation."""
        return False

    @classmethod
    def apply_filters(cls, data: Any, features: Any) -> Any:
        """
        Apply filters to Iceberg table using predicate pushdown.

        Args:
            data: Iceberg table
            features: Feature set with filter specifications

        Returns:
            Filtered Iceberg table scan result
        """
        if not isinstance(data, IcebergTable):
            # If it's not an Iceberg table, fall back to default filtering
            return super().apply_filters(data, features)

        if features.filters is None or len(features.filters) == 0:
            return data

        # Build Iceberg filter expressions
        filter_expressions = []
        for single_filter in features.filters:
            if single_filter.filter_feature.name not in features.get_all_names():
                continue

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

        # Apply filter to Iceberg table scan
        return data.scan(row_filter=combined_filter)

    @classmethod
    def _build_iceberg_expression(cls, filter_feature: SingleFilter) -> Any:
        """Build an Iceberg filter expression from a SingleFilter."""
        if any(expr is None for expr in [EqualTo, GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual]):
            return None

        column_name = filter_feature.filter_feature.name
        filter_type = filter_feature.filter_type

        if filter_type == "equal":
            value = cls._extract_parameter_value(filter_feature, "value")
            return EqualTo(column_name, value) if value is not None else None

        elif filter_type == "min":
            value = cls._extract_parameter_value(filter_feature, "value")
            return GreaterThanOrEqual(column_name, value) if value is not None else None

        elif filter_type == "max":
            # Handle both simple and complex max parameters
            if cls._has_parameter(filter_feature, "max"):
                _, max_param, is_max_exclusive = cls.get_min_max_operator(filter_feature)
                if max_param is not None:
                    return (
                        LessThan(column_name, max_param)
                        if is_max_exclusive
                        else LessThanOrEqual(column_name, max_param)
                    )
            else:
                value = cls._extract_parameter_value(filter_feature, "value")
                return LessThanOrEqual(column_name, value) if value is not None else None

        elif filter_type == "range":
            min_param, max_param, is_max_exclusive = cls.get_min_max_operator(filter_feature)
            expressions = []

            if min_param is not None:
                expressions.append(GreaterThanOrEqual(column_name, min_param))

            if max_param is not None:
                max_expr = (
                    LessThan(column_name, max_param) if is_max_exclusive else LessThanOrEqual(column_name, max_param)
                )
                expressions.append(max_expr)  # type: ignore

            if len(expressions) == 1:
                return expressions[0]
            elif len(expressions) == 2 and And is not None:
                return And(expressions[0], expressions[1])

        return None

    @classmethod
    def _extract_parameter_value(cls, filter_feature: SingleFilter, param_name: str) -> Any:
        """Extract a parameter value from filter feature."""
        for param in filter_feature.parameter:
            if param[0] == param_name:
                return param[1]
        return None

    @classmethod
    def _has_parameter(cls, filter_feature: SingleFilter, param_name: str) -> bool:
        """Check if filter feature has a specific parameter."""
        return any(param[0] == param_name for param in filter_feature.parameter)

    # Standard filter methods - not used for Iceberg but required by interface
    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Use apply_filters method for Iceberg filtering")

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Use apply_filters method for Iceberg filtering")

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Use apply_filters method for Iceberg filtering")

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Use apply_filters method for Iceberg filtering")

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Regex filtering is not supported for Iceberg tables")

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Categorical inclusion filtering is not yet implemented for Iceberg tables")

    @classmethod
    def do_custom_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError("Custom filtering is not supported for Iceberg tables")
