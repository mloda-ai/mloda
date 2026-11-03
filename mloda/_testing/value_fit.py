"""Type-strict value-fit rules per DataType, stricter than Arrow wherever Arrow coerces."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from mloda._testing.result import CheckResult
from mloda.core.abstract_plugins.components.data_types import DataType

_FLOAT32_OVERFLOW = 2**128 - 2**103


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _int_range(low: int, high: int) -> Callable[[Any], bool]:
    return lambda v: _is_int(v) and low <= v <= high


def _float_fits(v: Any) -> bool:
    if _is_int(v):
        return bool(abs(v) <= 2**24)
    if isinstance(v, float):
        return v != v or abs(v) == float("inf") or abs(int(v)) < _FLOAT32_OVERFLOW
    return False


def _double_fits(v: Any) -> bool:
    if _is_int(v):
        return bool(abs(v) <= 2**53)
    return isinstance(v, float)


def _naive(v: Any) -> bool:
    return isinstance(v, datetime) and v.utcoffset() is None


def _aware(v: Any) -> bool:
    return isinstance(v, datetime) and v.utcoffset() is not None


def _decimal_fits(v: Any) -> bool:
    if not isinstance(v, Decimal) or not v.is_finite():
        return False
    _, digits, exponent = v.as_tuple()
    if not isinstance(exponent, int):
        return False
    fractional = max(-exponent, 0)
    integer = max(len(digits) + exponent, 0)
    return fractional <= 18 and integer <= 20


_RULES: dict[DataType, Callable[[Any], bool]] = {
    DataType.INT8: _int_range(-(2**7), 2**7 - 1),
    DataType.INT32: _int_range(-(2**31), 2**31 - 1),
    DataType.INT64: _int_range(-(2**63), 2**63 - 1),
    DataType.UINT8: _int_range(0, 2**8 - 1),
    DataType.UINT64: _int_range(0, 2**64 - 1),
    DataType.FLOAT: _float_fits,
    DataType.DOUBLE: _double_fits,
    DataType.BOOLEAN: lambda v: isinstance(v, bool),
    DataType.STRING: lambda v: isinstance(v, str),
    DataType.BINARY: lambda v: isinstance(v, bytes),
    DataType.DATE: lambda v: isinstance(v, date) and not isinstance(v, datetime),
    DataType.TIMESTAMP_MILLIS: lambda v: _naive(v) and v.microsecond % 1000 == 0,
    DataType.TIMESTAMP_MICROS: _naive,
    DataType.TIMESTAMP_MICROS_TZ: _aware,
    DataType.DECIMAL: _decimal_fits,
    DataType.LIST: lambda v: isinstance(v, list),
}


def value_fits(data_type: DataType, value: Any) -> bool:
    """Whether value fits data_type; None fits every member."""
    if data_type not in _RULES:
        raise ValueError(f"No value-fit rule for data type {data_type.name}")
    return value is None or _RULES[data_type](value)


def check_values_fit(columns: Mapping[str, Sequence[Any]], types: Mapping[str, DataType]) -> CheckResult:
    """One failure per misfit value, undeclared column and column-less declared type."""
    failures: list[str] = []
    for name, values in columns.items():
        if name not in types:
            failures.append(f"column '{name}' has no declared type")
            continue
        for row, value in enumerate(values):
            if not value_fits(types[name], value):
                failures.append(f"column '{name}' row {row}: value {value!r} does not fit {types[name].name}")
    for name in types:
        if name not in columns:
            failures.append(f"declared type for column '{name}' has no column")
    return CheckResult(check="values_fit", subject=", ".join(columns), failures=tuple(failures))
