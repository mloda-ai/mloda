"""Rules deciding whether a Python value fits a declared DataType."""

import datetime
import decimal
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.result import CheckResult

_FLOAT32_OVERFLOW = float(2**128 - 2**103)
_DECIMAL_SCALE = 18
_DECIMAL_INTEGER_DIGITS = 20


def _int_in(bits: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2 ** (bits - 1)) <= v < 2 ** (bits - 1)


def _is_float32(v: Any) -> bool:
    return isinstance(v, float) and (not math.isfinite(v) or abs(v) < _FLOAT32_OVERFLOW)


# Declared timestamp types carry no time zone, so only naive datetimes fit.
def _is_naive_datetime(v: Any) -> bool:
    return isinstance(v, datetime.datetime) and v.tzinfo is None


def _is_millis(v: Any) -> bool:
    return _is_naive_datetime(v) and v.microsecond % 1000 == 0


def _is_date(v: Any) -> bool:
    return isinstance(v, datetime.date) and not isinstance(v, datetime.datetime)


def _is_decimal(v: Any) -> bool:
    if not isinstance(v, decimal.Decimal) or not v.is_finite():
        return False
    if v.adjusted() >= _DECIMAL_INTEGER_DIGITS:
        return False
    with decimal.localcontext() as ctx:
        ctx.prec = _DECIMAL_INTEGER_DIGITS + _DECIMAL_SCALE + 10
        ctx.traps[decimal.Inexact] = False
        ctx.traps[decimal.Rounded] = False
        return v == v.quantize(decimal.Decimal(1).scaleb(-_DECIMAL_SCALE))


_RULES: dict[DataType, Callable[[Any], bool]] = {
    DataType.INT32: _int_in(32),
    DataType.INT64: _int_in(64),
    DataType.FLOAT: _is_float32,
    DataType.DOUBLE: lambda v: isinstance(v, float),
    DataType.BOOLEAN: lambda v: isinstance(v, bool),
    DataType.STRING: lambda v: isinstance(v, str),
    DataType.BINARY: lambda v: isinstance(v, bytes),
    DataType.DATE: _is_date,
    DataType.TIMESTAMP_MILLIS: _is_millis,
    DataType.TIMESTAMP_MICROS: _is_naive_datetime,
    DataType.DECIMAL: _is_decimal,
}


def value_fits(data_type: DataType, value: Any) -> bool:
    """Return whether value fits data_type; None fits every type."""
    rule = _RULES.get(data_type)
    if rule is None:
        raise ValueError(f"No value-fit rule for DataType.{data_type.name}")
    return value is None or rule(value)


def check_values_fit(columns: Mapping[str, Sequence[Any]], types: Mapping[str, DataType]) -> CheckResult:
    """Check every value of every column against its declared type."""
    failures: list[str] = []
    for name, values in columns.items():
        data_type = types.get(name)
        if data_type is None:
            failures.append(f"column '{name}' has no declared type")
            continue
        for index, value in enumerate(values):
            if not value_fits(data_type, value):
                failures.append(
                    f"column '{name}' row {index}: {type(value).__name__} value {value!r} does not fit {data_type.name}"
                )
    failures.extend(f"declared column '{name}' has no data" for name in types if name not in columns)
    return CheckResult("values_fit", "columns", tuple(failures))
