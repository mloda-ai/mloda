"""Value-fit rules: one table of (member, value, fits) plus the check_values_fit messages."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from mloda._testing import value_fit
from mloda._testing.result import CheckResult
from mloda._testing.value_fit import check_values_fit, value_fits
from mloda.core.abstract_plugins.components.data_types import DataType

_NAIVE = datetime(2024, 1, 2, 3, 4, 5)
_AWARE = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

_NUMERIC = [
    DataType.INT8,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT64,
    DataType.FLOAT,
    DataType.DOUBLE,
]

_FIT_CASES: list[tuple[DataType, Any, bool]] = [
    # integer boundaries, both sides
    (DataType.INT8, 127, True),
    (DataType.INT8, -128, True),
    (DataType.INT8, 128, False),
    (DataType.INT8, -129, False),
    (DataType.INT8, 300, False),
    (DataType.INT32, 2**31 - 1, True),
    (DataType.INT32, -(2**31), True),
    (DataType.INT32, 2**31, False),
    (DataType.INT32, -(2**31) - 1, False),
    (DataType.INT64, 2**63 - 1, True),
    (DataType.INT64, -(2**63), True),
    (DataType.INT64, 2**63, False),
    (DataType.INT64, -(2**63) - 1, False),
    (DataType.UINT8, 0, True),
    (DataType.UINT8, 255, True),
    (DataType.UINT8, -1, False),
    (DataType.UINT8, 256, False),
    (DataType.UINT64, 0, True),
    (DataType.UINT64, 2**64 - 1, True),
    (DataType.UINT64, 2**64, False),
    (DataType.UINT64, -1, False),
    # type strictness
    (DataType.INT64, "x", False),
    (DataType.INT64, 75.0, False),
    (DataType.INT8, 1.0, False),
    (DataType.UINT64, 1.0, False),
    (DataType.STRING, 1, False),
    (DataType.BINARY, "x", False),
    (DataType.BOOLEAN, 1, False),
    # float and double
    (DataType.FLOAT, 2**24, True),
    (DataType.FLOAT, 2**24 + 1, False),
    (DataType.FLOAT, 1.5, True),
    (DataType.FLOAT, 1e38, True),
    (DataType.FLOAT, 1e39, False),
    (DataType.FLOAT, float("inf"), True),
    (DataType.FLOAT, float("nan"), True),
    (DataType.DOUBLE, 2**53, True),
    (DataType.DOUBLE, 2**53 + 1, False),
    (DataType.DOUBLE, 1e300, True),
    (DataType.DOUBLE, "1.0", False),
    # simple types
    (DataType.BOOLEAN, True, True),
    (DataType.STRING, "x", True),
    (DataType.BINARY, b"x", True),
    (DataType.DATE, date(2024, 1, 2), True),
    (DataType.DATE, _NAIVE, False),
    (DataType.DATE, "2024-01-02", False),
    # timestamps
    (DataType.TIMESTAMP_MILLIS, _NAIVE, True),
    (DataType.TIMESTAMP_MILLIS, _NAIVE.replace(microsecond=1000), True),
    (DataType.TIMESTAMP_MILLIS, _NAIVE.replace(microsecond=1), False),
    (DataType.TIMESTAMP_MILLIS, _AWARE, False),
    (DataType.TIMESTAMP_MICROS, _NAIVE.replace(microsecond=1), True),
    (DataType.TIMESTAMP_MICROS, _AWARE, False),
    (DataType.TIMESTAMP_MICROS_TZ, _AWARE, True),
    (DataType.TIMESTAMP_MICROS_TZ, _NAIVE, False),
    # decimal(38, 18)
    (DataType.DECIMAL, Decimal("1" * 20), True),
    (DataType.DECIMAL, Decimal("1" * 21), False),
    (DataType.DECIMAL, Decimal("0." + "1" * 18), True),
    (DataType.DECIMAL, Decimal("0." + "1" * 19), False),
    (DataType.DECIMAL, Decimal("NaN"), False),
    (DataType.DECIMAL, 1.5, False),
    # list
    (DataType.LIST, [1, 2], True),
    (DataType.LIST, [], True),
    (DataType.LIST, (1, 2), False),
    # exact-integer and range edges
    (DataType.FLOAT, -(2**24), True),
    (DataType.FLOAT, -(2**24) - 1, False),
    (DataType.DOUBLE, -(2**53), True),
    (DataType.DOUBLE, -(2**53) - 1, False),
    (DataType.FLOAT, 3.4028234663852886e38, True),
    (DataType.FLOAT, 3.4028235677973366e38, False),
    (DataType.FLOAT, -3.4028235677973366e38, False),
    (DataType.DECIMAL, Decimal("1E+20"), False),
    (DataType.DECIMAL, Decimal("Infinity"), False),
    (DataType.DECIMAL, Decimal("-1.5"), True),
]
_FIT_CASES += [(member, True, member is DataType.BOOLEAN) for member in _NUMERIC + [DataType.BOOLEAN]]
_FIT_CASES += [(member, None, True) for member in DataType]


@pytest.mark.parametrize("data_type, value, fits", _FIT_CASES)
def test_value_fits(data_type: DataType, value: Any, fits: bool) -> None:
    assert value_fits(data_type, value) is fits


def test_member_without_a_rule_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(value_fit._RULES, DataType.LIST)

    with pytest.raises(ValueError, match="No value-fit rule"):
        value_fits(DataType.LIST, None)


def test_check_values_fit_passes_and_names_the_check() -> None:
    result = check_values_fit({"a": [1, None]}, {"a": DataType.INT8})
    assert isinstance(result, CheckResult)
    assert result.passed
    assert result.check == "values_fit"
    assert result.failures == ()


def test_check_values_fit_reports_misfit_undeclared_and_missing() -> None:
    result = check_values_fit({"a": [1, 300], "extra": [1]}, {"a": DataType.INT8, "gone": DataType.STRING})

    assert not result.passed
    assert len(result.failures) == 3
    assert "column 'a' row 1: value 300 does not fit INT8" in result.failures
    assert any("extra" in f for f in result.failures)
    assert any("gone" in f for f in result.failures)
    assert all("_testing" not in f for f in result.failures)
