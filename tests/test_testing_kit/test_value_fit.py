"""Tests for the DataType value-fit rules and check_values_fit."""

import datetime
import decimal
from decimal import Decimal
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.value_fit import _RULES, check_values_fit, value_fits

UTC = datetime.UTC

FITS: list[tuple[DataType, Any]] = [
    (DataType.INT32, 5),
    (DataType.INT32, 2**31 - 1),
    (DataType.INT32, -(2**31)),
    (DataType.INT64, 2**31),
    (DataType.INT64, 2**63 - 1),
    (DataType.INT64, -(2**63)),
    (DataType.FLOAT, 1.5),
    (DataType.FLOAT, float("inf")),
    (DataType.FLOAT, float("nan")),
    (DataType.DOUBLE, 1e300),
    (DataType.BOOLEAN, True),
    (DataType.STRING, "a"),
    (DataType.BINARY, b"a"),
    (DataType.DATE, datetime.date(2024, 1, 1)),
    (DataType.TIMESTAMP_MILLIS, datetime.datetime(2024, 1, 1, 0, 0, 0, 5000)),
    (DataType.TIMESTAMP_MICROS, datetime.datetime(2024, 1, 1, 0, 0, 0, 5)),
    (DataType.DECIMAL, Decimal("1.5")),
    (DataType.DECIMAL, Decimal("1.000000000000000000000")),
    (DataType.DECIMAL, Decimal("0.000000000000000001")),
    (DataType.DECIMAL, Decimal("99999999999999999999.999999999999999999")),
    (DataType.DECIMAL, Decimal("-99999999999999999999.999999999999999999")),
    (DataType.FLOAT, 3.4028235e38),
]

MISFITS: list[tuple[DataType, Any]] = [
    (DataType.INT32, True),
    (DataType.INT32, 2**31),
    (DataType.INT32, -(2**31) - 1),
    (DataType.INT32, 1.0),
    (DataType.INT64, True),
    (DataType.INT64, 2**63),
    (DataType.INT64, -(2**63) - 1),
    (DataType.FLOAT, 1),
    (DataType.FLOAT, 1e39),
    (DataType.FLOAT, float(2**128 - 2**103)),
    (DataType.DOUBLE, 1),
    (DataType.BOOLEAN, 1),
    (DataType.STRING, b"a"),
    (DataType.BINARY, "a"),
    (DataType.DATE, datetime.datetime(2024, 1, 1)),
    (DataType.TIMESTAMP_MILLIS, datetime.datetime(2024, 1, 1, 0, 0, 0, 5)),
    (DataType.TIMESTAMP_MILLIS, datetime.datetime(2024, 1, 1, tzinfo=UTC)),
    (DataType.TIMESTAMP_MICROS, datetime.datetime(2024, 1, 1, tzinfo=UTC)),
    (DataType.TIMESTAMP_MICROS, datetime.date(2024, 1, 1)),
    (DataType.DECIMAL, Decimal("1e-19")),
    (DataType.DECIMAL, Decimal(10**20)),
    (DataType.DECIMAL, Decimal("NaN")),
    (DataType.DECIMAL, 1.5),
]


@pytest.mark.parametrize("data_type, value", FITS)
def test_value_fits(data_type: DataType, value: Any) -> None:
    assert value_fits(data_type, value) is True


@pytest.mark.parametrize("data_type, value", MISFITS)
def test_value_misfits(data_type: DataType, value: Any) -> None:
    assert value_fits(data_type, value) is False


@pytest.mark.parametrize("data_type", list(DataType))
def test_null_fits_every_member(data_type: DataType) -> None:
    assert value_fits(data_type, None) is True


@pytest.mark.parametrize("data_type", list(DataType))
def test_every_member_has_a_rule(data_type: DataType) -> None:
    assert value_fits(data_type, object()) is False


def test_member_without_rule_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(_RULES, DataType.DATE)
    with pytest.raises(ValueError, match="DATE"):
        value_fits(DataType.DATE, datetime.date(2024, 1, 1))
    assert value_fits(DataType.INT32, 1) is True


def test_check_values_fit_passes() -> None:
    result = check_values_fit({"a": [1, None], "b": ["x", "y"]}, {"a": DataType.INT64, "b": DataType.STRING})
    assert result.check == "values_fit"
    assert result.passed is True
    assert result.failures == ()


def test_check_values_fit_reports_each_misfit() -> None:
    result = check_values_fit({"a": [7, "x", 8, "y"]}, {"a": DataType.INT64})
    assert result.passed is False
    assert len(result.failures) == 2
    for part in ("a", "row 1", "str", "INT64"):
        assert part in result.failures[0]
    assert "row 3" in result.failures[1]


@pytest.mark.parametrize(
    "columns, types, name",
    [
        ({"a": [1], "extra": [1]}, {"a": DataType.INT64}, "extra"),
        ({"a": [1]}, {"a": DataType.INT64, "gone": DataType.STRING}, "gone"),
    ],
)
def test_check_values_fit_column_mismatch(columns: dict[str, list[Any]], types: dict[str, DataType], name: str) -> None:
    result = check_values_fit(columns, types)
    assert result.passed is False
    assert len(result.failures) == 1
    assert name in result.failures[0]


def test_decimal_fit_ignores_caller_context_traps() -> None:
    with decimal.localcontext() as ctx:
        ctx.traps[decimal.Inexact] = True
        ctx.traps[decimal.Rounded] = True
        assert value_fits(DataType.DECIMAL, Decimal("99999999999999999999.999999999999999999")) is True
        assert value_fits(DataType.DECIMAL, Decimal("1e-19")) is False
