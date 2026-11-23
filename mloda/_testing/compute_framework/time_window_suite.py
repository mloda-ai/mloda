"""Conformance suite: a time window implementation follows the time zone computation rule on every framework."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.time_rule import WINDOW_CASES, WINDOW_FUNCTIONS, WindowCase
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require
from mloda.provider import DefaultOptionKeys, FeatureSet, TimeRule
from mloda.user import DataType, Feature, Options

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["TimeWindowRuleSuite"]


_CASES_BY_ID = {case.id: case for case in WINDOW_CASES}
_REFERENCE_TIME = str(DefaultOptionKeys.reference_time)


def _zone_of(name: str) -> tzinfo:
    if name[0] in "+-":
        sign = -1 if name[0] == "-" else 1
        hours, minutes = name[1:].split(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
    return ZoneInfo(name)


def _expected(function: str, members: tuple[int, ...]) -> float | None:
    values = [float(v) for v in members]
    if function in ("std", "var"):
        if len(values) < 2:
            return None
        return statistics.stdev(values) if function == "std" else statistics.variance(values)
    reducers: dict[str, Callable[[list[float]], float]] = {
        "sum": sum,
        "min": min,
        "max": max,
        "avg": statistics.fmean,
        "mean": statistics.fmean,
        "count": lambda v: float(len(v)),
        "median": statistics.median,
        "first": lambda v: v[0],
        "last": lambda v: v[-1],
    }
    return reducers[function](values)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class TimeWindowRuleSuite(ComputeFrameworkSuite, suite_base=True):
    """Every window case and function against a time window implementation; every available framework is served."""

    aspect = "time window rule"
    required_hooks = ("implementation",)
    case_ids = frozenset(_CASES_BY_ID)
    implementation: ClassVar[Any]

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        case = _CASES_BY_ID.get(name)
        if case is not None and case.zone is not None:
            return {_REFERENCE_TIME: require("pyarrow", "building Arrow test tables").timestamp("us", tz=case.zone)}
        return super().arrow_types(name)

    @staticmethod
    def _data(case: WindowCase, typed_columns: bool) -> TypedTestData:
        if case.zone is None:
            times = [datetime.fromisoformat(i) for i in case.instants]
            time_type = DataType.TIMESTAMP_MICROS
        else:
            times = [datetime.fromisoformat(i).replace(tzinfo=UTC) for i in case.instants]
            if not typed_columns:
                zone = _zone_of(case.zone)
                times = [t.astimezone(zone) for t in times]
            time_type = DataType.TIMESTAMP_MICROS_TZ
        values = [10**i for i in range(len(times))]
        return TypedTestData({"value": (DataType.INT64, values), _REFERENCE_TIME: (time_type, times)})

    @pytest.mark.parametrize("function", WINDOW_FUNCTIONS)
    @pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in WINDOW_CASES])
    def test_window_follows_the_time_rule(self, adapter: FrameworkAdapter, case: WindowCase, function: str) -> None:
        self.skip_unsupported_case(case.id)
        # UTC values: PYARROW_IGNORE_TIMEZONE drops tzinfo, and a UTC wall time cannot move; arrow_types pins the zone.
        frame = self.frame(adapter, case.id, self._data(case, adapter.capabilities.typed_columns))
        name = f"value__{function}_{case.size}_{case.unit}_window"
        rule_option = case.rule.as_option()
        options = Options(group={TimeRule.OPTION_KEY: rule_option}) if rule_option else Options()
        features = FeatureSet()
        features.add(Feature(name, options))
        implementation = self.implementation
        if case.members is None:
            with pytest.raises(ValueError):
                implementation.calculate(implementation.feature_group, frame, features, None)
            return
        result = implementation.calculate(implementation.feature_group, frame, features, None)
        actual = self.column_values(adapter, result, name)
        expected = [_expected(function, members) for members in case.members]
        assert len(actual) == len(expected)
        for row, (got, want) in enumerate(zip(actual, expected)):
            if want is None:
                assert _missing(got), f"{case.id} {function} row {row}: expected missing, got {got!r}"
            else:
                assert not _missing(got) and got == pytest.approx(want), (
                    f"{case.id} {function} row {row}: expected {want}, got {got!r}"
                )
