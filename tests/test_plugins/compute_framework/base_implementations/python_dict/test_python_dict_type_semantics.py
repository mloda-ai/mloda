"""Failing tests for the python_dict column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(rows, column)``
in ``mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics``.
Because python_dict carries no schema, the function inspects the first non-null value.
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics import (
    column_semantics,
)


class TestPythonDictColumnSemantics:
    def data(self) -> dict[str, list[Any]]:
        return {
            "ts_naive": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "ts_aware": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
            ],
            "num": [1, 2],
            "s": ["a", "b"],
        }

    def test_ts_naive(self) -> None:
        sem = column_semantics(self.data(), "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_ts_aware(self) -> None:
        sem = column_semantics(self.data(), "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit is None

    def test_num(self) -> None:
        sem = column_semantics(self.data(), "num")
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_s(self) -> None:
        sem = column_semantics(self.data(), "s")
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None


class TestPythonDictIso8601StringColumns:
    """ISO-8601 string columns must be value-inspected and classified as temporal.

    python_dict carries no schema, so a column of ISO-8601 datetime STRINGS looks
    non-temporal under first-value type inference. Value-inspection must classify
    these as temporal with the correct timezone awareness.
    """

    def test_naive_iso_datetime_strings_are_temporal(self) -> None:
        data: dict[str, list[Any]] = {"ts": ["2024-01-01T00:00:00", "2024-06-01T12:30:00"]}
        sem = column_semantics(data, "ts")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_offset_iso_datetime_strings_are_tz_aware(self) -> None:
        data: dict[str, list[Any]] = {"ts": ["2024-01-01T00:00:00+00:00", "2024-06-01T12:30:00+00:00"]}
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_z_suffix_iso_datetime_strings_are_tz_aware(self) -> None:
        data: dict[str, list[Any]] = {"ts": ["2024-01-01T00:00:00Z", "2024-06-01T12:30:00Z"]}
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_plain_strings_stay_non_temporal(self) -> None:
        data: dict[str, list[Any]] = {"s": ["a", "b"]}
        sem = column_semantics(data, "s")
        assert sem.is_temporal is False
        assert sem.is_ordered is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None


class TestPythonDictDatetimeObjectRegression:
    """Real datetime objects must keep their existing schema-based classification."""

    def test_naive_datetime_objects_unchanged(self) -> None:
        data: dict[str, list[Any]] = {"ts": [datetime(2021, 1, 1), datetime(2021, 1, 2)]}
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_tz_aware is False

    def test_aware_datetime_objects_unchanged(self) -> None:
        data: dict[str, list[Any]] = {
            "ts": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
            ]
        }
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_tz_aware is True


class TestPythonDictTimezoneIdentity:
    @pytest.mark.parametrize(
        ("zone", "expected"),
        [
            (UTC, "UTC"),
            (ZoneInfo("Europe/Berlin"), "Europe/Berlin"),
            (timezone(timedelta(hours=5, minutes=30)), "UTC+05:30"),
            (timezone(-timedelta(hours=3, minutes=30)), "UTC-03:30"),
        ],
    )
    def test_native_zone_survives_nulls_and_daylight_saving(self, zone: tzinfo, expected: str) -> None:
        data = {"ts": [None, datetime(2024, 1, 1, tzinfo=zone), None, datetime(2024, 7, 1, tzinfo=zone)]}
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") == expected

    @pytest.mark.parametrize("other_zone", ["America/New_York", "Africa/Lagos"])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_mixed_native_zones_do_not_claim_the_first_zone(self, other_zone: str, reverse: bool) -> None:
        values = [
            datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            datetime(2024, 1, 1, tzinfo=ZoneInfo(other_zone)),
        ]
        sem = column_semantics({"ts": list(reversed(values)) if reverse else values}, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") is None

    @pytest.mark.parametrize(
        "values", [[], [None, None], [datetime(2024, 1, 1)], [date(2024, 1, 1)], [1], ["plain text"]]
    )
    def test_naive_or_untyped_values_have_no_zone(self, values: list[Any]) -> None:
        sem = column_semantics({"ts": values}, "ts")
        assert sem.is_tz_aware is False
        assert getattr(sem, "tz") is None

    @pytest.mark.parametrize(
        ("values", "zone", "aware"),
        [
            ([None, "2024-01-01T00:00:00Z", "2024-06-01T00:00:00+00:00"], "UTC", True),
            ([None, "2024-01-01T00:00:00+05:30", "2024-06-01T00:00:00+05:30"], "UTC+05:30", True),
            (["2024-01-01T00:00:00+01:00", "2024-06-01T00:00:00+02:00"], None, True),
            (["2024-01-01T00:00:00", "2024-06-01T00:00:00"], None, False),
        ],
    )
    def test_iso_string_route_preserves_zone_information(
        self, values: list[str | None], zone: str | None, aware: bool
    ) -> None:
        sem = column_semantics({"ts": values}, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is aware
        assert getattr(sem, "tz") == zone
