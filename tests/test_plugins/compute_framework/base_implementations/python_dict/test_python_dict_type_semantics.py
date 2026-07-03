"""Failing tests for the python_dict column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(rows, column)``
in ``mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics``.
Because python_dict carries no schema, the function inspects the first non-null value.
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

from datetime import datetime, timezone
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics import (
    column_semantics,
)


class TestPythonDictColumnSemantics:
    def data(self) -> dict[str, list[Any]]:
        return {
            "ts_naive": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "ts_aware": [
                datetime(2021, 1, 1, tzinfo=timezone.utc),
                datetime(2021, 1, 2, tzinfo=timezone.utc),
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
                datetime(2021, 1, 1, tzinfo=timezone.utc),
                datetime(2021, 1, 2, tzinfo=timezone.utc),
            ]
        }
        sem = column_semantics(data, "ts")
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_tz_aware is True
