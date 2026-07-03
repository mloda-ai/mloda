"""Failing tests for the ISO-8601 value-inspection helper (epic #518, follow-up).

Defines the not-yet-implemented ``iso8601_string_semantics`` helper in
``mloda.core.abstract_plugins.components.contract.value_inspection``. On
dynamically typed backends (e.g. sqlite storing datetimes as ISO TEXT), a
column may look like plain strings under schema-only introspection. This
helper inspects a bounded sample of the actual values and, when they are all
parseable ISO-8601 date/datetime strings, returns temporal semantics with the
correct timezone awareness. Otherwise it returns ``None`` so the caller keeps
its existing schema-based semantics.

Expected to fail at import time (ModuleNotFoundError) until Green creates the
module.
"""

from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.contract.value_inspection import iso8601_string_semantics


class TestReturnsNone:
    """The helper returns None when it cannot confidently classify a column."""

    def test_empty_iterable_returns_none(self) -> None:
        assert iso8601_string_semantics([]) is None

    def test_all_none_returns_none(self) -> None:
        assert iso8601_string_semantics([None, None]) is None

    def test_plain_strings_return_none(self) -> None:
        assert iso8601_string_semantics(["hello", "world"]) is None

    def test_any_non_iso_value_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01T00:00:00", "not-a-date"]) is None

    def test_non_string_values_return_none(self) -> None:
        assert iso8601_string_semantics([5]) is None


class TestNaiveDatetimeStrings:
    """All-ISO naive datetime strings classify as temporal, ordered, tz-naive."""

    def test_naive_datetime_semantics(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00", "2024-06-01T12:30:00"])
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_numeric is False
        assert sem.unit is None
        assert sem.is_tz_aware is False


class TestTimezoneAwareness:
    """Timezone awareness is derived from the parsed ISO strings."""

    def test_offset_is_tz_aware(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00+00:00"])
        assert sem is not None
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_trailing_z_is_tz_aware(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00Z"])
        assert sem is not None
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_date_only_is_temporal_and_naive(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01", "2024-02-01"])
        assert sem is not None
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False


class TestNoneSkipping:
    """None values are skipped when sampling."""

    def test_none_values_are_skipped(self) -> None:
        values: list[Any] = ["2024-01-01T00:00:00", None, "2024-02-01T00:00:00"]
        sem = iso8601_string_semantics(values)
        assert sem is not None
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False
