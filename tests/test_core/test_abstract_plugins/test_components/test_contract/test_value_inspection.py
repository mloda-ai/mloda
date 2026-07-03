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
from mloda.core.abstract_plugins.components.contract.value_inspection import (
    is_iso8601_string,
    iso8601_string_semantics,
)


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


class TestSeparatorlessFormsRejected:
    """Bare-digit and week/ordinal forms must classify as non-temporal on all Python versions.

    ``datetime.fromisoformat`` is lenient on 3.11+ (accepts separator-less, week and basic
    forms) but strict on 3.10. Requiring a dash-bearing calendar date makes the result
    reproducible across the 3.10-3.14 CI matrix and avoids false-positive temporal
    classification of plain string codes (zip codes, product codes).
    """

    def test_bare_digits_return_none(self) -> None:
        assert iso8601_string_semantics(["20240101"]) is None

    def test_basic_datetime_returns_none(self) -> None:
        assert iso8601_string_semantics(["20240101T000000"]) is None

    def test_week_date_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024-W01-1"]) is None

    def test_ordinal_date_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024001"]) is None

    def test_extended_calendar_forms_still_accepted(self) -> None:
        for value, aware in [
            ("2024-01-01", False),
            ("2024-01-01T00:00:00", False),
            ("2024-01-01T00:00:00+00:00", True),
            ("2024-01-01T00:00:00Z", True),
        ]:
            sem = iso8601_string_semantics([value])
            assert sem is not None, value
            assert sem.is_temporal is True
            assert sem.is_tz_aware is aware


class TestSeparatorRejected:
    """Only 'T' or a single space is a valid ISO-8601 date/time separator.

    fromisoformat is lenient on 3.11+ and accepts an arbitrary single character between the
    date and the time (e.g. '_' or '/'), so such strings would falsely classify as temporal.
    The classifier requires the character at index 10 to be 'T' or ' ' when a time part exists.
    """

    def test_underscore_separator_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01_00:00:00"]) is None

    def test_slash_separator_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01/00:00:00"]) is None

    def test_letter_separator_returns_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01x00:00:00"]) is None

    def test_space_separator_still_accepted(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01 00:00:00"])
        assert sem is not None
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False

    def test_t_separator_still_accepted(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00"])
        assert sem is not None
        assert sem.is_temporal is True


class TestIsIso8601String:
    """``is_iso8601_string`` is a fast single-value ISO-8601 date/datetime probe.

    It uses the same shape rules as the private ``_parse_iso`` classifier: it requires a
    dash-bearing extended calendar date and rejects bare-digit codes, week dates, ordinal
    dates and any non-str input. It lets the sqlite value sampler fast-reject genuine
    string ID join keys after a single probed value, avoiding a full bounded sample query.
    """

    def test_iso_date_is_true(self) -> None:
        assert is_iso8601_string("2024-01-01") is True

    def test_naive_iso_datetime_is_true(self) -> None:
        assert is_iso8601_string("2024-01-01T00:00:00") is True

    def test_offset_iso_datetime_is_true(self) -> None:
        assert is_iso8601_string("2024-01-01T00:00:00+00:00") is True

    def test_trailing_z_iso_datetime_is_true(self) -> None:
        assert is_iso8601_string("2024-01-01T00:00:00Z") is True

    def test_bare_digits_is_false(self) -> None:
        assert is_iso8601_string("20240101") is False

    def test_week_date_is_false(self) -> None:
        assert is_iso8601_string("2024-W01-1") is False

    def test_plain_string_is_false(self) -> None:
        assert is_iso8601_string("hello") is False

    def test_non_string_int_is_false(self) -> None:
        assert is_iso8601_string(5) is False

    def test_none_is_false(self) -> None:
        assert is_iso8601_string(None) is False

    def test_underscore_separator_is_false(self) -> None:
        assert is_iso8601_string("2024-01-01_00:00:00") is False

    def test_slash_separator_is_false(self) -> None:
        assert is_iso8601_string("2024-01-01/00:00:00") is False

    def test_letter_separator_is_false(self) -> None:
        assert is_iso8601_string("2024-01-01x00:00:00") is False

    def test_space_separator_is_true(self) -> None:
        assert is_iso8601_string("2024-01-01 00:00:00") is True


class TestMixedTimezoneAwareness:
    """A sample mixing tz-aware and tz-naive/date-only values is not confidently classifiable."""

    def test_aware_and_naive_return_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01T00:00:00+00:00", "2024-06-01T00:00:00"]) is None

    def test_aware_and_date_only_return_none(self) -> None:
        assert iso8601_string_semantics(["2024-01-01T00:00:00Z", "2024-06-01"]) is None

    def test_all_aware_is_tz_aware_true(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00+00:00", "2024-06-01T00:00:00Z"])
        assert sem is not None
        assert sem.is_tz_aware is True

    def test_all_naive_is_tz_aware_false(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01T00:00:00", "2024-06-01T12:30:00"])
        assert sem is not None
        assert sem.is_tz_aware is False

    def test_all_date_only_is_tz_aware_false(self) -> None:
        sem = iso8601_string_semantics(["2024-01-01", "2024-06-01"])
        assert sem is not None
        assert sem.is_tz_aware is False
