"""Tests for TimeRule: pinned shift cases, knobs, option round trip, fixed spans and nanosecond shifts."""

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from mloda._testing.time_rule import SHIFT_CASES, WINDOW_CASES, ShiftCase, WindowCase, assert_time_rule
from mloda.core.abstract_plugins.components.contract.time_rule import DEFAULT_TIME_RULE, TimeRule

NY = ZoneInfo("America/New_York")
IST = timezone(timedelta(hours=5, minutes=30))
DAY_NS = 86_400_000_000_000
FIXED = TimeRule(calendar_units="fixed_length")


@pytest.mark.parametrize("case", SHIFT_CASES, ids=[c.id for c in SHIFT_CASES])
def test_the_reference_gives_every_pinned_shift_case(case: ShiftCase) -> None:
    assert_time_rule(cases=[case])


@pytest.mark.parametrize("cases", [SHIFT_CASES, WINDOW_CASES], ids=["shift", "window"])
def test_case_ids_are_unique(cases: tuple[Any, ...]) -> None:
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", WINDOW_CASES, ids=[c.id for c in WINDOW_CASES])
def test_a_window_case_is_well_shaped(case: WindowCase) -> None:
    assert list(case.instants) == sorted(case.instants)
    if case.members is not None:
        assert len(case.members) == len(case.instants)


@pytest.mark.parametrize(
    ("knob", "value"),
    [(knob, value) for knob, values in TimeRule.KNOBS.items() for value in values[1:]],
)
def test_every_non_default_knob_value_has_a_pinned_case(knob: str, value: str) -> None:
    assert any(getattr(case.rule, knob) == value for case in SHIFT_CASES)


@pytest.mark.parametrize("knob", sorted(TimeRule.KNOBS))
def test_the_field_default_is_the_first_value_of_the_knob(knob: str) -> None:
    assert getattr(DEFAULT_TIME_RULE, knob) == TimeRule.KNOBS[knob][0]
    assert getattr(TimeRule(), knob) == TimeRule.KNOBS[knob][0]


@pytest.mark.parametrize("knob", sorted(TimeRule.KNOBS))
def test_an_unknown_knob_value_names_the_knob_and_the_allowed_values(knob: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        TimeRule(**{knob: "sideways"})
    message = str(exc_info.value)
    assert knob in message
    assert all(allowed in message for allowed in TimeRule.KNOBS[knob])


@pytest.mark.parametrize(
    "call",
    [
        lambda: DEFAULT_TIME_RULE.shift(datetime(2023, 1, 1), 1, "fortnight"),
        lambda: DEFAULT_TIME_RULE.clock("fortnight"),
        lambda: DEFAULT_TIME_RULE.fixed_ns(1, "fortnight", None),
    ],
    ids=["shift", "clock", "fixed_ns"],
)
def test_an_unknown_unit_is_rejected(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="Unsupported time unit"):
        call()


@pytest.mark.parametrize(
    ("rule", "unit", "expected"),
    [
        (FIXED, "second", "elapsed"),
        (FIXED, "minute", "elapsed"),
        (DEFAULT_TIME_RULE, "second", "elapsed"),
        (DEFAULT_TIME_RULE, "minute", "elapsed"),
        (DEFAULT_TIME_RULE, "hour", "elapsed"),
        (DEFAULT_TIME_RULE, "day", "calendar"),
        (DEFAULT_TIME_RULE, "week", "calendar"),
        (DEFAULT_TIME_RULE, "month", "calendar"),
        (DEFAULT_TIME_RULE, "year", "calendar"),
        (FIXED, "hour", "elapsed"),
        (FIXED, "day", "elapsed"),
        (FIXED, "week", "elapsed"),
        (FIXED, "month", "elapsed"),
        (FIXED, "year", "elapsed"),
    ],
)
def test_clock_of_each_unit(rule: TimeRule, unit: str, expected: str) -> None:
    assert rule.clock(unit) == expected


@pytest.mark.parametrize("value", [None, {}])
def test_from_option_gives_the_default_for_nothing(value: object) -> None:
    assert TimeRule.from_option(value) == DEFAULT_TIME_RULE


def test_from_option_reads_a_knob_mapping() -> None:
    assert TimeRule.from_option({"gap": "raise", "repeated": "later"}) == TimeRule(gap="raise", repeated="later")


@pytest.mark.parametrize(
    "value",
    [{"nonsense": "x"}, {"gap": "raise", "nonsense": "x"}, "fixed_length", ["gap"], 3, TimeRule(gap="raise"), {1: "x"}],
    ids=["unknown_knob", "unknown_knob_beside_a_good_one", "string", "list", "int", "rule_instance", "non_string_key"],
)
def test_from_option_rejects_what_is_not_a_knob_mapping(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        TimeRule.from_option(value)
    assert all(knob in str(exc_info.value) for knob in TimeRule.KNOBS)


def test_from_option_rejects_an_unknown_knob_value() -> None:
    with pytest.raises(ValueError, match="gap"):
        TimeRule.from_option({"gap": "sideways"})


def test_the_default_rule_has_no_option() -> None:
    assert DEFAULT_TIME_RULE.as_option() == {}


def test_as_option_holds_only_the_non_default_knobs() -> None:
    assert TimeRule(gap="raise", repeated="source", month_end="raise").as_option() == {
        "gap": "raise",
        "month_end": "raise",
    }


@pytest.mark.parametrize(
    "rule",
    [
        DEFAULT_TIME_RULE,
        FIXED,
        TimeRule(gap="raise"),
        TimeRule(repeated="later", month_end="raise"),
        TimeRule("fixed_length", "raise", "raise", "raise"),
    ],
)
def test_from_option_inverts_as_option(rule: TimeRule) -> None:
    assert TimeRule.from_option(rule.as_option()) == rule


@pytest.mark.parametrize(
    ("rule", "size", "unit", "zone", "expected"),
    [
        (DEFAULT_TIME_RULE, 2, "hour", NY, 7_200_000_000_000),
        (DEFAULT_TIME_RULE, 1, "day", None, DAY_NS),
        (DEFAULT_TIME_RULE, 1, "week", IST, 7 * DAY_NS),
        (DEFAULT_TIME_RULE, 1, "day", ZoneInfo("Etc/GMT-5"), DAY_NS),
        (DEFAULT_TIME_RULE, 1, "day", NY, None),
        (DEFAULT_TIME_RULE, 1, "month", None, None),
        (DEFAULT_TIME_RULE, 1, "week", NY, None),
        (DEFAULT_TIME_RULE, 1, "year", UTC, None),
        (FIXED, 1, "day", NY, DAY_NS),
        (FIXED, 2, "week", NY, 14 * DAY_NS),
        (FIXED, 1, "month", NY, 30 * DAY_NS),
        (FIXED, 1, "month", None, 2_592_000_000_000_000),
        (FIXED, 1, "year", UTC, 365 * DAY_NS),
        (FIXED, 1, "year", NY, 31_536_000_000_000_000),
    ],
)
def test_fixed_ns(rule: TimeRule, size: int, unit: str, zone: tzinfo | None, expected: int | None) -> None:
    assert rule.fixed_ns(size, unit, zone) == expected


@pytest.mark.parametrize(
    ("zone", "expected"),
    [
        (None, True),
        (UTC, True),
        (ZoneInfo("UTC"), True),
        (IST, True),
        (ZoneInfo("Etc/GMT-5"), True),
        (NY, False),
    ],
)
def test_a_day_is_constant_only_without_daylight_saving_time(zone: tzinfo | None, expected: bool) -> None:
    assert (DEFAULT_TIME_RULE.fixed_ns(1, "day", zone) is not None) is expected


def _ns(text: str) -> int:
    seconds = int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())
    return seconds * 1_000_000_000


def test_shift_ns_aware_column_uses_its_zone() -> None:
    assert DEFAULT_TIME_RULE.shift_ns(_ns("2023-03-11T17:00:00") + 1, 1, "day", NY) == _ns("2023-03-12T16:00:00") + 1


def test_shift_ns_naive_column_keeps_the_sub_microsecond_remainder() -> None:
    expected = _ns("2023-02-28T12:00:00") + 999
    assert DEFAULT_TIME_RULE.shift_ns(_ns("2023-01-31T12:00:00") + 999, 1, "month", None) == expected


def test_shift_ns_before_1970() -> None:
    assert DEFAULT_TIME_RULE.shift_ns(-1, -1, "day", None) == -1 - DAY_NS


def test_shift_ns_follows_the_rule() -> None:
    assert FIXED.shift_ns(_ns("2023-03-11T17:00:00"), 1, "day", NY) == _ns("2023-03-12T17:00:00")


DOCS = Path(__file__).resolve().parents[5] / "docs" / "docs" / "in_depth" / "timezone-computation.md"


def _table_rows(header: str) -> list[list[str]]:
    """The body rows of the docs table whose header row starts with header, as stripped cells."""
    lines = DOCS.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _backticked(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def test_the_docs_unit_table_equals_the_code() -> None:
    clocks = {"elapsed time": "elapsed", "wall clock": "calendar"}
    documented: dict[str, str] = {}
    for cells in _table_rows("| Unit | Clock | Adding one |"):
        clock = next(value for prefix, value in clocks.items() if cells[1].startswith(prefix))
        for unit in _backticked(cells[0]):
            assert unit not in documented, f"unit {unit} appears twice"
            documented[unit] = clock
    assert documented == {unit: DEFAULT_TIME_RULE.clock(unit) for unit in TimeRule.UNITS}


def test_the_docs_knob_table_equals_the_code() -> None:
    documented = {
        _backticked(cells[0])[0]: tuple(_backticked(cells[1]))
        for cells in _table_rows("| Knob | Values (default first) | Meaning |")
    }
    assert documented == TimeRule.KNOBS
