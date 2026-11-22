"""check_time_rule: the reference passes, each broken shift fails with its own prefix, cases= narrows the run."""

import functools
from datetime import UTC, datetime, timedelta

import pytest

from mloda._testing.time_rule import (
    SHIFT_CASES,
    Shift,
    assert_time_rule,
    check_time_rule,
)
from mloda.core.abstract_plugins.components.contract.time_rule import DEFAULT_TIME_RULE, TimeRule


pytest_plugins = ["pytester"]


def _adds_hours_on_the_wall_clock(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return value + timedelta(hours=amount) if unit == "hour" else rule.shift(value, amount, unit)


def _returns_utc(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return rule.shift(value, amount, unit).astimezone(UTC)


def _returns_another_instant(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return rule.shift(value, amount, unit) + timedelta(microseconds=1)


def _ignores_the_rule(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return DEFAULT_TIME_RULE.shift(value, amount, unit)


def _reference(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return rule.shift(value, amount, unit)


def _always_raises(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    raise ValueError("always")


def _adds_a_plain_aware_day(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return value + timedelta(days=amount) if unit == "day" else rule.shift(value, amount, unit)


def _returns_naive(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return rule.shift(value, amount, unit).replace(tzinfo=None)


def _failures(shift: Shift) -> tuple[str, ...]:
    return check_time_rule(shift).failures


def test_the_reference_passes() -> None:
    result = check_time_rule()
    assert result.check == "time_rule"
    assert result.failures == ()
    assert result.passed


def test_the_reference_passes_as_an_explicit_shift() -> None:
    result = check_time_rule(_reference)
    assert result.failures == ()
    assert result.subject == "_reference"


def test_a_callable_without_a_name_is_accepted() -> None:
    result = check_time_rule(functools.partial(_reference))
    assert result.failures == ()
    assert result.passed


def _overflows(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    raise OverflowError("too big")


def test_a_shift_that_raises_another_exception_is_reported_not_propagated() -> None:
    case = next(c for c in SHIFT_CASES if c.expected is not None)
    result = check_time_rule(_overflows, cases=[case])
    assert any(f.startswith(f"raises: {case.id}") and "OverflowError" in f for f in result.failures), result.failures


def test_a_shift_that_adds_hours_on_the_wall_clock_fails() -> None:
    result = check_time_rule(_adds_hours_on_the_wall_clock)
    assert not result.passed


def test_a_shift_that_adds_hours_on_the_wall_clock_fails_on_elapsed_time() -> None:
    assert any(f.startswith("elapsed:") for f in _failures(_adds_hours_on_the_wall_clock))


@pytest.mark.parametrize("shift", [_returns_utc, _returns_naive], ids=["utc", "naive"])
def test_a_shift_that_returns_another_zone_fails_on_the_zone(shift: Shift) -> None:
    assert any(f.startswith("zone:") for f in _failures(shift))


def test_a_shift_that_returns_another_instant_fails_on_the_value() -> None:
    assert any(f.startswith("value:") for f in _failures(_returns_another_instant))


def test_a_shift_that_adds_a_plain_aware_day_fails_on_the_wall_time_of_the_gap() -> None:
    failures = _failures(_adds_a_plain_aware_day)
    assert any(f.startswith("wall_time: gap_going_forward") for f in failures), failures


def test_a_shift_that_always_raises_fails_on_raises_where_a_value_is_expected() -> None:
    failures = _failures(_always_raises)
    expecting_a_value = next(c for c in SHIFT_CASES if c.expected is not None)
    assert any(f.startswith(f"raises: {expecting_a_value.id}") for f in failures), failures


def test_a_shift_that_never_raises_fails_on_raises() -> None:
    result = check_time_rule(_ignores_the_rule)
    raising = [c.id for c in SHIFT_CASES if c.expected is None]
    assert raising
    for case_id in raising:
        assert any(f.startswith(f"raises: {case_id}") for f in result.failures), result.failures


def test_a_failure_names_its_case_and_cases_narrows_the_run() -> None:
    assert any(f.startswith(f"value: {SHIFT_CASES[0].id}") for f in _failures(_returns_another_instant))
    day = next(c for c in SHIFT_CASES if c.unit == "day" and c.expected is not None)
    result = check_time_rule(_returns_another_instant, cases=[day])
    assert result.failures
    assert all(day.id in f for f in result.failures), result.failures


def test_assert_time_rule_fails_a_broken_shift() -> None:
    with pytest.raises(pytest.fail.Exception, match="time_rule"):
        assert_time_rule(_returns_another_instant)


INNER = """
from mloda._testing.framework_adapter import PandasAdapter
from mloda.core.abstract_plugins.components.divergence import Cause
from mloda._testing.compute_framework.time_window_suite import TimeWindowRuleSuite
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow


class TimerulefmtOffByOne:
    feature_group = PandasTimeWindow.feature_group

    @classmethod
    def calculate(cls, feature_group, data, features, options):
        result = PandasTimeWindow.calculate(feature_group, data, features, options)
        for name in features.get_all_names():
            result[name] = result[name] + 1
        return result


class TestFaithful(TimeWindowRuleSuite):
    adapter_class = PandasAdapter
    implementation = PandasTimeWindow


class TestBroken(TimeWindowRuleSuite):
    adapter_class = PandasAdapter
    implementation = TimerulefmtOffByOne


class TestSkipping(TimeWindowRuleSuite):
    adapter_class = PandasAdapter
    implementation = PandasTimeWindow
    unsupported_cases = {"dst_short_day_calendar": (Cause.BACKEND, "declared unsupported")}
"""

SELECTED = (
    "dst_short_day_calendar or dst_short_day_elapsed or fixed_length_short_day or gap_raise_lower_bound_in_the_gap"
)


@pytest.mark.parametrize("ignore_timezone", [None, "1"], ids=["unset", "pyarrow_ignore_timezone"])
def test_a_broken_time_window_implementation_fails_the_suite_and_a_faithful_one_passes(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, ignore_timezone: str | None
) -> None:
    if ignore_timezone is None:
        monkeypatch.delenv("PYARROW_IGNORE_TIMEZONE", raising=False)
    else:
        monkeypatch.setenv("PYARROW_IGNORE_TIMEZONE", ignore_timezone)
    pytester.makepyfile(test_inner=INNER)

    result = pytester.runpytest_inprocess(
        "-p", "no:cacheprovider", "-p", "no:randomly", "--tb=short", "-rfs", "-k", SELECTED
    )

    outcomes = result.parseoutcomes()
    assert outcomes.get("passed", 0) > 0
    assert outcomes.get("failed", 0) > 0
    assert outcomes.get("skipped", 0) > 0
    lines = result.stdout.str().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    assert failed
    assert all("TestBroken" in line for line in failed), failed
    assert any("dst_short_day_calendar" in line for line in failed)
    assert any("fixed_length_short_day" in line for line in failed)
    skipped = [line for line in lines if line.startswith("SKIPPED ")]
    assert any("declared unsupported" in line for line in skipped), skipped
