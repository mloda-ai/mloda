"""Plugin tests collect failed rows through public entry points."""

import pytest

from mloda import steward
from mloda.provider import FailedRows, instrument, report_failed_rows
from mloda.steward import ExtenderHook, HookContext


def _context(feature_group: str) -> HookContext:
    return HookContext(
        hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        feature_group_class=feature_group,
        feature_group_version=None,
        compute_framework_name=None,
        feature_names=("value",),
    )


@pytest.mark.parametrize("raises", [False, True], ids=["success", "exception"])
def test_public_collector_populates_manual_context_and_does_not_leak(raises: bool) -> None:
    from mloda.steward import collect_failed_rows

    assert "collect_failed_rows" in steward.__all__
    context = _context("ManualPlugin")
    failure = ValueError("synthetic calculation failure")

    def calculate() -> list[int | None]:
        report_failed_rows("value", [1], "missing")
        if raises:
            raise failure
        return [7, None]

    if raises:
        with pytest.raises(ValueError) as caught:
            with collect_failed_rows("ManualPlugin"):
                instrument(context, calculate)()
        assert caught.value is failure
    else:
        with collect_failed_rows("ManualPlugin"):
            result = instrument(context, calculate)()
        assert result == [7, None]

    assert context.failed_rows == (FailedRows("ManualPlugin", "value", (1,), "missing"),)

    outside = _context("OutsidePlugin")
    instrument(outside, report_failed_rows)("value", [0], "outside")
    assert outside.failed_rows == ()


def test_public_nested_collectors_isolate_reports_and_restore_the_outer_scope() -> None:
    from mloda.steward import collect_failed_rows

    outer = _context("OuterPlugin")
    inner = _context("InnerPlugin")

    def calculate_outer() -> None:
        report_failed_rows("value", [0], "before")
        with collect_failed_rows("InnerPlugin"):
            instrument(inner, report_failed_rows)("value", [1], "inside")
        report_failed_rows("value", [2], "after")

    with collect_failed_rows("OuterPlugin"):
        instrument(outer, calculate_outer)()

    assert outer.failed_rows == (
        FailedRows("OuterPlugin", "value", (0,), "before"),
        FailedRows("OuterPlugin", "value", (2,), "after"),
    )
    assert inner.failed_rows == (FailedRows("InnerPlugin", "value", (1,), "inside"),)

    outside = _context("OutsidePlugin")
    instrument(outside, report_failed_rows)("value", [3], "outside")
    assert outside.failed_rows == ()
