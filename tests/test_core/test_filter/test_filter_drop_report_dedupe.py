"""The drop and falsy-match reports dedupe on the rendered line, not on the column key.

Two filters declared on one column can be dropped by one feature group for different
reasons, and one declaration can be probed against different host features. Keying the
dedupe on ``(feature group, filter feature name)`` muted every report after the first to
DEBUG, even when its text was new. ``_warn_on_diverging_options`` already dedupes on the
message; these two now match it.
"""

import logging

import pytest

from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class DropReportingFeatureGroup(FeatureGroup):
    """Only used as a ledger key; nothing is computed."""


def _lines(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def test_distinct_drop_reasons_each_warn(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()

    with caplog.at_level(logging.DEBUG):
        global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")
        global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised B")

    warnings = _lines(caplog, logging.WARNING)
    assert len(warnings) == 2
    assert any("raised A" in line for line in warnings)
    assert any("raised B" in line for line in warnings)


def test_repeated_drop_reason_is_demoted_to_debug(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()

    with caplog.at_level(logging.DEBUG):
        global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")
        global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")

    assert len(_lines(caplog, logging.WARNING)) == 1
    assert len(_lines(caplog, logging.DEBUG)) == 1


def test_ledger_keeps_the_first_defect_per_key() -> None:
    """The ledger stays per declaration: a second reason does not overwrite the recorded one."""
    global_filter = GlobalFilter()

    global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")
    global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised B")

    recorded = global_filter.dropped_filters[(DropReportingFeatureGroup, "amount")]
    assert recorded.stage == "matcher_error"
    assert recorded.reason == "raised A"


def test_defect_still_outranks_a_stored_near_miss() -> None:
    """A near-miss recorded first must not stop the defect from claiming the key."""
    global_filter = GlobalFilter()

    global_filter._record_near_miss(DropReportingFeatureGroup, "amount", "domain", "lost at the domain gate")
    global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")

    recorded = global_filter.dropped_filters[(DropReportingFeatureGroup, "amount")]
    assert recorded.stage == "matcher_error"
    assert recorded.reason == "raised A"


def test_distinct_falsy_match_reports_each_warn(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()

    with caplog.at_level(logging.DEBUG):
        global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", 0)
        global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", "")

    warnings = _lines(caplog, logging.WARNING)
    assert len(warnings) == 2
    assert any("(int)" in line for line in warnings)
    assert any("(str)" in line for line in warnings)


def test_repeated_falsy_match_report_is_demoted_to_debug(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()

    with caplog.at_level(logging.DEBUG):
        global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", 0)
        global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", 0)

    assert len(_lines(caplog, logging.WARNING)) == 1
    assert len(_lines(caplog, logging.DEBUG)) == 1


def test_reset_match_tracking_clears_both_ledgers(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()

    global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")
    global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", 0)
    global_filter.reset_match_tracking()
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        global_filter._record_dropped_filter(DropReportingFeatureGroup, "amount", "raised A")
        global_filter._report_falsy_match(DropReportingFeatureGroup, "amount", 0)

    # Both report at WARNING again, and the ledger takes the drop again.
    assert len(_lines(caplog, logging.WARNING)) == 2
    assert (DropReportingFeatureGroup, "amount") in global_filter.dropped_filters
