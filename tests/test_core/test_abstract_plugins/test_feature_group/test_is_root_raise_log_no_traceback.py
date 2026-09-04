"""A contained input_features raise is logged as text, never with exc_info.

``exc_info=True`` puts (type, exc, traceback) on the record, and a retained record then pins the
traceback frames and through them the dynamically loaded FeatureGroup class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup

FEATURE_GROUP_LOGGER_NAME = FeatureGroup.__module__

ISROOT_FEATURE = "is_root_log_probe_feat"
ISROOT_CLASS_NAME = "IsRootLogProbeFG"
ISROOT_MESSAGE = "is_root probe cannot answer for this feature"


class IsRootProbeExploded(RuntimeError):
    """An unmarked exception type, so the raise stays contained as a non-match."""


class IsRootLogProbeFG(FeatureGroup):
    """Stands in for a group whose input_features cannot answer for this feature name.

    Module level on purpose: a locally defined double is registered mid-test, and the record under
    test pins it, so the registry-pollution guard would fire on the leak instead of the log contract.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        raise IsRootProbeExploded(ISROOT_MESSAGE)


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _feature_group_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only the feature_group module's own records; a neighbouring module's DEBUG record is not this contract."""
    return [record for record in caplog.records if record.name == FEATURE_GROUP_LOGGER_NAME]


def _raise_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in _feature_group_records(caplog)
        if record.levelno == logging.DEBUG and "input_features" in record.getMessage()
    ]


def _run_is_root(caplog: pytest.LogCaptureFixture) -> bool:
    """Drive the contained input_features branch of is_root under DEBUG capture."""
    with caplog.at_level(logging.DEBUG, logger=FEATURE_GROUP_LOGGER_NAME):
        return IsRootLogProbeFG().is_root(Options(), ISROOT_FEATURE)


class TestIsRootContainedRaiseLogsNoTraceback:
    """The contained input_features raise must reach the log as text only."""

    def test_record_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _run_is_root(caplog) is False

        assert _raise_records(caplog), "the contained raise must report at DEBUG"
        for record in _feature_group_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"

    def test_record_retains_no_traceback_via_exc_info(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _run_is_root(caplog) is False

        assert _raise_records(caplog), "the contained raise must report at DEBUG"
        for record in _feature_group_records(caplog):
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_message_still_names_class_feature_and_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_is_root(caplog)

        messages = [record.getMessage() for record in _raise_records(caplog)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert ISROOT_CLASS_NAME in messages[0], f"the class must stay in the message: {messages[0]}"
        assert ISROOT_FEATURE in messages[0], f"the feature name must stay in the message: {messages[0]}"
        assert "IsRootProbeExploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert ISROOT_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
