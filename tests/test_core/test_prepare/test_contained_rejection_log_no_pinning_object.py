"""A contained value rejection at the identify seam is logged as text, never as pinning objects.

A retained record pins whatever sits in ``LogRecord.args``: the candidate class directly, and the
rejection object through ``exc.__traceback__``, whose frames reach the same dynamically loaded class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import PropertySpec

IDENTIFY_LOGGER_NAME = IdentifyFeatureGroupClass.__module__

CRLOG_KEY = "crlog_rejected_key"
CRLOG_VALUE = "crlog_rejected_value"
CRLOG_FEATURE = "source__rejected_crlog"
CRLOG_PATTERN = r".*__rejected_crlog$"
CRLOG_CLASS_NAME = "RejectingValueFGCrlog"
CRLOG_VALIDATOR_MESSAGE = "crlog validator cannot judge"


def _crlog_validator_raises(value: Any) -> bool:
    raise TypeError(CRLOG_VALIDATOR_MESSAGE)


CRLOG_PROPERTY_MAPPING = {
    CRLOG_KEY: PropertySpec(
        "crlog touchy", context=True, strict_validation=True, element_validator=_crlog_validator_raises
    )
}


class CrlogFw(ComputeFramework):
    """Dummy compute framework for the contained-rejection log tests."""


class RejectingValueFGCrlog(FeatureGroup):
    """Plugin double that calls the parser directly, so a rejection reaches the identify seam.

    Module level on purpose: a locally defined double is registered mid-test, and the record under
    test pins it, so the registry-pollution guard would fire on the leak instead of the log contract.
    It is inert for every other feature name.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CrlogFw}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != CRLOG_FEATURE:
            return False
        return FeatureChainParser.match_configuration_feature_chain_parser(
            str(feature_name),
            options,
            property_mapping=CRLOG_PROPERTY_MAPPING,
            prefix_patterns=[CRLOG_PATTERN],
        )

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _class_args(record: logging.LogRecord) -> list[type]:
    """Class objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, type)]


def _rejection_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only the identify seam's own records; a neighbouring module's DEBUG record is not this contract."""
    return [
        record
        for record in caplog.records
        if record.name == IDENTIFY_LOGGER_NAME
        and record.levelno == logging.DEBUG
        and "rejected an option value" in record.getMessage()
    ]


def _identify_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == IDENTIFY_LOGGER_NAME]


def _evaluate(caplog: pytest.LogCaptureFixture) -> None:
    """Drive one evaluation whose only candidate rejects the option value."""
    feature = Feature(CRLOG_FEATURE, Options(context={CRLOG_KEY: [CRLOG_VALUE]}))
    plugins: FeatureGroupEnvironmentMapping = {RejectingValueFGCrlog: {CrlogFw}}
    with caplog.at_level(logging.DEBUG, logger=IDENTIFY_LOGGER_NAME):
        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None)
    assert not result.identified, "the rejected candidate must not win the feature"


class TestContainedRejectionLogsNoPinningObject:
    """The contained PropertyValueRejection must reach the log as text only."""

    def test_record_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        _evaluate(caplog)

        assert _rejection_records(caplog), "the contained rejection must report at DEBUG"
        for record in _identify_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"

    def test_record_retains_no_class_object(self, caplog: pytest.LogCaptureFixture) -> None:
        _evaluate(caplog)

        assert _rejection_records(caplog), "the contained rejection must report at DEBUG"
        for record in _identify_records(caplog):
            assert _class_args(record) == [], f"record pins a class object: {record.getMessage()}"

    def test_record_retains_no_traceback_via_exc_info(self, caplog: pytest.LogCaptureFixture) -> None:
        _evaluate(caplog)

        assert _rejection_records(caplog), "the contained rejection must report at DEBUG"
        for record in _identify_records(caplog):
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_message_still_names_group_feature_and_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        _evaluate(caplog)

        messages = [record.getMessage() for record in _rejection_records(caplog)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the rejection, got: {messages}"
        assert CRLOG_CLASS_NAME in messages[0], f"the rejecting group must stay in the message: {messages[0]}"
        assert CRLOG_FEATURE in messages[0], f"the feature name must stay in the message: {messages[0]}"
        assert CRLOG_KEY in messages[0], f"the rejected key must stay in the message: {messages[0]}"
        assert CRLOG_VALUE in messages[0], f"the rejection text must stay readable: {messages[0]}"
