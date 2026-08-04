"""A contained chainer-side raise is logged as text, never as the exception object.

A retained record holding the object would pin exc.__traceback__ and through it the plugin class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards import check_required_when
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    PropertyValueRejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.user import Options

PARSER_LOGGER_NAME = FeatureChainParser.__module__
GUARDS_LOGGER_NAME = check_required_when.__module__

CLR975_VALIDATOR_KEY = "clr975_validator_key"
CLR975_PREDICATE_KEY = "clr975_predicate_key"
CLR975_VALUE = "clr975_value"
CLR975_FEATURE = "text__validated_clr975"
CLR975_PREFIX_PATTERN = r".*__validated_clr975$"
CLR975_OWNER = "Clr975Owner"

CLR975_VALIDATOR_BROKEN_MESSAGE = "clr975 validator looks broken"
CLR975_VALIDATOR_JUDGMENT_MESSAGE = "clr975 validator cannot judge"
CLR975_PREDICATE_BROKEN_MESSAGE = "clr975 predicate looks broken"
CLR975_PREDICATE_JUDGMENT_MESSAGE = "clr975 predicate cannot judge"


class Clr975Exploded(RuntimeError):
    """An exception type no judgment failure uses, so the callable looks broken rather than undecided."""


def _clr975_validator_raises_unexpected(value: Any) -> bool:
    raise Clr975Exploded(CLR975_VALIDATOR_BROKEN_MESSAGE)


def _clr975_validator_raises_type_error(value: Any) -> bool:
    raise TypeError(CLR975_VALIDATOR_JUDGMENT_MESSAGE)


def _clr975_predicate_raises_unexpected(options: Any) -> bool:
    raise Clr975Exploded(CLR975_PREDICATE_BROKEN_MESSAGE)


def _clr975_predicate_raises_type_error(options: Any) -> bool:
    raise TypeError(CLR975_PREDICATE_JUDGMENT_MESSAGE)


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _raise_records(caplog: pytest.LogCaptureFixture, logger_name: str, level: int) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == level and "raised" in record.getMessage()
    ]


def _module_records(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[logging.LogRecord]:
    """Only the module under test; DEBUG capture is global, so a neighbour's record is not this contract."""
    return [record for record in caplog.records if record.name == logger_name]


def _run_validator(validator: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Drive the parser's element_validator seam."""
    property_mapping = {
        CLR975_VALIDATOR_KEY: PropertySpec(
            "clr975 touchy", context=True, strict_validation=True, element_validator=validator
        )
    }
    with caplog.at_level(logging.DEBUG, logger=PARSER_LOGGER_NAME):
        with pytest.raises(PropertyValueRejection):
            FeatureChainParser.match_configuration_feature_chain_parser(
                CLR975_FEATURE,
                Options(context={CLR975_VALIDATOR_KEY: [CLR975_VALUE]}),
                property_mapping=property_mapping,
                prefix_patterns=[CLR975_PREFIX_PATTERN],
            )


def _run_predicate(predicate: Any, caplog: pytest.LogCaptureFixture) -> bool:
    """Drive the author-guard required_when seam."""
    property_mapping = {CLR975_PREDICATE_KEY: PropertySpec("clr975 undecidable", required_when=predicate)}
    with caplog.at_level(logging.DEBUG, logger=GUARDS_LOGGER_NAME):
        return check_required_when(CLR975_OWNER, CLR975_FEATURE, [], property_mapping, Options())


class TestParserElementValidatorRaiseLogsNoExceptionObject:
    """The parser's contained element_validator raise must reach the log as text only."""

    def test_warning_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_validator(_clr975_validator_raises_unexpected, caplog)

        assert _raise_records(caplog, PARSER_LOGGER_NAME, logging.WARNING), "the broken-looking validator warns"
        for record in _module_records(caplog, PARSER_LOGGER_NAME):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_debug_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_validator(_clr975_validator_raises_type_error, caplog)

        assert _raise_records(caplog, PARSER_LOGGER_NAME, logging.DEBUG), "the judgment failure reports at DEBUG"
        for record in _module_records(caplog, PARSER_LOGGER_NAME):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_warning_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_validator(_clr975_validator_raises_unexpected, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, PARSER_LOGGER_NAME, logging.WARNING)]
        assert len(messages) == 1, f"exactly one WARNING record reports the raise, got: {messages}"
        assert CLR975_VALIDATOR_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "Clr975Exploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert CLR975_VALIDATOR_BROKEN_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
        assert "raised " in messages[0], f"the message must still read as a contained raise: {messages[0]}"
        assert CLR975_VALUE not in messages[0], f"the raw value stays out of WARNING logs: {messages[0]}"

    def test_debug_message_still_names_key_exception_type_and_value(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_validator(_clr975_validator_raises_type_error, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, PARSER_LOGGER_NAME, logging.DEBUG)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert CLR975_VALIDATOR_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "TypeError" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert CLR975_VALIDATOR_JUDGMENT_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
        assert CLR975_VALUE in messages[0], f"DEBUG keeps the rejected value visible: {messages[0]}"


class TestRequiredWhenRaiseLogsNoExceptionObject:
    """The author guard's contained required_when raise must reach the log as text only."""

    def test_warning_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _run_predicate(_clr975_predicate_raises_unexpected, caplog) is False
        assert _raise_records(caplog, GUARDS_LOGGER_NAME, logging.WARNING), "the broken-looking predicate warns"
        for record in _module_records(caplog, GUARDS_LOGGER_NAME):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_debug_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _run_predicate(_clr975_predicate_raises_type_error, caplog) is False
        assert _raise_records(caplog, GUARDS_LOGGER_NAME, logging.DEBUG), "the judgment failure reports at DEBUG"
        for record in _module_records(caplog, GUARDS_LOGGER_NAME):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_warning_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_predicate(_clr975_predicate_raises_unexpected, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, GUARDS_LOGGER_NAME, logging.WARNING)]
        assert len(messages) == 1, f"exactly one WARNING record reports the raise, got: {messages}"
        assert CLR975_PREDICATE_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert CLR975_OWNER in messages[0], f"the owner must stay in the message: {messages[0]}"
        assert "Clr975Exploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert CLR975_PREDICATE_BROKEN_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
        assert "raised " in messages[0], f"the message must still read as a contained raise: {messages[0]}"

    def test_debug_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_predicate(_clr975_predicate_raises_type_error, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, GUARDS_LOGGER_NAME, logging.DEBUG)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert CLR975_PREDICATE_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "TypeError" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert CLR975_PREDICATE_JUDGMENT_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
