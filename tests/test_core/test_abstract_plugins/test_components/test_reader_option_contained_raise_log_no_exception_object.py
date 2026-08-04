"""A contained reader-option raise is logged as text, never as the exception object.

A retained record holding the object would pin exc.__traceback__ and through it the reader class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.user import Options

READER_LOGGER_NAME = BaseInputData.__module__

RIC975_PREDICATE_KEY = "ric975_predicate_key"
RIC975_VALIDATOR_KEY = "ric975_validator_key"
RIC975_VALUE = "ric975_value"

RIC975_BROKEN_MESSAGE = "ric975 predicate looks broken"
RIC975_JUDGMENT_MESSAGE = "ric975 predicate cannot judge"
RIC975_VALIDATOR_BROKEN_MESSAGE = "ric975 validator looks broken"
RIC975_VALIDATOR_JUDGMENT_MESSAGE = "ric975 validator cannot judge"


class Ric975Exploded(RuntimeError):
    """An exception type no judgment failure uses, so the callable looks broken rather than undecided."""


def _ric975_predicate_raises_unexpected(options: Any) -> bool:
    raise Ric975Exploded(RIC975_BROKEN_MESSAGE)


def _ric975_predicate_raises_type_error(options: Any) -> bool:
    raise TypeError(RIC975_JUDGMENT_MESSAGE)


def _ric975_validator_raises_unexpected(value: Any) -> bool:
    raise Ric975Exploded(RIC975_VALIDATOR_BROKEN_MESSAGE)


def _ric975_validator_raises_type_error(value: Any) -> bool:
    raise TypeError(RIC975_VALIDATOR_JUDGMENT_MESSAGE)


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _reader_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only the module under test; DEBUG capture is global, so a neighbour's record is not this contract."""
    return [record for record in caplog.records if record.name == READER_LOGGER_NAME]


def _raise_records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == READER_LOGGER_NAME and record.levelno == level and "raised" in record.getMessage()
    ]


def _absent_key_verdict(spec: PropertySpec, caplog: pytest.LogCaptureFixture) -> bool:
    """Run the absent-key seam under DEBUG capture."""
    with caplog.at_level(logging.DEBUG, logger=READER_LOGGER_NAME):
        return BaseInputData._absent_reader_option_admits(RIC975_PREDICATE_KEY, spec, Options(), False)


def _element_verdict(spec: PropertySpec, caplog: pytest.LogCaptureFixture) -> bool:
    """Run the element seam under DEBUG capture."""
    with caplog.at_level(logging.DEBUG, logger=READER_LOGGER_NAME):
        return BaseInputData._reader_option_element_admits(RIC975_VALIDATOR_KEY, spec, RIC975_VALUE)


class TestRequiredWhenRaiseLogsNoExceptionObject:
    """The contained required_when raise of a reader option must reach the log as text only."""

    def test_warning_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec("ric975 undecidable", required_when=_ric975_predicate_raises_unexpected)

        assert _absent_key_verdict(spec, caplog) is False
        assert _raise_records(caplog, logging.WARNING), "the broken-looking predicate must report at WARNING"
        for record in _reader_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_debug_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec("ric975 undecidable", required_when=_ric975_predicate_raises_type_error)

        assert _absent_key_verdict(spec, caplog) is False
        assert _raise_records(caplog, logging.DEBUG), "the expected judgment failure must report at DEBUG"
        for record in _reader_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_warning_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec("ric975 undecidable", required_when=_ric975_predicate_raises_unexpected)
        _absent_key_verdict(spec, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, logging.WARNING)]
        assert len(messages) == 1, f"exactly one WARNING record reports the raise, got: {messages}"
        assert RIC975_PREDICATE_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "Ric975Exploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert RIC975_BROKEN_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
        assert "raised " in messages[0], f"the message must still read as a contained raise: {messages[0]}"

    def test_debug_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec("ric975 undecidable", required_when=_ric975_predicate_raises_type_error)
        _absent_key_verdict(spec, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, logging.DEBUG)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert RIC975_PREDICATE_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "TypeError" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert RIC975_JUDGMENT_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"


class TestElementValidatorRaiseLogsNoExceptionObject:
    """The contained element_validator raise of a reader option must reach the log as text only."""

    def test_warning_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec(
            "ric975 touchy", element_validator=_ric975_validator_raises_unexpected, strict_validation=True
        )

        assert _element_verdict(spec, caplog) is False
        assert _raise_records(caplog, logging.WARNING), "the broken-looking validator must report at WARNING"
        for record in _reader_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_debug_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec(
            "ric975 touchy", element_validator=_ric975_validator_raises_type_error, strict_validation=True
        )

        assert _element_verdict(spec, caplog) is False
        assert _raise_records(caplog, logging.DEBUG), "the expected judgment failure must report at DEBUG"
        for record in _reader_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_warning_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec(
            "ric975 touchy", element_validator=_ric975_validator_raises_unexpected, strict_validation=True
        )
        _element_verdict(spec, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, logging.WARNING)]
        assert len(messages) == 1, f"exactly one WARNING record reports the raise, got: {messages}"
        assert RIC975_VALIDATOR_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "Ric975Exploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert RIC975_VALIDATOR_BROKEN_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
        assert "raised " in messages[0], f"the message must still read as a contained raise: {messages[0]}"

    def test_debug_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = PropertySpec(
            "ric975 touchy", element_validator=_ric975_validator_raises_type_error, strict_validation=True
        )
        _element_verdict(spec, caplog)

        messages = [record.getMessage() for record in _raise_records(caplog, logging.DEBUG)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert RIC975_VALIDATOR_KEY in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "TypeError" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert RIC975_VALIDATOR_JUDGMENT_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
