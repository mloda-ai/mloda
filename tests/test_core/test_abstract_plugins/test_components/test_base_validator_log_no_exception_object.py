"""A validation failure contained by a non-error log level is logged as text, never with exc_info.

``exc_info=<exception>`` puts (type, exc, traceback) on the record, and a retained record then pins
the traceback frames and through them the validator and the data it was judging.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.base_validator import BaseValidator

VALIDATOR_LOGGER_NAME = BaseValidator.__module__

VALIDATOR_ERROR = "row count below the configured minimum"
VALIDATOR_MESSAGE = "validator cannot judge this data"


class ValidationExploded(ValueError):
    """The failure the validator contains at the non-error log levels."""


class ProbeValidator(BaseValidator):
    def validate(self, data: Any) -> None:
        self.handle_log_level(VALIDATOR_ERROR, ValidationExploded(VALIDATOR_MESSAGE))


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _validator_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == VALIDATOR_LOGGER_NAME]


def _run_validator(caplog: pytest.LogCaptureFixture, log_level: str) -> None:
    with caplog.at_level(logging.DEBUG, logger=VALIDATOR_LOGGER_NAME):
        ProbeValidator({}, log_level=log_level).validate(data=None)


CONTAINING_LEVELS = ["warning", "info", "debug"]


class TestValidatorContainedFailureLogsNoExceptionObject:
    """Every non-error level must reach the log as text only."""

    @pytest.mark.parametrize("log_level", CONTAINING_LEVELS)
    def test_record_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture, log_level: str) -> None:
        _run_validator(caplog, log_level)

        assert _validator_records(caplog), f"the {log_level} level must report the contained failure"
        for record in _validator_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"

    @pytest.mark.parametrize("log_level", CONTAINING_LEVELS)
    def test_record_retains_no_traceback_via_exc_info(self, caplog: pytest.LogCaptureFixture, log_level: str) -> None:
        _run_validator(caplog, log_level)

        assert _validator_records(caplog), f"the {log_level} level must report the contained failure"
        for record in _validator_records(caplog):
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    @pytest.mark.parametrize("log_level", CONTAINING_LEVELS)
    def test_message_still_names_error_and_exception(self, caplog: pytest.LogCaptureFixture, log_level: str) -> None:
        _run_validator(caplog, log_level)

        messages = [record.getMessage() for record in _validator_records(caplog)]
        assert len(messages) == 1, f"exactly one record reports the failure, got: {messages}"
        assert VALIDATOR_ERROR in messages[0], f"the error must stay in the message: {messages[0]}"
        assert "ValidationExploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert VALIDATOR_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"

    def test_the_error_level_still_raises(self) -> None:
        """Only the containing levels changed; the default level must still hand the exception to the caller."""
        with pytest.raises(ValidationExploded):
            ProbeValidator({}, log_level="error").validate(data=None)
