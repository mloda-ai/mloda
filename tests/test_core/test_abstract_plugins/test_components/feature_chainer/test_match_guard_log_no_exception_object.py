"""A match_guard raise is logged as text, never as the exception object (#945).

The object would land in LogRecord.args, and a retained record then pins exc.__traceback__,
whose frames hold ``cls`` and with it the dynamically loaded FeatureGroup class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import PropertySpec
from mloda.user import Options

MIXIN_LOGGER_NAME = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin"


class GuardExploded(RuntimeError):
    """An exception class no judgment failure uses, so the guard looks broken rather than undecided."""


def _guard_raises_type_error(value: Any) -> bool:
    raise TypeError("guard cannot judge this value")


def _guard_raises_unexpected(value: Any) -> bool:
    raise GuardExploded("guard is broken")


class _JudgmentRaiseGuard945(FeatureChainParserMixin):
    """Guard raising an expected judgment class, which takes the DEBUG branch."""

    PREFIX_PATTERN = r".*__([\w]+)_g945debug$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec("Operation", allowed_values={"op1": "Operation 1"}, context=True),
        "items": PropertySpec("Guarded items", context=True, match_guard=_guard_raises_type_error),
    }


class _UnexpectedRaiseGuard945(FeatureChainParserMixin):
    """Guard raising an unexpected class, which takes the WARNING branch."""

    PREFIX_PATTERN = r".*__([\w]+)_g945warn$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec("Operation", allowed_values={"op1": "Operation 1"}, context=True),
        "items": PropertySpec("Guarded items", context=True, match_guard=_guard_raises_unexpected),
    }


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Every exception object the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _guard_raise_records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == MIXIN_LOGGER_NAME and record.levelno == level and "raised" in record.getMessage()
    ]


class TestMatchGuardRaiseLogsNoExceptionObject:
    """The contained guard raise must reach the log as text only."""

    def test_debug_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        options = Options(context={"operation": "op1", "items": 5})
        with caplog.at_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME):
            result = _JudgmentRaiseGuard945.match_feature_group_criteria("src__op1_g945debug", options)

        assert result is False
        assert _guard_raise_records(caplog, logging.DEBUG), "the DEBUG branch must report the contained raise"
        for record in caplog.records:
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_warning_branch_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        options = Options(context={"operation": "op1", "items": 5})
        with caplog.at_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME):
            result = _UnexpectedRaiseGuard945.match_feature_group_criteria("src__op1_g945warn", options)

        assert result is False
        assert _guard_raise_records(caplog, logging.WARNING), "the WARNING branch must report the contained raise"
        for record in caplog.records:
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_debug_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        options = Options(context={"operation": "op1", "items": 5})
        with caplog.at_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME):
            _JudgmentRaiseGuard945.match_feature_group_criteria("src__op1_g945debug", options)

        messages = [record.getMessage() for record in _guard_raise_records(caplog, logging.DEBUG)]
        assert len(messages) == 1, f"exactly one DEBUG record reports the raise, got: {messages}"
        assert "items" in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "TypeError" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert "guard cannot judge this value" in messages[0], f"the reason must stay readable: {messages[0]}"

    def test_warning_message_still_names_key_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        options = Options(context={"operation": "op1", "items": 5})
        with caplog.at_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME):
            _UnexpectedRaiseGuard945.match_feature_group_criteria("src__op1_g945warn", options)

        messages = [record.getMessage() for record in _guard_raise_records(caplog, logging.WARNING)]
        assert len(messages) == 1, f"exactly one WARNING record reports the raise, got: {messages}"
        assert "items" in messages[0], f"the key must stay in the message: {messages[0]}"
        assert "GuardExploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert repr(5) not in messages[0], f"the raw value stays out of WARNING logs: {messages[0]}"
