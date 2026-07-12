"""Tests for the safe_field plugin-catalog graceful-degradation helper.

safe_field is the "annotate" tier: it reads one introspected field and, if the
read raises an exception in `catching`, returns a caller-supplied fallback
instead of propagating. Exceptions outside `catching` still propagate.

A swallowed read is not silent: safe_field logs a WARNING naming the field
label (the optional `field` argument) and the exception it swallowed, so a
broken plugin is diagnosable from the logs even though the catalog call
degrades instead of failing.
"""

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import safe_field

SAFE_FIELD_LOGGER = "mloda.core.abstract_plugins.components.utils"


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return WARNING messages emitted by the safe_field module logger."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == SAFE_FIELD_LOGGER
    ]


class TestSafeFieldSuccessPath:
    """When the read succeeds, its value is returned and the fallback is ignored."""

    def test_returns_read_value_on_success(self) -> None:
        assert safe_field(lambda: 42, -1) == 42


class TestSafeFieldDefaultCatching:
    """The default catching=(Exception,) swallows ordinary exceptions."""

    def test_default_catching_swallows_runtime_error(self) -> None:
        def raises() -> str:
            raise RuntimeError("boom")

        assert safe_field(raises, "unavailable") == "unavailable"


class TestSafeFieldNarrowCatching:
    """A narrow catching tuple swallows listed types and propagates the rest."""

    def test_returns_fallback_for_listed_exception_type(self) -> None:
        def raises() -> str:
            raise OSError("disk gone")

        assert safe_field(raises, "fallback", catching=(OSError, TypeError)) == "fallback"

    def test_propagates_unlisted_exception_type(self) -> None:
        def raises() -> str:
            raise ValueError("not caught")

        with pytest.raises(ValueError):
            safe_field(raises, "fallback", catching=(OSError, TypeError))


class TestSafeFieldFallbackIdentity:
    """The fallback value is returned as-is, preserving type and identity."""

    @pytest.mark.parametrize("fallback", [False, [], None, "", 0])
    def test_fallback_returned_as_is_on_failure(self, fallback: Any) -> None:
        def raises() -> Any:
            raise RuntimeError("boom")

        result = safe_field(raises, fallback)

        assert result is fallback


class TestSafeFieldWarnsOnSwallow:
    """A swallowed read logs a WARNING naming the field label and the exception."""

    def test_swallowed_read_logs_warning_with_field_label_and_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        def raises() -> str:
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger=SAFE_FIELD_LOGGER):
            result = safe_field(raises, "unavailable", field="description")

        assert result == "unavailable"

        messages = _warning_messages(caplog)
        assert len(messages) == 1, f"Expected exactly one WARNING, got {messages}"
        assert "description" in messages[0], "Warning must name the field that was being read"
        assert "boom" in messages[0], "Warning must carry the swallowed exception message"

    def test_swallowed_read_warning_names_the_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        def raises() -> str:
            raise KeyError("missing_key")

        with caplog.at_level(logging.WARNING, logger=SAFE_FIELD_LOGGER):
            result = safe_field(raises, "unavailable", field="prefix")

        assert result == "unavailable"

        messages = _warning_messages(caplog)
        assert len(messages) == 1, f"Expected exactly one WARNING, got {messages}"
        assert "prefix" in messages[0]
        assert "missing_key" in messages[0]


class TestSafeFieldSuccessPathIsSilent:
    """The success path never logs: only a degraded read is worth a warning."""

    def test_success_path_logs_nothing_with_field_label(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger=SAFE_FIELD_LOGGER):
            result = safe_field(lambda: 42, -1, field="version")

        assert result == 42
        assert [record.getMessage() for record in caplog.records if record.name == SAFE_FIELD_LOGGER] == []

    def test_success_path_logs_nothing_without_field_label(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger=SAFE_FIELD_LOGGER):
            result = safe_field(lambda: 42, -1)

        assert result == 42
        assert [record.getMessage() for record in caplog.records if record.name == SAFE_FIELD_LOGGER] == []


class TestSafeFieldFieldLabelKeepsExistingBehavior:
    """The field label is additive: fallback and narrow-catching semantics are unchanged."""

    def test_field_label_is_optional(self) -> None:
        """Existing call sites that pass no field label keep working."""

        def raises() -> str:
            raise RuntimeError("boom")

        assert safe_field(raises, "unavailable") == "unavailable"

    def test_fallback_still_returned_with_field_label(self) -> None:
        def raises() -> str:
            raise OSError("disk gone")

        assert safe_field(raises, "fallback", catching=(OSError, TypeError), field="version") == "fallback"

    def test_unlisted_exception_still_propagates_with_field_label(self, caplog: pytest.LogCaptureFixture) -> None:
        """A narrow catching tuple still propagates unlisted types, and nothing is logged."""

        def raises() -> str:
            raise ValueError("not caught")

        with caplog.at_level(logging.DEBUG, logger=SAFE_FIELD_LOGGER):
            with pytest.raises(ValueError):
                safe_field(raises, "fallback", catching=(OSError, TypeError), field="version")

        assert _warning_messages(caplog) == [], "A propagated exception is not a swallowed read, so it must not warn"
