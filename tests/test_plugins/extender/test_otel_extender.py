import logging
from typing import Any
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.function_extender import _CompositeExtender
from mloda.steward import ExtenderHook
from mloda_plugins.function_extender.base_implementations.otel.otel_extender import OtelExtender


def test_otel_extender(caplog: Any) -> None:
    otel = OtelExtender()
    assert otel.wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def func(arg1: Any, arg2: Any) -> Any:
        return arg1 + arg2

    result = otel(func, 1, 2)
    assert result == 3
    assert "OtelExtender" in caplog.text


def test_otel_extender_wraps_returns_empty_set_when_trace_missing() -> None:
    with patch(
        "mloda_plugins.function_extender.base_implementations.otel.otel_extender.trace",
        None,
    ):
        otel = OtelExtender()
        assert otel.wraps() == set()


def test_otel_extender_call_still_works_when_trace_missing(caplog: Any) -> None:
    with patch(
        "mloda_plugins.function_extender.base_implementations.otel.otel_extender.trace",
        None,
    ):
        otel = OtelExtender()

        def func(x: int, y: int) -> int:
            return x + y

        result = otel(func, 3, 4)
        assert result == 7
        assert "OtelExtender" in caplog.text


def test_otel_extender_opts_into_warning_only() -> None:
    """OtelExtender is observability-only: its failures must not break calculations."""
    assert OtelExtender().raise_on_error is False


def test_failing_otel_extender_falls_back_with_warning(caplog: Any) -> None:
    """A failing OtelExtender must not propagate: it logs a warning and the inner result is returned.

    The extender body is forced to fail (independent of the wrapped function) by making
    its logger raise; the wrapped function itself succeeds, so the fallback returns its result.
    """
    otel = OtelExtender()
    composite = _CompositeExtender([otel])

    def func(x: int, y: int) -> int:
        return x + y

    with patch(
        "mloda_plugins.function_extender.base_implementations.otel.otel_extender.logger.warning",
        side_effect=RuntimeError("otel instrumentation boom"),
    ):
        with caplog.at_level(logging.WARNING):
            result = composite(func, 3, 4)

    assert result == 7, "Failing OtelExtender must fall back to the wrapped function result"
    assert any(record.levelno == logging.WARNING and "OtelExtender" in record.message for record in caplog.records), (
        "A failing warning-only OtelExtender should be logged at WARNING level"
    )


def test_otel_extender_constructor_raise_on_error_true() -> None:
    """The constructor exposes raise_on_error so callers can opt into breaking behavior."""
    assert OtelExtender(raise_on_error=True).raise_on_error is True


def test_otel_extender_constructor_raise_on_error_false_explicit() -> None:
    """Passing raise_on_error=False explicitly keeps warning-only behavior."""
    assert OtelExtender(raise_on_error=False).raise_on_error is False


def test_failing_otel_extender_propagates_when_raise_on_error_true(caplog: Any) -> None:
    """With raise_on_error=True the extender's own failure must PROPAGATE, not fall back.

    Same failure-injection technique as test_failing_otel_extender_falls_back_with_warning
    (patch the otel module's logger.warning to raise), but the extender is constructed with
    raise_on_error=True, so the composite call must re-raise the injected error rather than
    returning the inner result.
    """
    otel = OtelExtender(raise_on_error=True)
    composite = _CompositeExtender([otel])

    def func(x: int, y: int) -> int:
        return x + y

    with patch(
        "mloda_plugins.function_extender.base_implementations.otel.otel_extender.logger.warning",
        side_effect=RuntimeError("otel instrumentation boom"),
    ):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="otel instrumentation boom"):
                composite(func, 3, 4)
