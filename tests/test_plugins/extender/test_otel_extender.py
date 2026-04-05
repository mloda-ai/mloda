from typing import Any
from unittest.mock import patch

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
