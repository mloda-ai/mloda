from typing import Any
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
