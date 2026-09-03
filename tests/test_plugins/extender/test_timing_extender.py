import logging
from typing import Any

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda_plugins.function_extender.base_implementations.timing.timing_extender import TimingExtender


class TestTimingExtender:
    def test_wraps_feature_group_calculate_feature(self) -> None:
        assert TimingExtender().wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def test_call_returns_wrapped_result(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        assert TimingExtender()(add, 1, 2) == 3

    def test_call_logs_duration(self, caplog: Any) -> None:
        def noop() -> str:
            return "ok"

        with caplog.at_level(logging.INFO):
            TimingExtender()(noop)

        assert len(caplog.records) == 1
