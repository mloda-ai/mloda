import logging
from typing import Any

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda.steward import HookContext
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

    def test_log_names_feature_group_when_context_active(self, caplog: Any) -> None:
        context = HookContext(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            feature_group_class="my_module.MyFeatureGroup",
            feature_group_version="v1",
            compute_framework_name="FakeFramework",
        )

        with caplog.at_level(logging.INFO), context.activate():
            result = TimingExtender()(lambda: "ok")

        assert result == "ok"
        assert "Time taken for my_module.MyFeatureGroup:" in caplog.records[0].getMessage()

    def test_log_falls_back_without_hook_context(self, caplog: Any) -> None:
        with caplog.at_level(logging.INFO):
            result = TimingExtender()(lambda: "ok")

        assert result == "ok"
        assert "Time taken for unknown feature group:" in caplog.records[0].getMessage()
