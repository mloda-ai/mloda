"""Tests for tri-state strict mode on Extender instances (issue #526, cycle 5).

Contract: mloda.core.prepare.accessible_plugins gains
``filter_extenders_by_strict_mode(extenders, plugin_collector)``:

- off (or ``extenders is None``): returned unchanged.
- warn: returned unchanged; one aggregated WARNING per process per
  unregistered extender CLASS (module:qualname keyed, reusing the existing
  _warned_unregistered dedup set, cleared per test by conftest).
- strict: instances whose class is not registered in the injected-or-default
  registry are dropped with a WARNING listing the dropped classes; registered
  instances survive; dropping every instance yields an empty set, no raise.

mlodaAPI wires the helper at the seam where ``self.plugin_collector`` and
``function_extender`` meet: ``_enter_runner_context`` must hand the runner the
FILTERED set under strict mode.

The helper is imported inside each test so every test fails with a precise
ImportError until the Green agent implements it; the integration test fails
on the unfiltered set instead. All doubles are local, keeping xdist
parallel-safety.
"""

import logging
from typing import Any, Optional, cast

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.api.request import mlodaAPI
from mloda.core.runtime.run import ExecutionOrchestrator


class _ExtStrictRegistered(Extender):
    """Local double whose class tests register explicitly."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _ExtStrictUnregisteredA(Extender):
    """Local double whose class is never registered."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _ExtStrictUnregisteredB(Extender):
    """Second never-registered local double, for aggregation tests."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _ExtStrictInjectedOnly(Extender):
    """Local double registered ONLY in a fresh injected registry."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_OUTPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _qualid(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _messages_naming(caplog: pytest.LogCaptureFixture, cls: type) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if _qualid(cls) in rec.getMessage()]


class TestFilterExtendersOffAndNone:
    def test_none_and_off_mode_pass_through_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        monkeypatch.delenv("MLODA_PLUGIN_REGISTRY_STRICT", raising=False)
        collector = PluginCollector().set_strict_mode("strict")
        assert filter_extenders_by_strict_mode(None, collector) is None

        unregistered = _ExtStrictUnregisteredA()
        extenders: set[Extender] = {unregistered}
        off_collector = PluginCollector().set_strict_mode("off")
        assert filter_extenders_by_strict_mode(extenders, off_collector) == extenders
        # No collector: strict mode comes from the env (off when unset).
        assert filter_extenders_by_strict_mode(extenders, None) == extenders


class TestFilterExtendersWarn:
    def test_warn_returns_unchanged_logs_once_per_process(self, caplog: pytest.LogCaptureFixture) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        register_plugin(_ExtStrictRegistered)
        registered = _ExtStrictRegistered()
        unregistered = _ExtStrictUnregisteredA()
        extenders: set[Extender] = {registered, unregistered}
        collector = PluginCollector().set_strict_mode("warn")

        with caplog.at_level(logging.WARNING):
            first = filter_extenders_by_strict_mode(extenders, collector)
            second = filter_extenders_by_strict_mode(extenders, collector)

        assert first == extenders, "warn mode must not drop extender instances"
        assert second == extenders
        naming = _messages_naming(caplog, _ExtStrictUnregisteredA)
        assert len(naming) == 1, (
            f"warn mode must warn exactly once per process per unregistered extender class, got {len(naming)}"
        )
        assert "not registered" in naming[0]
        assert not _messages_naming(caplog, _ExtStrictRegistered), "warn mode must not name registered extender classes"


class TestFilterExtendersStrict:
    def test_strict_drops_unregistered_keeps_registered_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        register_plugin(_ExtStrictRegistered)
        registered = _ExtStrictRegistered()
        unregistered = _ExtStrictUnregisteredA()
        collector = PluginCollector().set_strict_mode("strict")

        with caplog.at_level(logging.WARNING):
            result = filter_extenders_by_strict_mode({registered, unregistered}, collector)

        assert result == {registered}, "strict mode must keep registered and drop unregistered extender instances"
        assert _messages_naming(caplog, _ExtStrictUnregisteredA), (
            "strict mode must warn with the dropped classes as module:qualname"
        )

    def test_strict_dropping_all_yields_empty_set_without_raising(self) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        extenders: set[Extender] = {_ExtStrictUnregisteredA(), _ExtStrictUnregisteredB()}
        collector = PluginCollector().set_strict_mode("strict")
        assert filter_extenders_by_strict_mode(extenders, collector) == set()

    def test_strict_consults_injected_registry(self) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        fresh = PluginRegistry()
        fresh.register(_ExtStrictInjectedOnly)
        register_plugin(_ExtStrictRegistered)  # default registry only: must NOT count

        injected_only = _ExtStrictInjectedOnly()
        default_only = _ExtStrictRegistered()
        collector = PluginCollector().set_strict_mode("strict").set_registry(fresh)

        result = filter_extenders_by_strict_mode({injected_only, default_only}, collector)
        assert result == {injected_only}, (
            "strict mode must consult the injected registry: only the class registered there survives"
        )


class _RecordingRunner:
    """Stand-in for ExecutionOrchestrator that records what __enter__ receives."""

    def __init__(self) -> None:
        self.received_extenders: Optional[set[Extender]] = None

    def __enter__(
        self,
        parallelization_modes: Optional[set[ParallelizationMode]] = None,
        function_extender: Optional[set[Extender]] = None,
        api_data: Optional[dict[str, Any]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> None:
        self.received_extenders = function_extender


class TestRequestWiring:
    def test_runner_receives_filtered_extenders_under_strict_mode(self) -> None:
        """mlodaAPI._enter_runner_context is the seam where self.plugin_collector and
        function_extender meet; under strict mode the runner must receive the
        FILTERED set, not the raw one."""
        register_plugin(_ExtStrictRegistered)
        registered = _ExtStrictRegistered()
        unregistered = _ExtStrictUnregisteredA()

        api = mlodaAPI.__new__(mlodaAPI)
        api.plugin_collector = PluginCollector().set_strict_mode("strict")

        recorder = _RecordingRunner()
        api._enter_runner_context(
            cast(ExecutionOrchestrator, recorder),
            {ParallelizationMode.SYNC},
            {registered, unregistered},
            None,
        )

        assert recorder.received_extenders == {registered}, (
            "the runner must receive the strict-filtered extender set, not the raw one passed to the API"
        )
