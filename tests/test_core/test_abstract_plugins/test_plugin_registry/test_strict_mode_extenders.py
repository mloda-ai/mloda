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

mlodaAPI filters extenders once at prepare time into ``self.engine.function_extender``.
``_enter_runner_context`` reuses that snapshot; it never re-reads or re-filters the live
``self.function_extender``, so reassigning or mutating it after prepare has no effect.

The helper is imported inside each test rather than at module scope. The
runner-wiring tests build a real session via PluginLoader with
PythonDictFramework. All doubles are local, keeping xdist parallel-safety.
"""

import logging
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.abstract_plugins.run_context import RunContext
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


class _ExtStrictUnregisteredGate(Extender):
    """Never-registered gate (never_fall_back)."""

    never_fall_back = True

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _ExtStrictInjectedOnly(Extender):
    """Local double registered ONLY in a fresh injected registry."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_OUTPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


_STRICT_WIRING_FEAT = "strict_mode_extenders_wiring_feature_unique_xyz"


class _StrictWiringFeatureGroup(FeatureGroup):
    """Root feature group used to build a real session for the runner-wiring tests."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_STRICT_WIRING_FEAT})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_STRICT_WIRING_FEAT: [1, 2, 3]}


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

    def test_strict_unregistered_gate_raises_naming_class(self) -> None:
        from mloda.core.prepare.accessible_plugins import EnvironmentPreconditionError, filter_extenders_by_strict_mode

        collector = PluginCollector().set_strict_mode("strict")
        with pytest.raises(EnvironmentPreconditionError, match="_ExtStrictUnregisteredGate"):
            filter_extenders_by_strict_mode({_ExtStrictUnregisteredGate(), _ExtStrictUnregisteredA()}, collector)

    def test_strict_registered_gate_is_kept(self) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        register_plugin(_ExtStrictUnregisteredGate)
        gate = _ExtStrictUnregisteredGate()
        collector = PluginCollector().set_strict_mode("strict")
        assert filter_extenders_by_strict_mode({gate}, collector) == {gate}

    def test_warn_keeps_unregistered_gate(self) -> None:
        from mloda.core.prepare.accessible_plugins import filter_extenders_by_strict_mode

        gate = _ExtStrictUnregisteredGate()
        collector = PluginCollector().set_strict_mode("warn")
        assert filter_extenders_by_strict_mode({gate}, collector) == {gate}

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
        self.received_extenders: set[Extender] | None = None

    def __enter__(
        self,
        parallelization_modes: set[ParallelizationMode] | None = None,
        function_extender: set[Extender] | None = None,
        api_data: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        self.received_extenders = function_extender


class TestRequestWiring:
    """Runner receives the engine's prepare-time extender snapshot, never a live re-read of function_extender."""

    def test_reassigning_function_extender_after_prepare_does_not_change_what_the_runner_receives(self) -> None:
        PluginLoader().load_matching("compute_framework", "*python_dict*")
        register_plugin(_StrictWiringFeatureGroup)
        register_plugin(_ExtStrictRegistered)
        registered = _ExtStrictRegistered()
        unregistered = _ExtStrictUnregisteredA()

        session = mlodaAPI.prepare(
            [Feature(_STRICT_WIRING_FEAT)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector().set_strict_mode("strict"),
            function_extender={registered, unregistered},
        )

        session.function_extender = {unregistered}

        recorder = _RecordingRunner()
        session._enter_runner_context(
            cast(ExecutionOrchestrator, recorder),
            {ParallelizationMode.SYNC},
            None,
            run_context=RunContext(),
        )

        assert recorder.received_extenders == {registered}, (
            "the runner must reuse the engine's prepare-time filtered set, not a live re-read of "
            "the (reassigned) api.function_extender attribute"
        )

    def test_mutating_the_set_passed_to_prepare_in_place_does_not_change_what_the_runner_receives(self) -> None:
        # Needs strict mode OFF: strict always builds a new set, off returns the caller's set unchanged.
        PluginLoader().load_matching("compute_framework", "*python_dict*")
        register_plugin(_StrictWiringFeatureGroup)
        register_plugin(_ExtStrictRegistered)
        original = _ExtStrictRegistered()
        extenders: set[Extender] = {original}

        session = mlodaAPI.prepare(
            [Feature(_STRICT_WIRING_FEAT)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_StrictWiringFeatureGroup}).set_strict_mode("off"),
            function_extender=extenders,
        )

        extenders.add(_ExtStrictRegistered())  # mutate the caller's own set object after prepare

        recorder = _RecordingRunner()
        session._enter_runner_context(
            cast(ExecutionOrchestrator, recorder),
            {ParallelizationMode.SYNC},
            None,
            run_context=RunContext(),
        )

        assert recorder.received_extenders == {original}, (
            "the runner must reuse the engine's own snapshot, not the caller's mutable set object"
        )

    def test_no_extenders_means_the_runner_receives_none_not_an_empty_set(self) -> None:
        PluginLoader().load_matching("compute_framework", "*python_dict*")
        register_plugin(_StrictWiringFeatureGroup)

        session = mlodaAPI.prepare(
            [Feature(_STRICT_WIRING_FEAT)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector().set_strict_mode("strict"),
        )

        recorder = _RecordingRunner()
        session._enter_runner_context(
            cast(ExecutionOrchestrator, recorder),
            {ParallelizationMode.SYNC},
            None,
            run_context=RunContext(),
        )

        assert recorder.received_extenders is None, "with no extenders the runner must receive None, not set()"
