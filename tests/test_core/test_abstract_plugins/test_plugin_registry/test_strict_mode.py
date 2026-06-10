"""Failing tests for the tri-state strict parameter (issue #526, work item 5).

Contract: PluginCollector carries a strict_mode of "off", "warn", or "strict",
defaulting from the MLODA_PLUGIN_REGISTRY_STRICT env var ("off" when unset,
ValueError on invalid values). PreFilterPlugins consults the mode during
resolution: "strict" keeps only FeatureGroups registered in the default
PluginRegistry, "warn" keeps everything but logs unregistered classes,
"off" keeps today's behavior. The engine honors the env var even when no
PluginCollector is passed.

Parallel-safety: other tests define many FeatureGroup subclasses, so all
assertions are membership or absence of this module's own doubles, never
global counts. Registry writes are restored by the autouse conftest fixture;
env writes go through monkeypatch only.
"""

import logging

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import register
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

ENV_VAR = "MLODA_PLUGIN_REGISTRY_STRICT"


class _StrictUnregisteredFG(FeatureGroup):
    """Local double that is never registered in the default registry."""


class _StrictRegisteredFG(FeatureGroup):
    """Local double that tests register explicitly in the default registry."""


def _cfws() -> set[type[ComputeFramework]]:
    return {PythonDictFramework}


class TestSetStrictModeApi:
    def test_set_strict_mode_is_chainable(self) -> None:
        collector = PluginCollector()
        assert collector.set_strict_mode("warn") is collector

    @pytest.mark.parametrize("mode", ["off", "warn", "strict"])
    def test_set_strict_mode_accepts_valid_values(self, mode: str) -> None:
        collector = PluginCollector().set_strict_mode(mode)
        assert collector.strict_mode == mode

    @pytest.mark.parametrize("mode", ["on", "OFF", "Strict", "", "true", "1"])
    def test_set_strict_mode_rejects_invalid_values(self, mode: str) -> None:
        with pytest.raises(ValueError):
            PluginCollector().set_strict_mode(mode)

    def test_default_is_off_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert PluginCollector().strict_mode == "off"


class TestEnvVarDefault:
    @pytest.mark.parametrize("mode", ["warn", "strict"])
    def test_env_var_sets_default_mode(self, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        monkeypatch.setenv(ENV_VAR, mode)
        assert PluginCollector().strict_mode == mode

    def test_invalid_env_value_raises_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "bogus")
        with pytest.raises(ValueError):
            PluginCollector()

    def test_explicit_set_strict_mode_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "strict")
        collector = PluginCollector().set_strict_mode("off")
        assert collector.strict_mode == "off"


class TestEngineStrictModeOff:
    def test_off_mode_keeps_unregistered_feature_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Non-regression guard: today's behavior, unregistered subclasses stay accessible.
        monkeypatch.delenv(ENV_VAR, raising=False)
        accessible = PreFilterPlugins(_cfws(), PluginCollector()).get_accessible_plugins()
        assert _StrictUnregisteredFG in accessible


class TestEngineStrictModeStrict:
    def test_strict_mode_keeps_only_registered_feature_groups(self) -> None:
        register(_StrictRegisteredFG)
        collector = PluginCollector().set_strict_mode("strict")
        accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()
        assert _StrictRegisteredFG in accessible
        assert _StrictUnregisteredFG not in accessible


class TestEngineStrictModeWarn:
    def test_warn_mode_resolution_matches_off_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        off_accessible = PreFilterPlugins(_cfws(), PluginCollector()).get_accessible_plugins()
        warn_collector = PluginCollector().set_strict_mode("warn")
        warn_accessible = PreFilterPlugins(_cfws(), warn_collector).get_accessible_plugins()
        assert _StrictUnregisteredFG in warn_accessible
        assert set(warn_accessible.keys()) == set(off_accessible.keys())

    def test_warn_mode_logs_unregistered_class(self, caplog: pytest.LogCaptureFixture) -> None:
        collector = PluginCollector().set_strict_mode("warn")
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), collector)
        matching = [
            record
            for record in caplog.records
            if _StrictUnregisteredFG.__name__ in record.getMessage() and "not registered" in record.getMessage()
        ]
        assert matching, "warn mode must log a warning naming the unregistered class and saying 'not registered'"


class TestEnvWithoutCollector:
    def test_env_strict_applies_without_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "strict")
        register(_StrictRegisteredFG)
        accessible = PreFilterPlugins(_cfws(), None).get_accessible_plugins()
        assert _StrictRegisteredFG in accessible
        assert _StrictUnregisteredFG not in accessible

    def test_env_unset_keeps_unregistered_without_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Non-regression guard: today's behavior without collector and without env var.
        monkeypatch.delenv(ENV_VAR, raising=False)
        accessible = PreFilterPlugins(_cfws(), None).get_accessible_plugins()
        assert _StrictUnregisteredFG in accessible
