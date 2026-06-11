"""Tests for registry instance injection via PluginCollector (issue #526, cycle 5).

Contract: PluginCollector gains ``set_registry(registry)`` (chainable, like
``set_strict_mode``) and a ``registry`` attribute defaulting to ``None``.
``mloda.core.prepare.accessible_plugins`` gains ``registry_for(plugin_collector)``
returning the collector's injected registry when set, else
``PluginRegistry.default()``. PreFilterPlugins strict/warn checks consult the
injected registry instead of hard-coding the default one.

All tests are written to FAIL until the feature exists. Registry writes to the
default registry are restored by the autouse conftest fixture; injected
registries are fresh local instances, so the tests stay xdist parallel-safe.
"""

import logging

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _InjectionInjectedOnlyFG(FeatureGroup):
    """Local double registered ONLY in a fresh injected registry, never in the default one."""


class _InjectionDefaultOnlyFG(FeatureGroup):
    """Local double registered ONLY in the default registry."""


def _cfws() -> set[type[ComputeFramework]]:
    return {PythonDictFramework}


def _qualid(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class TestSetRegistryApi:
    def test_registry_attribute_defaults_to_none(self) -> None:
        assert PluginCollector().registry is None

    def test_set_registry_is_chainable_and_stores_the_instance(self) -> None:
        fresh = PluginRegistry()
        collector = PluginCollector()
        assert collector.set_registry(fresh) is collector
        assert collector.registry is fresh


class TestRegistryFor:
    def test_registry_for_returns_default_without_injection(self) -> None:
        from mloda.core.prepare.accessible_plugins import registry_for

        assert registry_for(None) is PluginRegistry.default()
        assert registry_for(PluginCollector()) is PluginRegistry.default()

    def test_registry_for_returns_injected_registry(self) -> None:
        from mloda.core.prepare.accessible_plugins import registry_for

        fresh = PluginRegistry()
        collector = PluginCollector().set_registry(fresh)
        assert registry_for(collector) is fresh


class TestStrictModeConsultsInjectedRegistry:
    def test_strict_keeps_injected_only_fg_and_drops_default_only_fg(self) -> None:
        """The injected registry, not the hard-coded default, decides strict survival.

        With the old hard-coding of PluginRegistry.default(), the outcome would be
        exactly inverted: the default-only class would survive and the
        injected-only class would be dropped.
        """
        fresh = PluginRegistry()
        fresh.register(_InjectionInjectedOnlyFG)
        register_plugin(_InjectionDefaultOnlyFG)

        collector = PluginCollector().set_strict_mode("strict").set_registry(fresh)
        accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()

        assert _InjectionInjectedOnlyFG in accessible, (
            "a FeatureGroup registered in the injected registry must survive strict filtering"
        )
        assert _InjectionDefaultOnlyFG not in accessible, (
            "a FeatureGroup registered only in the default registry must be dropped when "
            "a different registry is injected via the PluginCollector"
        )

    def test_warn_mode_consults_injected_registry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warn mode must not flag a class that IS registered in the injected registry."""
        fresh = PluginRegistry()
        fresh.register(_InjectionInjectedOnlyFG)

        collector = PluginCollector().set_strict_mode("warn").set_registry(fresh)
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), collector)

        message = " ".join(
            rec.getMessage() for rec in caplog.records if rec.name == "mloda.core.prepare.accessible_plugins"
        )
        assert _qualid(_InjectionInjectedOnlyFG) not in message, (
            "warn mode must consult the injected registry; a class registered there is not unregistered"
        )
        assert _qualid(_InjectionDefaultOnlyFG) in message, (
            "sanity: a class registered nowhere relevant must still be flagged by warn mode"
        )
