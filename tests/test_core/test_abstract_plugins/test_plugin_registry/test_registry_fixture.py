"""Tests for the isolated_plugin_registry pytest fixture (issue #526, work item 2).

The fixture snapshots PluginRegistry.default(), yields it, and restores the
snapshot on teardown so tests cannot leak registrations into each other.
"""

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.fixtures import isolated_plugin_registry
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry


class _FixtureThrowawayFG(FeatureGroup):
    pass


class _FixtureSmokeFG(FeatureGroup):
    pass


def test_isolated_plugin_registry_is_a_pytest_fixture() -> None:
    # pytest >= 8.4 exposes _fixture_function_marker on the FixtureFunctionDefinition,
    # older pytest set _pytestfixturefunction on the function itself.
    assert hasattr(isolated_plugin_registry, "_fixture_function_marker") or hasattr(
        isolated_plugin_registry, "_pytestfixturefunction"
    )
    assert hasattr(isolated_plugin_registry, "__wrapped__")


def test_isolated_plugin_registry_yields_default_and_restores_on_teardown() -> None:
    default = PluginRegistry.default()
    snap = default.snapshot()
    try:
        wrapped = getattr(isolated_plugin_registry, "__wrapped__")
        gen = wrapped()
        reg = next(gen)
        assert reg is PluginRegistry.default()
        key = reg.register(_FixtureThrowawayFG, name="_registry_fixture_throwaway")
        assert reg.get(key) is _FixtureThrowawayFG
        next(gen, None)
        assert PluginRegistry.default().get(key) is None
        assert PluginRegistry.default().is_registered(_FixtureThrowawayFG) is False
    finally:
        default.restore(snap)


def test_isolated_plugin_registry_smoke_usage(isolated_plugin_registry: PluginRegistry) -> None:
    key = isolated_plugin_registry.register(_FixtureSmokeFG, name="_registry_fixture_smoke")
    assert isolated_plugin_registry.get(key) is _FixtureSmokeFG
    assert isolated_plugin_registry is PluginRegistry.default()
