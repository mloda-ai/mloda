"""Tests for resolve_plugin_version, the owning-distribution lookup used to
populate HookContext.plugin_version.
"""

import importlib.metadata

from mloda.core.abstract_plugins.plugin_version import resolve_plugin_version


class TestResolvePluginVersion:
    """resolve_plugin_version resolves the installed distribution version owning a module."""

    def test_resolves_mloda_module_to_installed_mloda_version(self) -> None:
        result = resolve_plugin_version("mloda.core.abstract_plugins.function_extender")

        assert result is not None
        assert result != ""
        assert result == importlib.metadata.version("mloda")

    def test_returns_none_for_unknown_top_level_package(self) -> None:
        result = resolve_plugin_version("definitely_not_a_real_top_level_package_xyz.submodule")

        assert result is None

    def test_repeated_calls_with_different_modules_resolve_independently(self) -> None:
        first = resolve_plugin_version("mloda.core.abstract_plugins.function_extender")
        second = resolve_plugin_version("definitely_not_a_real_top_level_package_xyz.submodule")

        assert first == importlib.metadata.version("mloda")
        assert second is None
