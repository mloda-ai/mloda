"""Persona-export matrix for the plugin registry public API (pre-merge tightening).

Contract: the module-level registration function is named register_plugin.
mloda.user exports register_plugin, list_registered, and PluginRegistryCollisionError,
but NOT PluginRegistry. mloda.provider exports register_plugin and
PluginRegistryCollisionError. mloda.steward exports PluginRegistry and
list_registered. Every export is the identical object from its core module
and is listed in the namespace's __all__.
"""

from types import ModuleType

import pytest

import mloda.core.abstract_plugins.plugin_registry.plugin_registry as core_registry
import mloda.core.api.plugin_docs as plugin_docs
import mloda.provider
import mloda.steward
import mloda.user

_NAMESPACES: dict[str, ModuleType] = {
    "user": mloda.user,
    "provider": mloda.provider,
    "steward": mloda.steward,
}

_SOURCES: dict[str, ModuleType] = {
    "core_registry": core_registry,
    "plugin_docs": plugin_docs,
}

# (namespace, symbol, source module holding the canonical object)
EXPORT_MATRIX: list[tuple[str, str, str]] = [
    ("user", "register_plugin", "core_registry"),
    ("user", "list_registered", "plugin_docs"),
    ("user", "PluginRegistryCollisionError", "core_registry"),
    ("provider", "register_plugin", "core_registry"),
    ("provider", "PluginRegistryCollisionError", "core_registry"),
    ("steward", "PluginRegistry", "core_registry"),
    ("steward", "list_registered", "plugin_docs"),
]

_MATRIX_IDS = [f"{namespace}-{symbol}" for namespace, symbol, _source in EXPORT_MATRIX]


class TestPersonaExportMatrix:
    @pytest.mark.parametrize(("namespace_name", "symbol", "source_name"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_listed_in_namespace_all(self, namespace_name: str, symbol: str, source_name: str) -> None:
        namespace = _NAMESPACES[namespace_name]
        assert symbol in namespace.__all__, f"mloda.{namespace_name} must list '{symbol}' in __all__"

    @pytest.mark.parametrize(("namespace_name", "symbol", "source_name"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_is_identical_to_core_object(self, namespace_name: str, symbol: str, source_name: str) -> None:
        namespace = _NAMESPACES[namespace_name]
        source = _SOURCES[source_name]
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert hasattr(namespace, symbol), f"mloda.{namespace_name} must expose '{symbol}'"
        assert getattr(namespace, symbol) is getattr(source, symbol), (
            f"mloda.{namespace_name}.{symbol} must be the identical object from {source.__name__}"
        )


class TestUserDoesNotExportPluginRegistry:
    def test_plugin_registry_absent_from_user_all(self) -> None:
        assert "PluginRegistry" not in mloda.user.__all__, (
            "mloda.user must not list PluginRegistry in __all__; it moves to mloda.steward"
        )

    def test_plugin_registry_not_a_user_attribute(self) -> None:
        assert not hasattr(mloda.user, "PluginRegistry"), (
            "mloda.user must not expose a PluginRegistry attribute; stewards use mloda.steward.PluginRegistry"
        )
