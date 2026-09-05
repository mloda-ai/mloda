"""Pins Python 3.10's GenericAlias isinstance(obj, type) quirk in the plugin registry.

On Python 3.10 alone, isinstance(tuple[str, ...], type) is True for a types.GenericAlias, so a
module-level type alias attribute passes the registry's isinstance(obj, type) filter and then
issubclass() raises TypeError instead of being skipped as a non-class attribute.
"""

import gc
import sys
import types
from typing import Any

import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    _resolve_plugin_type,
    register_module_plugins,
)
from mloda.user import PluginLoader

SYNTHETIC_MODULE_NAME = "synthetic_generic_alias_plugin_module"


def _default_key(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _build_synthetic_module() -> tuple[types.ModuleType, type[Any]]:
    module = types.ModuleType(SYNTHETIC_MODULE_NAME)

    class _GenericAliasProbeFG(FeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return {}

    _GenericAliasProbeFG.__module__ = module.__name__
    setattr(module, "_GenericAliasProbeFG", _GenericAliasProbeFG)
    setattr(module, "Alias", tuple[tuple[str, str | None], ...])
    return module, _GenericAliasProbeFG


class TestRegisterModulePluginsSkipsGenericAliasAttribute:
    def test_register_module_plugins_skips_generic_alias_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        module, cls = _build_synthetic_module()
        monkeypatch.setitem(sys.modules, module.__name__, module)

        try:
            keys = register_module_plugins(module)
            assert keys == [_default_key(cls)], (
                "register_module_plugins must register only the concrete feature group, skipping the "
                "module-level generic alias attribute"
            )
        finally:
            registry.clear()
            delattr(module, "_GenericAliasProbeFG")
            delattr(module, "Alias")
            del cls, module
            gc.collect()


class TestResolvePluginTypeRejectsGenericAlias:
    def test_resolve_plugin_type_rejects_generic_alias_with_value_error(self) -> None:
        with pytest.raises(ValueError):
            _resolve_plugin_type(tuple[str, ...])


class TestIsClassHelper:
    def test_is_class_rejects_generic_alias_and_accepts_classes(self) -> None:
        from mloda.core.abstract_plugins.plugin_registry.plugin_registry import _is_class

        assert _is_class(tuple[str, ...]) is False
        assert _is_class(list[int]) is False
        assert _is_class(int) is True
        assert _is_class(FeatureGroup) is True
        assert _is_class(object()) is False


class TestEntryPointManifestRejectsGenericAlias:
    def test_entry_point_manifest_reports_generic_alias_as_non_class(self) -> None:
        loader = PluginLoader()
        with pytest.raises(TypeError, match="non-class item"):
            loader._register_manifest("synthetic_label", "mloda.feature_groups", FeatureGroup, (tuple[str, ...],))
