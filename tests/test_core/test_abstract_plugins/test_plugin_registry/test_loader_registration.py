"""Failing tests for the loader -> registry discovery funnel (issue #526, work item 3).

Contract: PluginLoader loads feed the default PluginRegistry. Every FeatureGroup,
ComputeFramework, or Extender subclass DEFINED in a loaded module (cls.__module__
equals the loaded module name) is registered exactly once under its defining module,
with source="loader". Registration is idempotent, happens on the sys.modules cache
path too (import-order and import-history independent), and never picks up ad-hoc
subclasses defined outside loaded plugin modules.

All scopes used here import without optional dependencies (python_dict and sqlite).
The autouse conftest fixture restores the default registry around every test, so
clearing it inside a test is safe.
"""

import sys
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register
from mloda.user import PluginLoader

PYTHON_DICT_FRAMEWORK_MODULE = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework"

_PLUGIN_BASE_TYPES: tuple[type[Any], ...] = (FeatureGroup, ComputeFramework, Extender)


def _default_key(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _python_dict_framework_cls() -> type[Any]:
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework,
    )

    return PythonDictFramework


def _sqlite_framework_cls() -> type[Any]:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

    return SqliteFramework


def _load_python_dict_compute_scope() -> PluginLoader:
    """Small, dependency-free scope: the python_dict compute framework files only."""
    loader = PluginLoader()
    loader.load_matching("compute_framework", "*python_dict*")
    return loader


class TestLoaderRegistersLoadedPlugins:
    def test_loader_registers_classes_defined_in_loaded_modules(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        assert registry.is_registered(python_dict_framework), (
            "PluginLoader.load_matching must register plugin classes defined in loaded modules "
            "in the default PluginRegistry"
        )

        module = sys.modules[PYTHON_DICT_FRAMEWORK_MODULE]
        defined_plugin_classes = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, _PLUGIN_BASE_TYPES) and obj.__module__ == module.__name__
        ]
        assert defined_plugin_classes, "sanity: the loaded module defines at least one plugin class"
        for cls in defined_plugin_classes:
            assert registry.is_registered(cls), f"{cls!r} is defined in a loaded module but is not registered"

    def test_loader_entries_carry_loader_provenance(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        key = _default_key(python_dict_framework)
        assert key in registry.snapshot(), f"loader must register '{key}' in the default registry"

        entry = registry.get_entry(key)
        assert entry.cls is python_dict_framework
        assert entry.source == "loader"
        assert entry.source_module == python_dict_framework.__module__


class TestLoaderRegistrationKeying:
    def test_imported_classes_register_once_under_their_defining_module(self) -> None:
        """A class imported into other loaded modules must not get a second entry there.

        Both mloda_plugins.feature_group.experimental.text_cleaning.python_dict and
        mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict
        import PythonDictFramework; only the defining module produces an entry.
        """
        registry = PluginRegistry.default()
        registry.clear()

        loader = PluginLoader()
        loader.load_matching("compute_framework", "*python_dict*")
        loader.load_matching("feature_group", "*python_dict*")

        python_dict_framework = _python_dict_framework_cls()
        entries = registry.snapshot()
        keys_for_class = [key for key, entry in entries.items() if entry.cls is python_dict_framework]
        assert keys_for_class == [_default_key(python_dict_framework)], (
            "PythonDictFramework must be registered exactly once, keyed by its own defining module, "
            f"got keys: {keys_for_class}"
        )

        for entry in entries.values():
            assert entry.source_module == entry.cls.__module__
            assert entry.name == _default_key(entry.cls)


class TestLoaderRegistrationIdempotency:
    def test_repeated_load_registers_nothing_new(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()
        first_snapshot = registry.snapshot()
        assert first_snapshot, "loader run must populate the cleared default registry"

        _load_python_dict_compute_scope()
        assert registry.snapshot() == first_snapshot, "loading the same scope twice must not add or change entries"

    def test_load_of_already_imported_modules_reregisters(self) -> None:
        """Registration must also happen on the sys.modules cache path of _load_plugin."""
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()
        first_snapshot = registry.snapshot()
        assert first_snapshot, "loader run must populate the cleared default registry"
        assert PYTHON_DICT_FRAMEWORK_MODULE in sys.modules, "sanity: module is cached in sys.modules"

        registry.clear()
        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        assert registry.is_registered(python_dict_framework), (
            "loading a scope whose modules are already in sys.modules must still register the plugin classes"
        )
        assert registry.snapshot() == first_snapshot


class TestLoaderRegistrationOrderIndependence:
    def test_load_order_does_not_change_registry_contents(self) -> None:
        registry = PluginRegistry.default()

        registry.clear()
        loader_ab = PluginLoader()
        loader_ab.load_matching("compute_framework", "*python_dict*")
        loader_ab.load_matching("compute_framework", "*sqlite*")
        result_ab = registry.list_registered(ComputeFramework)

        registry.clear()
        loader_ba = PluginLoader()
        loader_ba.load_matching("compute_framework", "*sqlite*")
        loader_ba.load_matching("compute_framework", "*python_dict*")
        result_ba = registry.list_registered(ComputeFramework)

        assert result_ab, "loading two compute framework scopes must register compute frameworks"
        assert _python_dict_framework_cls() in result_ab
        assert _sqlite_framework_cls() in result_ab
        assert result_ab == result_ba, "registry contents must not depend on load order"


class TestLoaderIgnoresTransientSubclasses:
    def test_local_subclass_stays_out_but_explicit_register_opts_in(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _AdHocFeatureGroup(FeatureGroup):
            """Simulates a test double or notebook-cell class; must never auto-register."""

        _load_python_dict_compute_scope()

        assert not registry.is_registered(_AdHocFeatureGroup), (
            "ad-hoc subclasses defined outside loaded plugin modules must stay out of the registry"
        )

        key = register(_AdHocFeatureGroup)
        assert registry.is_registered(_AdHocFeatureGroup), "explicit register() must opt a class in"
        assert registry.get_entry(key).source == "manual"

        assert registry.is_registered(_python_dict_framework_cls()), (
            "loader-run plugins must be registered while ad-hoc subclasses are not"
        )
