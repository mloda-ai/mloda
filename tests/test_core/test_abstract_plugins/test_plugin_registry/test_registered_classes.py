"""Tests for PluginRegistry.registered_classes().

Contract: registered_classes() returns a fresh set[type[Any]] of all
registered classes; empty registry yields an empty set, each class appears
exactly once, and mutating the returned set never touches registry state.
"""

from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry


class _RegisteredClassesFGA(FeatureGroup):
    pass


class _RegisteredClassesFGB(FeatureGroup):
    pass


class _RegisteredClassesFGC(FeatureGroup):
    pass


class _RegisteredClassesCF(ComputeFramework):
    pass


class _RegisteredClassesExtender(Extender):
    pass


class TestRegisteredClasses:
    def test_empty_registry_returns_empty_set(self) -> None:
        reg = PluginRegistry()
        result = reg.registered_classes()
        assert isinstance(result, set)
        assert result == set()

    def test_returns_exactly_the_registered_classes(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegisteredClassesFGA)
        reg.register(_RegisteredClassesFGB)
        reg.register(_RegisteredClassesFGC)
        assert reg.registered_classes() == {
            _RegisteredClassesFGA,
            _RegisteredClassesFGB,
            _RegisteredClassesFGC,
        }

    def test_includes_all_plugin_base_types(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegisteredClassesFGA)
        reg.register(_RegisteredClassesCF)
        reg.register(_RegisteredClassesExtender)
        assert reg.registered_classes() == {
            _RegisteredClassesFGA,
            _RegisteredClassesCF,
            _RegisteredClassesExtender,
        }

    def test_class_appears_exactly_once(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegisteredClassesFGA)
        reg.register(_RegisteredClassesFGA)
        result = reg.registered_classes()
        assert result == {_RegisteredClassesFGA}
        assert len(result) == 1

    def test_unregister_removes_class_from_result(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_RegisteredClassesFGA)
        reg.register(_RegisteredClassesFGB)
        reg.unregister(key)
        assert reg.registered_classes() == {_RegisteredClassesFGB}

    def test_mutating_returned_set_does_not_affect_registry(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegisteredClassesFGA)
        first: set[type[Any]] = reg.registered_classes()
        first.clear()
        first.add(_RegisteredClassesFGB)
        assert reg.registered_classes() == {_RegisteredClassesFGA}
        assert reg.is_registered(_RegisteredClassesFGA) is True
        assert reg.is_registered(_RegisteredClassesFGB) is False

    def test_returns_a_fresh_set_each_call(self) -> None:
        reg = PluginRegistry()
        reg.register(_RegisteredClassesFGA)
        assert reg.registered_classes() is not reg.registered_classes()
