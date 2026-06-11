"""Tests for the PluginSource provenance enum.

Contract: PluginSource is a public str-subclassing enum (MANUAL="manual",
LOADER="loader", ENTRY_POINT="entry_point"). PluginRegistry.register()
normalizes plain strings to enum members, stores PluginSource.MANUAL by
default, and rejects unknown sources with a ValueError that lists the valid
values. register_module_plugins keeps stamping loader provenance, now as
PluginSource.LOADER.
"""

import enum
import importlib

import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    PluginSource,
    register_module_plugins,
)

PYTHON_DICT_FRAMEWORK_MODULE = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework"


class _SourceTestFGA(FeatureGroup):
    pass


class _SourceTestFGB(FeatureGroup):
    pass


class TestPluginSourceEnum:
    def test_plugin_source_is_str_enum(self) -> None:
        assert issubclass(PluginSource, str)
        assert issubclass(PluginSource, enum.Enum)

    def test_plugin_source_members_and_values(self) -> None:
        assert PluginSource.MANUAL == "manual"
        assert PluginSource.LOADER == "loader"
        assert PluginSource.ENTRY_POINT == "entry_point"

    def test_plugin_source_keeps_string_comparisons_working(self) -> None:
        """Existing call sites compare entry.source against bare strings; str subclassing keeps that True."""
        assert PluginSource.LOADER == "loader"
        assert isinstance(PluginSource.LOADER, str)


class TestRegisterSourceNormalization:
    def test_register_default_source_is_manual_enum_member(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_SourceTestFGA)
        entry = reg.get_entry(key)
        assert entry.source is PluginSource.MANUAL

    def test_register_plain_string_is_normalized_to_enum_member(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_SourceTestFGA, source="loader")
        entry = reg.get_entry(key)
        assert entry.source is PluginSource.LOADER
        assert isinstance(entry.source, PluginSource), "a plain-string source must be stored as a PluginSource member"

    def test_register_enum_member_is_stored_as_is(self) -> None:
        reg = PluginRegistry()
        key = reg.register(_SourceTestFGA, source=PluginSource.ENTRY_POINT)
        assert reg.get_entry(key).source is PluginSource.ENTRY_POINT

    def test_register_unknown_source_raises_value_error(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.register(_SourceTestFGB, source="invented")

    def test_register_unknown_source_error_lists_valid_values(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError) as exc_info:
            reg.register(_SourceTestFGB, source="invented")
        message = str(exc_info.value)
        assert "manual" in message
        assert "loader" in message
        assert "entry_point" in message

    def test_register_unknown_source_does_not_register_the_class(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.register(_SourceTestFGB, source="invented")
        assert reg.is_registered(_SourceTestFGB) is False


class TestRegisterModulePluginsProvenance:
    def test_register_module_plugins_stamps_loader_enum_member(self) -> None:
        """The loader funnel keeps working; its entries now carry PluginSource.LOADER."""
        registry = PluginRegistry.default()
        registry.clear()

        module = importlib.import_module(PYTHON_DICT_FRAMEWORK_MODULE)
        keys = register_module_plugins(module)
        assert keys, "sanity: the python_dict framework module defines at least one plugin class"

        for key in keys:
            entry = registry.get_entry(key)
            assert entry.source is PluginSource.LOADER
            assert entry.source == "loader", "loader provenance must keep comparing equal to the bare string"
