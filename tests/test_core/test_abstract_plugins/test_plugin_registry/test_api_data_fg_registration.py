"""ApiDataFG lives in mloda.core, so the plugin loader never registers it; pins the consequences."""

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, PluginRegistryEntry
from mloda.core.api.plugin_docs import get_feature_group_docs, list_registered
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.provider import ApiDataFG
from mloda.user import PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


ENV_VAR = "MLODA_PLUGIN_REGISTRY_STRICT"

API_KEY = "ApiDataRegistryProbeSource"
ID_COLUMN = "api_data_registry_probe_id"
VALUE_COLUMN = "api_data_registry_probe_value"
API_DATA: dict[str, dict[str, Any]] = {API_KEY: {ID_COLUMN: [1, 2, 3], VALUE_COLUMN: ["a", "b", "c"]}}

IDENTIFIER = f"{ApiDataFG.__module__}:{ApiDataFG.__qualname__}"


def _loaded_registry() -> PluginRegistry:
    """Default registry holding exactly what one full PluginLoader run discovers."""
    registry = PluginRegistry.default()
    registry.clear()
    PluginLoader.all()
    return registry


def _api_data_fg_entries(registry: PluginRegistry) -> list[PluginRegistryEntry]:
    return [entry for entry in registry.snapshot().values() if entry.cls is ApiDataFG]


def _column_values(frame: Any) -> list[Any]:
    """Extract the probe column, tolerant of a columnar dict or a list of row dicts."""
    if isinstance(frame, dict):
        return list(frame[VALUE_COLUMN])
    return [row[VALUE_COLUMN] for row in frame]


class TestApiDataFGIsLoaderRegistered:
    def test_plugin_loader_registers_api_data_fg(self) -> None:
        registry = _loaded_registry()
        assert registry.is_registered(ApiDataFG), (
            "PluginLoader.all() must register ApiDataFG in the default PluginRegistry. "
            f"It is defined in '{ApiDataFG.__module__}', which the loader never scans, so the "
            "class silently left the registry when it moved out of mloda_plugins."
        )

    def test_registry_entry_carries_loader_provenance(self) -> None:
        registry = _loaded_registry()
        entries = _api_data_fg_entries(registry)
        assert len(entries) == 1, f"ApiDataFG must hold exactly one registry entry, got {len(entries)}"

        entry = entries[0]
        assert entry.name == IDENTIFIER, "the entry must use the default '<module>:<qualname>' key"
        assert entry.plugin_type is FeatureGroup
        assert entry.source_module == ApiDataFG.__module__
        assert entry.source == "loader", (
            "ApiDataFG ships with mloda, so its provenance must be the loader, like every "
            f"bundled plugin; got source '{entry.source}'"
        )


class TestApiDataFGEnumeration:
    def test_list_registered_includes_api_data_fg(self) -> None:
        _loaded_registry()
        assert ApiDataFG in list_registered(FeatureGroup), (
            "list_registered(FeatureGroup) must list ApiDataFG after PluginLoader.all()"
        )

    def test_registered_only_docs_include_api_data_fg(self) -> None:
        _loaded_registry()
        names = [fg.name for fg in get_feature_group_docs(registered_only=True)]
        assert "ApiDataFG" in names, (
            "get_feature_group_docs(registered_only=True) must document ApiDataFG; an unregistered "
            "first-party feature group disappears from every registered-only listing"
        )


class TestApiDataFGUnderStrictMode:
    def test_strict_mode_api_data_run_resolves_the_feature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End to end: strict mode must not filter ApiDataFG out of an api_data run."""
        monkeypatch.setenv(ENV_VAR, "strict")
        _loaded_registry()

        result = mloda.run_all(
            [VALUE_COLUMN],
            compute_frameworks=[PythonDictFramework],
            api_data=API_DATA,
        )

        assert len(result) == 1, f"Expected exactly one result frame, got: {result}"
        assert _column_values(result[0]) == ["a", "b", "c"]


class TestApiDataFGUnderWarnMode:
    def test_warn_mode_does_not_flag_api_data_fg(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bundled, first-party feature group must never show up in the unregistered warning."""
        _loaded_registry()

        collector = PluginCollector().set_strict_mode("warn")
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins({PythonDictFramework}, collector)

        message = " ".join(
            record.getMessage()
            for record in caplog.records
            if record.name == "mloda.core.prepare.accessible_plugins" and "not registered" in record.getMessage()
        )
        assert IDENTIFIER not in message, (
            "warn mode must not report ApiDataFG as unregistered: the warning is permanent and "
            "unfixable for users, since the class ships with mloda"
        )
