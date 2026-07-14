"""ApiInputDataFeature must stay discoverable through the plugin registry (issue #726).

ApiInputDataFeature moved from mloda_plugins to mloda.core. PluginLoader scans the
mloda_plugins base package only, and register_module_plugins registers a class under the
module that DEFINES it, so nothing in the loader funnel registers a core-defined plugin
class any more. Consequences pinned here: strict mode drops the class (an api_data run then
fails to resolve its features), warn mode names a first-party class as unregistered, and the
registered-only enumeration/docs lose it.

Parallel-safety: assertions are membership-based, the api_data key and columns are unique to
this module, and the autouse conftest fixture restores the default registry (and the warn-mode
dedup set) around every test, so clearing the registry here is safe.
"""

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, PluginRegistryEntry
from mloda.core.api.plugin_docs import get_feature_group_docs, list_registered
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.provider import ApiInputDataFeature
from mloda.user import PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

ENV_VAR = "MLODA_PLUGIN_REGISTRY_STRICT"

API_KEY = "ApiInputRegistryProbeSource"
ID_COLUMN = "api_input_registry_probe_id"
VALUE_COLUMN = "api_input_registry_probe_value"
API_DATA: dict[str, dict[str, Any]] = {API_KEY: {ID_COLUMN: [1, 2, 3], VALUE_COLUMN: ["a", "b", "c"]}}

IDENTIFIER = f"{ApiInputDataFeature.__module__}:{ApiInputDataFeature.__qualname__}"


def _loaded_registry() -> PluginRegistry:
    """Default registry holding exactly what one full PluginLoader run discovers."""
    registry = PluginRegistry.default()
    registry.clear()
    PluginLoader.all()
    return registry


def _api_input_data_feature_entries(registry: PluginRegistry) -> list[PluginRegistryEntry]:
    return [entry for entry in registry.snapshot().values() if entry.cls is ApiInputDataFeature]


def _column_values(frame: Any) -> list[Any]:
    """Extract the probe column, tolerant of a columnar dict or a list of row dicts."""
    if isinstance(frame, dict):
        return list(frame[VALUE_COLUMN])
    return [row[VALUE_COLUMN] for row in frame]


class TestApiInputDataFeatureIsLoaderRegistered:
    def test_plugin_loader_registers_api_input_data_feature(self) -> None:
        registry = _loaded_registry()
        assert registry.is_registered(ApiInputDataFeature), (
            "PluginLoader.all() must register ApiInputDataFeature in the default PluginRegistry. "
            f"It is defined in '{ApiInputDataFeature.__module__}', which the loader never scans, so the "
            "class silently left the registry when it moved out of mloda_plugins."
        )

    def test_registry_entry_carries_loader_provenance(self) -> None:
        registry = _loaded_registry()
        entries = _api_input_data_feature_entries(registry)
        assert len(entries) == 1, f"ApiInputDataFeature must hold exactly one registry entry, got {len(entries)}"

        entry = entries[0]
        assert entry.name == IDENTIFIER, "the entry must use the default '<module>:<qualname>' key"
        assert entry.plugin_type is FeatureGroup
        assert entry.source_module == ApiInputDataFeature.__module__
        assert entry.source == "loader", (
            "ApiInputDataFeature ships with mloda, so its provenance must be the loader, like every "
            f"bundled plugin; got source '{entry.source}'"
        )


class TestApiInputDataFeatureEnumeration:
    def test_list_registered_includes_api_input_data_feature(self) -> None:
        _loaded_registry()
        assert ApiInputDataFeature in list_registered(FeatureGroup), (
            "list_registered(FeatureGroup) must list ApiInputDataFeature after PluginLoader.all()"
        )

    def test_registered_only_docs_include_api_input_data_feature(self) -> None:
        _loaded_registry()
        names = [fg.name for fg in get_feature_group_docs(registered_only=True)]
        assert "ApiInputDataFeature" in names, (
            "get_feature_group_docs(registered_only=True) must document ApiInputDataFeature; an unregistered "
            "first-party feature group disappears from every registered-only listing"
        )


class TestApiInputDataFeatureUnderStrictMode:
    def test_strict_mode_api_data_run_resolves_the_feature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End to end: strict mode must not filter ApiInputDataFeature out of an api_data run."""
        monkeypatch.setenv(ENV_VAR, "strict")
        _loaded_registry()

        result = mloda.run_all(
            [VALUE_COLUMN],
            compute_frameworks={PythonDictFramework},
            api_data=API_DATA,
        )

        assert len(result) == 1, f"Expected exactly one result frame, got: {result}"
        assert _column_values(result[0]) == ["a", "b", "c"]


class TestApiInputDataFeatureUnderWarnMode:
    def test_warn_mode_does_not_flag_api_input_data_feature(self, caplog: pytest.LogCaptureFixture) -> None:
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
            "warn mode must not report ApiInputDataFeature as unregistered: the warning is permanent and "
            "unfixable for users, since the class ships with mloda"
        )
