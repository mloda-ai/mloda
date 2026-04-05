from unittest.mock import patch

from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    _collect_filtered_subclasses,
    get_all_filtered_subclasses,
)
from mloda.user import PluginLoader


class TestPluginLoader:
    def test_plugin_loader_init(self) -> None:
        plugin_loader = PluginLoader()
        assert plugin_loader.base_package == "mloda_plugins"
        assert plugin_loader.plugins == {}

    def test_load_group(self) -> None:
        plugin_loader = PluginLoader()
        plugin_loader.load_group("feature_group")
        assert "mloda_plugins.feature_group.input_data.read_files.parquet" in plugin_loader.plugins
        assert "mloda_plugins.feature_group.input_data.read_files.csv" in plugin_loader.plugins

    def test_load_all_groups(self) -> None:
        plugin_loader = PluginLoader()
        plugin_loader.load_all_plugins()
        # This test ensures that not accidentally a plugin is added or removed
        loaded_modules = plugin_loader.list_loaded_modules()
        cp_loaded_modules = plugin_loader.list_loaded_modules("compute_framework")
        assert len(cp_loaded_modules) < len(loaded_modules)

    def test_display_graph(self) -> None:
        plugin_loader = PluginLoader()
        plugin_loader.load_all_plugins()
        result = plugin_loader.display_plugin_graph("compute_framework")
        assert "mloda_plugins.compute_framework.base_implementations.pandas.dataframe -> []" in result
        for res in result:
            assert "feature_group" not in res

    def test_disable_auto_load_adds_to_disabled_groups(self) -> None:
        PluginLoader._disabled_groups.discard("_test_group")
        PluginLoader.disable_auto_load("_test_group")
        assert "_test_group" in PluginLoader._disabled_groups
        PluginLoader._disabled_groups.discard("_test_group")

    def test_disable_auto_load_suppresses_lazy_load(self) -> None:
        """When auto-load is disabled for a group, get_all_filtered_subclasses returns empty without loading."""
        from unittest.mock import MagicMock

        from mloda_plugins.feature_group.input_data.read_file import ReadFile

        PluginLoader.disable_auto_load("feature_group/input_data/read_files")
        mock_load = MagicMock()
        try:
            with patch(
                "mloda.core.abstract_plugins.components.input_data.base_input_data._collect_filtered_subclasses",
                return_value=[],
            ):
                with patch(
                    "mloda.core.abstract_plugins.plugin_loader.plugin_loader.PluginLoader.load_group",
                    mock_load,
                ):
                    result = get_all_filtered_subclasses(ReadFile, ReadFile)
            assert result == []
            mock_load.assert_not_called()
        finally:
            PluginLoader._disabled_groups.discard("feature_group/input_data/read_files")

    def test_auto_load_triggers_when_subclasses_empty(self) -> None:
        """Auto-load fires load_group when _collect_filtered_subclasses returns empty."""
        from unittest.mock import MagicMock

        from mloda_plugins.feature_group.input_data.read_file import ReadFile

        PluginLoader._disabled_groups.discard("feature_group/input_data/read_files")

        mock_load = MagicMock()

        with patch(
            "mloda.core.abstract_plugins.components.input_data.base_input_data._collect_filtered_subclasses",
            return_value=[],
        ):
            with patch(
                "mloda.core.abstract_plugins.plugin_loader.plugin_loader.PluginLoader.load_group",
                mock_load,
            ):
                get_all_filtered_subclasses(ReadFile, ReadFile)

        mock_load.assert_called_once_with("feature_group/input_data/read_files")

    def test_load_nested_group_builds_correct_module_path(self) -> None:
        """Nested group paths like 'feature_group/input_data/read_files' produce correct module names."""
        plugin_loader = PluginLoader()
        plugin_loader.load_group("feature_group/input_data/read_files")
        assert "mloda_plugins.feature_group.input_data.read_files.csv" in plugin_loader.plugins
        assert "mloda_plugins.feature_group.input_data.read_files.parquet" in plugin_loader.plugins

    def test_load_matching_only_loads_transformer_files(self) -> None:
        """load_matching with '*transformer*' loads only transformer files, not dataframe/filter/merge."""
        from unittest.mock import MagicMock

        plugin_loader = PluginLoader()
        mock_load_plugin = MagicMock()

        with patch.object(plugin_loader, "_load_plugin", mock_load_plugin):
            plugin_loader.load_matching("compute_framework", "*transformer*")

        loaded = [c.args[0] for c in mock_load_plugin.call_args_list]
        assert all("transformer" in m for m in loaded), f"Non-transformer file loaded: {loaded}"
        assert any("transformer" in m for m in loaded), "No transformer files were loaded"
        assert not any("dataframe" in m for m in loaded), f"Dataframe file loaded unexpectedly: {loaded}"
