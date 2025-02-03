from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader


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
        assert len(loaded_modules) == 28  # If a plugin is added or removed, this number will change.
        cp_loaded_modules = plugin_loader.list_loaded_modules("compute_framework")
        assert len(cp_loaded_modules) < len(loaded_modules)

    def test_display_graph(self) -> None:
        plugin_loader = PluginLoader()
        plugin_loader.load_all_plugins()
        result = plugin_loader.display_plugin_graph("compute_framework")
        assert "mloda_plugins.compute_framework.base_implementations.pandas.dataframe -> []" in result
        for res in result:
            assert "feature_group" not in res
