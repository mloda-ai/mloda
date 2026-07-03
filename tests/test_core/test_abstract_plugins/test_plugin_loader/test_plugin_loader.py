from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    _collect_filtered_subclasses,  # noqa: F401
    get_all_filtered_subclasses,
)
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
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

    def test_all_returns_cached_instance(self) -> None:
        """Repeated all() calls return the identical cached PluginLoader instance."""
        first = PluginLoader.all()
        second = PluginLoader.all()
        assert first is second

    def test_all_second_call_skips_load_work(self) -> None:
        """The second all() call reuses the cache and does not re-run the load work."""
        from unittest.mock import MagicMock

        with patch.object(PluginLoader, "load_all_plugins", MagicMock()) as mock_load_all:
            with patch.object(PluginLoader, "load_entry_points", MagicMock()) as mock_entry_points:
                PluginLoader.all()
                PluginLoader.all()
                mock_load_all.assert_called_once()
                mock_entry_points.assert_called_once()

    def test_all_force_reload_rebuilds(self) -> None:
        """all(force_reload=True) rebuilds and returns a different instance than the cached one."""
        first = PluginLoader.all()
        second = PluginLoader.all(force_reload=True)
        assert first is not second

    def test_reset_cache_forces_rebuild(self) -> None:
        """After reset_cache(), the next all() rebuilds a fresh instance."""
        first = PluginLoader.all()
        PluginLoader.reset_cache()
        second = PluginLoader.all()
        assert first is not second

    def test_all_thread_safe_single_build(self) -> None:
        """Concurrent all() calls build the loader exactly once under contention."""
        import threading
        import time
        from unittest.mock import MagicMock

        # Widen the double-checked-locking contention window so threads actually
        # overlap inside the build; without this the mocked build is instantaneous.
        def slow_build(*args: object, **kwargs: object) -> None:
            time.sleep(0.02)

        with patch.object(PluginLoader, "load_all_plugins", MagicMock(side_effect=slow_build)) as mock_load_all:
            with patch.object(PluginLoader, "load_entry_points", MagicMock()) as mock_entry_points:
                barrier = threading.Barrier(10)
                results: list[PluginLoader] = []
                results_lock = threading.Lock()

                def worker() -> None:
                    barrier.wait()
                    loader = PluginLoader.all()
                    with results_lock:
                        results.append(loader)

                threads = [threading.Thread(target=worker) for _ in range(10)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                mock_load_all.assert_called_once()
                mock_entry_points.assert_called_once()
                assert all(loader is results[0] for loader in results)

    def test_all_rebuilds_after_registry_clear(self) -> None:
        """Plain all() must repopulate the default registry after it has been cleared."""
        loader1 = PluginLoader.all()
        assert PluginLoader.all() is loader1

        registry = PluginRegistry.default()
        registry.clear()

        loader2 = PluginLoader.all()
        assert len(registry.registered_classes()) > 0, "plain all() must repopulate a cleared registry"
        assert loader2 is not loader1

    def test_all_cache_hit_when_registry_unchanged(self) -> None:
        """restore() to identical registry content must NOT invalidate the all() cache."""
        loader1 = PluginLoader.all()

        registry = PluginRegistry.default()
        snap = registry.snapshot()
        registry.restore(snap)

        assert PluginLoader.all() is loader1

    def test_all_rebuilds_after_registry_restore_to_different_content(self) -> None:
        """Plain all() must rebuild after the registry is restored to different content."""
        loader1 = PluginLoader.all()

        registry = PluginRegistry.default()
        registry.restore({})

        loader2 = PluginLoader.all()
        assert len(registry.registered_classes()) > 0, "plain all() must repopulate an emptied registry"
        assert loader2 is not loader1

    def test_all_reentrant_call_raises(self) -> None:
        """A re-entrant all() during the initial build must raise RuntimeError instead of deadlocking."""
        from unittest.mock import MagicMock

        def reentrant_build(*args: object, **kwargs: object) -> None:
            PluginLoader.all()

        with patch.object(PluginLoader, "load_all_plugins", MagicMock(side_effect=reentrant_build)):
            with patch.object(PluginLoader, "load_entry_points", MagicMock()):
                with pytest.raises(RuntimeError):
                    PluginLoader.all()
