import importlib
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import _write_broken_optional_root_package

import mloda.core.abstract_plugins.plugin_loader.plugin_loader as plugin_loader_module
from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    _collect_filtered_subclasses,  # noqa: F401
    get_all_filtered_subclasses,
)
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import OPTIONAL_PLUGIN_DEPENDENCIES
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.user import PluginLoader


def _write_fake_base_package(base_dir: Path, base_pkg_name: str, submodule_name: str, imports: str) -> None:
    """A fake base package standing in for mloda_plugins."""
    pkg_dir = base_dir / base_pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{submodule_name}.py").write_text(f"import {imports}\n")
    importlib.invalidate_caches()


def _write_root_module(base_dir: Path, module_name: str) -> None:
    """A standalone top-level module (not a package), so importing a missing name from it raises
    plain ImportError rather than ModuleNotFoundError."""
    (base_dir / f"{module_name}.py").write_text("")


def _write_fake_base_package_bad_from_import(
    base_dir: Path, base_pkg_name: str, submodule_name: str, root_module: str
) -> None:
    """A fake base package submodule doing `from <root_module> import missing_name`."""
    pkg_dir = base_dir / base_pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{submodule_name}.py").write_text(f"from {root_module} import missing_name\n")
    importlib.invalidate_caches()


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

    def test_optional_plugin_dependencies_has_no_orphaned_opentelemetry_entry(self) -> None:
        """opentelemetry was only imported by OtelExtender, deleted on this branch; nothing imports it now."""
        assert "opentelemetry" not in OPTIONAL_PLUGIN_DEPENDENCIES


class TestLoadPluginTransitiveOptionalDependency:
    def test_transitive_missing_dependency_inside_declared_optional_root_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared-optional root whose OWN import fails must be skipped by _load_plugin, mirroring
        the already-fixed load_entry_points traceback fallback."""
        optional_root_pkg = "pltest_transroot_optional_dep"
        missing_subdep = "pltest_transroot_missing_subdep"
        _write_broken_optional_root_package(tmp_path, optional_root_pkg, missing_subdep)

        base_pkg = "pltest_fake_base_pkg"
        submodule = "broken_consumer"
        _write_fake_base_package(tmp_path, base_pkg, submodule, optional_root_pkg)

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            plugin_loader_module,
            "OPTIONAL_PLUGIN_DEPENDENCIES",
            plugin_loader_module.OPTIONAL_PLUGIN_DEPENDENCIES | frozenset({optional_root_pkg}),
        )

        loader = PluginLoader()
        loader.base_package = base_pkg

        loader._load_plugin(submodule)

        assert f"{base_pkg}.{submodule}" not in loader.plugins


class TestLoadPluginPlainImportError:
    def test_declared_optional_root_catches_plain_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A plain ImportError (not ModuleNotFoundError) from an existing declared-optional root
        must be skipped, proving the ImportError widening isn't limited to ModuleNotFoundError."""
        root_module = "pltest_importerror_root"
        _write_root_module(tmp_path, root_module)

        base_pkg = "pltest_importerror_base_pkg"
        submodule = "bad_from_import"
        _write_fake_base_package_bad_from_import(tmp_path, base_pkg, submodule, root_module)

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            plugin_loader_module,
            "OPTIONAL_PLUGIN_DEPENDENCIES",
            plugin_loader_module.OPTIONAL_PLUGIN_DEPENDENCIES | frozenset({root_module}),
        )

        loader = PluginLoader()
        loader.base_package = base_pkg

        with caplog.at_level(logging.DEBUG, logger=plugin_loader_module.__name__):
            loader._load_plugin(submodule)

        assert f"{base_pkg}.{submodule}" not in loader.plugins
        debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
        assert any(root_module in message for message in debug_messages), (
            f"expected a DEBUG message naming the missing root, got: {debug_messages}"
        )

    def test_undeclared_root_import_error_still_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plain ImportError whose root is declared nowhere must still propagate, proving the
        propagation path isn't narrower than ImportError."""
        root_module = "pltest_undeclared_importerror_root"
        _write_root_module(tmp_path, root_module)

        base_pkg = "pltest_undeclared_importerror_base_pkg"
        submodule = "bad_from_import"
        _write_fake_base_package_bad_from_import(tmp_path, base_pkg, submodule, root_module)

        monkeypatch.syspath_prepend(str(tmp_path))

        loader = PluginLoader()
        loader.base_package = base_pkg

        with pytest.raises(ImportError) as exc_info:
            loader._load_plugin(submodule)
        assert not isinstance(exc_info.value, ModuleNotFoundError)


class TestLoadGroupContinuesPastSkippedPlugin:
    def test_broken_optional_dependency_plugin_does_not_abort_group_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One bundled plugin hitting a skippable optional-dependency failure must not abort the
        rest of the group scan, proving load_all_plugins()'s DoD through the real load_group path."""
        optional_root_pkg = "pltest_group_optional_dep"
        missing_subdep = "pltest_group_missing_subdep"
        _write_broken_optional_root_package(tmp_path, optional_root_pkg, missing_subdep)

        base_pkg = "pltest_group_fake_base_pkg"
        group_name = "mygroup"
        base_dir = tmp_path / base_pkg
        base_dir.mkdir()
        (base_dir / "__init__.py").write_text("")
        group_dir = base_dir / group_name
        group_dir.mkdir()
        (group_dir / "__init__.py").write_text("")
        (group_dir / "broken.py").write_text(f"import {optional_root_pkg}\n")
        (group_dir / "good.py").write_text("")
        importlib.invalidate_caches()

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            plugin_loader_module,
            "OPTIONAL_PLUGIN_DEPENDENCIES",
            plugin_loader_module.OPTIONAL_PLUGIN_DEPENDENCIES | frozenset({optional_root_pkg}),
        )

        loader = PluginLoader()
        loader.base_package = base_pkg

        loader.load_group(group_name)

        assert f"{base_pkg}.{group_name}.broken" not in loader.plugins
        assert f"{base_pkg}.{group_name}.good" in loader.plugins
