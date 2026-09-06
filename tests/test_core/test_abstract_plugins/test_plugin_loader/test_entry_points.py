"""Tests for entry-point based plugin discovery (PluginLoader.load_entry_points).

Contract under test:
- ENTRY_POINT_GROUPS maps "mloda.feature_groups" / "mloda.compute_frameworks" / "mloda.extenders"
  to FeatureGroup / ComputeFramework / Extender.
- PluginLoader.load_entry_points(group=None) discovers installed distributions' entry points via
  importlib.metadata, loads each manifest attribute (a sequence of plugin classes), registers the
  classes into PluginRegistry.default() with source=PluginSource.ENTRY_POINT under module:qualname
  keys, and returns the sorted, DEDUPLICATED list of registered keys (a manifest listing the same
  class twice yields its key once). The entry-point NAME is a label, never a key.
- Validation is loud (a non-sequence manifest raises an error naming the entry-point label and
  saying a list or tuple of plugin classes is expected; wrong base types raise naming group and
  class), abstract classes are skipped,
  missing optional dependencies skip only the affected entry point, key conflicts raise
  PluginRegistryCollisionError, double loads are idempotent, and PluginLoader.all() folds
  entry points in after the mloda_plugins scan.
- A companion `mloda.optional_dependencies` entry point (same name) declares per-entry-point
  optional import roots; ImportError (including ModuleNotFoundError) is checked against it,
  falling back to the global OPTIONAL_PLUGIN_DEPENDENCIES set, and logged at WARNING, except
  for the entry point's own package root, which always re-raises.

Each test builds real on-disk distributions (package + dist-info) in tmp_path with a unique
package name, so importlib.metadata discovery is exercised for real and tests stay xdist-safe.
"""

import importlib
import logging
import textwrap
from pathlib import Path

import pytest

import mloda.core.abstract_plugins.plugin_loader.plugin_loader as plugin_loader_module
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    PluginRegistryCollisionError,
    PluginSource,
    register_plugin,
)
from mloda.user import PluginLoader

FEATURE_GROUPS_GROUP = "mloda.feature_groups"
COMPUTE_FRAMEWORKS_GROUP = "mloda.compute_frameworks"
EXTENDERS_GROUP = "mloda.extenders"


def _build_distribution(base_dir: Path, pkg_name: str, manifest_source: str, entry_points_txt: str) -> None:
    """Create a real on-disk distribution: importable package plus dist-info metadata.

    With base_dir on sys.path, importlib.metadata discovers the dist-info (entry_points.txt)
    and the package itself is importable. No wheel building required.
    """
    pkg_dir = base_dir / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "manifest.py").write_text(textwrap.dedent(manifest_source))
    dist_info = base_dir / f"{pkg_name}-1.0.0.dist-info"
    dist_info.mkdir()
    metadata = f"Metadata-Version: 2.1\nName: {pkg_name.replace('_', '-')}\nVersion: 1.0.0\n"
    (dist_info / "METADATA").write_text(metadata)
    (dist_info / "entry_points.txt").write_text(textwrap.dedent(entry_points_txt))
    importlib.invalidate_caches()


def _manifest_class(pkg_name: str, class_name: str) -> type:
    """Import the fake package's manifest module and return one of its plugin classes."""
    module = importlib.import_module(f"{pkg_name}.manifest")
    cls = getattr(module, class_name)
    assert isinstance(cls, type)
    return cls


def _write_module(base_dir: Path, pkg_name: str, module_name: str, source: str) -> None:
    """Add an extra module to an already-built fake package (e.g. the optional_dependencies companion)."""
    (base_dir / pkg_name / f"{module_name}.py").write_text(textwrap.dedent(source))


def _write_root_module(base_dir: Path, module_name: str, source: str = "") -> None:
    """Write a standalone top-level module (not inside a package), so importing a missing name
    from it raises plain ImportError rather than ModuleNotFoundError."""
    (base_dir / f"{module_name}.py").write_text(textwrap.dedent(source))


def _import_error_fg_manifest_source(root_module: str, class_name: str) -> str:
    """A manifest raising ImportError, not ModuleNotFoundError, because `root_module` exists but
    doesn't define the name it imports."""
    return f"""
    from {root_module} import missing_name

    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class {class_name}(FeatureGroup):
        pass


    FEATURE_GROUPS = [{class_name}]
    """


def _own_package_broken_manifest_source(pkg_name: str, class_name: str) -> str:
    """A manifest whose own package fails importing a submodule of itself, so the failing
    import's root equals the entry point's own module root."""
    return f"""
    import {pkg_name}.missing_submodule

    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class {class_name}(FeatureGroup):
        pass


    FEATURE_GROUPS = [{class_name}]
    """


_FG_MANIFEST = """
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class EpFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [EpFeatureGroup]
"""

_TWO_FG_MANIFEST = """
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class ZebraEpFeatureGroup(FeatureGroup):
        pass


    class AlphaEpFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [ZebraEpFeatureGroup, AlphaEpFeatureGroup]
"""

_TRIPLE_MANIFEST = """
    from typing import Any

    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.feature_group import FeatureGroup
    from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


    class EpFeatureGroup(FeatureGroup):
        pass


    class EpComputeFramework(ComputeFramework):
        pass


    class EpExtender(Extender):
        def wraps(self) -> set[ExtenderHook]:
            return set()

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func


    FEATURE_GROUPS = [EpFeatureGroup]
    COMPUTE_FRAMEWORKS = [EpComputeFramework]
    EXTENDERS = [EpExtender]
"""

_ABSTRACT_MANIFEST = """
    from abc import abstractmethod
    from typing import Any

    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class AbstractEpFeatureGroup(FeatureGroup):
        @abstractmethod
        def _ep_probe(self) -> None: ...

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            # Inert for other tests' feature resolution in this worker; the default
            # matching falls back to cls(), which raises TypeError on abstract classes.
            return False


    class ConcreteEpFeatureGroup(AbstractEpFeatureGroup):
        def _ep_probe(self) -> None:
            return None


    FEATURE_GROUPS = [AbstractEpFeatureGroup, ConcreteEpFeatureGroup]
"""

_OPTIONAL_DEP_MANIFEST = """
    import eptest_fake_optional_dep

    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class EpOptionalDepFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [EpOptionalDepFeatureGroup]
"""

_HARD_DEP_MANIFEST = """
    import eptest_missing_hard_dep

    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class EpHardDepFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [EpHardDepFeatureGroup]
"""

# Fails with ModuleNotFoundError on a root NOT in the global OPTIONAL_PLUGIN_DEPENDENCIES set;
# only tolerated when a companion `mloda.optional_dependencies` entry declares it optional.
_DECLARED_OPTIONAL_EXTENDER_MANIFEST = """
    from typing import Any

    import eptest_declopt_missing_root

    from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


    class EpDeclOptExtender(Extender):
        def wraps(self) -> set[ExtenderHook]:
            return set()

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func


    EXTENDERS = [EpDeclOptExtender]
"""

_DECLARED_OPTIONAL_DEPS_MODULE_SOURCE = """
    OPTIONAL_DEPENDENCIES = frozenset({"eptest_declopt_missing_root"})
"""


class _PreexistingConflictFG(FeatureGroup):
    """Registered manually under an entry-point class's key to force a collision."""


class TestEntryPointGroupsConstant:
    def test_entry_point_groups_maps_groups_to_base_types(self) -> None:
        groups = plugin_loader_module.ENTRY_POINT_GROUPS
        assert groups == {
            FEATURE_GROUPS_GROUP: FeatureGroup,
            COMPUTE_FRAMEWORKS_GROUP: ComputeFramework,
            EXTENDERS_GROUP: Extender,
        }


class TestLoadEntryPointsDiscovery:
    def test_registers_class_with_module_qualname_key_and_entry_point_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_discovery_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        loader = PluginLoader()
        keys = loader.load_entry_points()

        expected_key = f"{pkg}.manifest:EpFeatureGroup"
        assert expected_key in keys
        entry = PluginRegistry.default().get_entry(expected_key)
        assert entry.cls is _manifest_class(pkg, "EpFeatureGroup")
        assert entry.source == PluginSource.ENTRY_POINT
        assert entry.plugin_type is FeatureGroup

    def test_returns_sorted_list_of_registered_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = "eptest_sortedkeys_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _TWO_FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        keys = PluginLoader().load_entry_points()

        assert keys == sorted(keys)
        expected = sorted(
            [
                f"{pkg}.manifest:AlphaEpFeatureGroup",
                f"{pkg}.manifest:ZebraEpFeatureGroup",
            ]
        )
        assert [key for key in keys if key.startswith(f"{pkg}.")] == expected

    def test_registers_all_three_plugin_kinds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = "eptest_kinds_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _TRIPLE_MANIFEST,
            f"""
            [mloda.feature_groups]
            main = {pkg}.manifest:FEATURE_GROUPS

            [mloda.compute_frameworks]
            main = {pkg}.manifest:COMPUTE_FRAMEWORKS

            [mloda.extenders]
            main = {pkg}.manifest:EXTENDERS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        PluginLoader().load_entry_points()

        registry = PluginRegistry.default()
        for class_name, base_type in [
            ("EpFeatureGroup", FeatureGroup),
            ("EpComputeFramework", ComputeFramework),
            ("EpExtender", Extender),
        ]:
            entry = registry.get_entry(f"{pkg}.manifest:{class_name}")
            assert entry.cls is _manifest_class(pkg, class_name)
            assert entry.plugin_type is base_type
            assert entry.source == PluginSource.ENTRY_POINT


class TestLoadEntryPointsGroupFilter:
    def test_specific_group_loads_only_that_group(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = "eptest_groupfilter_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _TRIPLE_MANIFEST,
            f"""
            [mloda.feature_groups]
            main = {pkg}.manifest:FEATURE_GROUPS

            [mloda.compute_frameworks]
            main = {pkg}.manifest:COMPUTE_FRAMEWORKS

            [mloda.extenders]
            main = {pkg}.manifest:EXTENDERS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        keys = PluginLoader().load_entry_points(group=COMPUTE_FRAMEWORKS_GROUP)

        registry = PluginRegistry.default()
        cf_key = f"{pkg}.manifest:EpComputeFramework"
        assert [key for key in keys if key.startswith(f"{pkg}.")] == [cf_key]
        assert registry.get_entry(cf_key).cls is _manifest_class(pkg, "EpComputeFramework")
        assert not registry.is_registered(_manifest_class(pkg, "EpFeatureGroup"))
        assert not registry.is_registered(_manifest_class(pkg, "EpExtender"))

    def test_unknown_group_raises_value_error_listing_valid_groups(self) -> None:
        loader = PluginLoader()
        with pytest.raises(ValueError) as exc_info:
            loader.load_entry_points(group="mloda.bogus_group")
        message = str(exc_info.value)
        assert "mloda.bogus_group" in message
        assert FEATURE_GROUPS_GROUP in message
        assert COMPUTE_FRAMEWORKS_GROUP in message
        assert EXTENDERS_GROUP in message


class TestLoadEntryPointsValidation:
    def test_non_sequence_manifest_error_names_label_and_expects_list_or_tuple(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_badmanifest_int_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            "FEATURE_GROUPS = 42\n",
            f"""
            [mloda.feature_groups]
            broken_label = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises((TypeError, ValueError)) as exc_info:
            PluginLoader().load_entry_points()
        message = str(exc_info.value)
        assert "broken_label" in message, "the error must name the entry-point label"
        assert "list" in message, "the error must say a list or tuple of plugin classes is expected"
        assert "tuple" in message, "the error must say a list or tuple of plugin classes is expected"

    def test_manifest_with_non_class_items_raises_naming_entry_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_badmanifest_strings_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            'FEATURE_GROUPS = ["not_a_class"]\n',
            f"""
            [mloda.feature_groups]
            broken_label = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises((TypeError, ValueError)) as exc_info:
            PluginLoader().load_entry_points()
        assert "broken_label" in str(exc_info.value)

    def test_wrong_base_type_for_group_raises_naming_group_and_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_wrongbase_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _TRIPLE_MANIFEST,
            f"""
            [mloda.feature_groups]
            wrong = {pkg}.manifest:EXTENDERS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises((TypeError, ValueError)) as exc_info:
            PluginLoader().load_entry_points()
        message = str(exc_info.value)
        assert FEATURE_GROUPS_GROUP in message
        assert "EpExtender" in message

    def test_abstract_classes_in_manifest_are_skipped_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_abstract_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _ABSTRACT_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        keys = PluginLoader().load_entry_points()

        registry = PluginRegistry.default()
        assert registry.is_registered(_manifest_class(pkg, "ConcreteEpFeatureGroup"))
        assert not registry.is_registered(_manifest_class(pkg, "AbstractEpFeatureGroup"))
        assert [key for key in keys if key.startswith(f"{pkg}.")] == [f"{pkg}.manifest:ConcreteEpFeatureGroup"]


class TestLoadEntryPointsMissingDependencies:
    def test_missing_optional_dependency_skips_entry_point_but_loads_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken_pkg = "eptest_optdep_broken_pkg"
        good_pkg = "eptest_optdep_good_pkg"
        _build_distribution(
            tmp_path,
            broken_pkg,
            _OPTIONAL_DEP_MANIFEST,
            f"""
            [mloda.feature_groups]
            broken = {broken_pkg}.manifest:FEATURE_GROUPS
            """,
        )
        _build_distribution(
            tmp_path,
            good_pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            good = {good_pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            plugin_loader_module,
            "OPTIONAL_PLUGIN_DEPENDENCIES",
            plugin_loader_module.OPTIONAL_PLUGIN_DEPENDENCIES | frozenset({"eptest_fake_optional_dep"}),
        )

        keys = PluginLoader().load_entry_points()

        registry = PluginRegistry.default()
        good_key = f"{good_pkg}.manifest:EpFeatureGroup"
        assert good_key in keys
        assert registry.get_entry(good_key).source == PluginSource.ENTRY_POINT
        assert not any(key.startswith(f"{broken_pkg}.") for key in keys)
        assert registry.get(f"{broken_pkg}.manifest:EpOptionalDepFeatureGroup") is None

    def test_missing_non_optional_dependency_propagates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = "eptest_harddep_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _HARD_DEP_MANIFEST,
            f"""
            [mloda.feature_groups]
            hard = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ModuleNotFoundError):
            PluginLoader().load_entry_points()


class TestLoadEntryPointsOptionalDependenciesDeclaration:
    """The new `mloda.optional_dependencies` group: per-entry-point optional-root declarations,
    plain ImportError handling, and the own-package re-raise guard."""

    def test_declared_optional_root_skips_entry_point_and_logs_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken_pkg = "eptest_declopt_broken_pkg"
        good_pkg = "eptest_declopt_good_pkg"
        _build_distribution(
            tmp_path,
            broken_pkg,
            _DECLARED_OPTIONAL_EXTENDER_MANIFEST,
            f"""
            [mloda.extenders]
            demo = {broken_pkg}.manifest:EXTENDERS

            [mloda.optional_dependencies]
            demo = {broken_pkg}.optional_deps:OPTIONAL_DEPENDENCIES
            """,
        )
        _write_module(tmp_path, broken_pkg, "optional_deps", _DECLARED_OPTIONAL_DEPS_MODULE_SOURCE)
        _build_distribution(
            tmp_path,
            good_pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            good = {good_pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with caplog.at_level(logging.WARNING, logger=plugin_loader_module.__name__):
            keys = PluginLoader().load_entry_points()

        registry = PluginRegistry.default()
        good_key = f"{good_pkg}.manifest:EpFeatureGroup"
        assert good_key in keys
        assert not any(key.startswith(f"{broken_pkg}.") for key in keys)
        assert registry.get(f"{broken_pkg}.manifest:EpDeclOptExtender") is None

        warning_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert any("demo" in message and "eptest_declopt_missing_root" in message for message in warning_messages), (
            f"expected a WARNING naming the entry point and the missing module, got: {warning_messages}"
        )

    def test_declared_optional_root_catches_plain_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        pkg = "eptest_declopt_importerror_pkg"
        root_module = "eptest_declopt_importerror_root"
        _write_root_module(tmp_path, root_module)
        _build_distribution(
            tmp_path,
            pkg,
            _import_error_fg_manifest_source(root_module, "EpImportErrorFeatureGroup"),
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS

            [mloda.optional_dependencies]
            demo = {pkg}.optional_deps:OPTIONAL_DEPENDENCIES
            """,
        )
        _write_module(tmp_path, pkg, "optional_deps", f'OPTIONAL_DEPENDENCIES = ("{root_module}",)\n')
        monkeypatch.syspath_prepend(str(tmp_path))

        with caplog.at_level(logging.WARNING, logger=plugin_loader_module.__name__):
            keys = PluginLoader().load_entry_points()

        assert not any(key.startswith(f"{pkg}.") for key in keys)
        assert PluginRegistry.default().get(f"{pkg}.manifest:EpImportErrorFeatureGroup") is None
        warning_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert any("demo" in message and root_module in message for message in warning_messages), (
            f"expected a WARNING naming the entry point and the missing module, got: {warning_messages}"
        )

    def test_undeclared_root_import_error_still_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plain ImportError (already uncaught today) must still propagate once caught, when
        its root is declared nowhere (neither per-entry-point nor in the global set)."""
        pkg = "eptest_undeclared_importerror_pkg"
        root_module = "eptest_undeclared_importerror_root"
        _write_root_module(tmp_path, root_module)
        _build_distribution(
            tmp_path,
            pkg,
            _import_error_fg_manifest_source(root_module, "EpUndeclaredImportErrorFeatureGroup"),
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ImportError) as exc_info:
            PluginLoader().load_entry_points()
        assert not isinstance(exc_info.value, ModuleNotFoundError)

    def test_own_package_root_always_raises_even_if_declared_optional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A (nonsensical) declaration that the entry point's own package root is optional must
        not suppress the error: it means the plugin's own package is broken."""
        pkg = "eptest_declopt_own_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _own_package_broken_manifest_source(pkg, "EpOwnPkgFeatureGroup"),
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS

            [mloda.optional_dependencies]
            demo = {pkg}.optional_deps:OPTIONAL_DEPENDENCIES
            """,
        )
        _write_module(tmp_path, pkg, "optional_deps", f'OPTIONAL_DEPENDENCIES = frozenset({{"{pkg}"}})\n')
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ModuleNotFoundError):
            PluginLoader().load_entry_points()

    def test_undeclared_root_in_global_set_still_falls_back_and_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: with no companion `mloda.optional_dependencies` entry at all, a
        missing root that IS in the global OPTIONAL_PLUGIN_DEPENDENCIES set still skips."""
        broken_pkg = "eptest_declopt_fallback_broken_pkg"
        good_pkg = "eptest_declopt_fallback_good_pkg"
        _build_distribution(
            tmp_path,
            broken_pkg,
            _OPTIONAL_DEP_MANIFEST,
            f"""
            [mloda.feature_groups]
            broken = {broken_pkg}.manifest:FEATURE_GROUPS
            """,
        )
        _build_distribution(
            tmp_path,
            good_pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            good = {good_pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            plugin_loader_module,
            "OPTIONAL_PLUGIN_DEPENDENCIES",
            plugin_loader_module.OPTIONAL_PLUGIN_DEPENDENCIES | frozenset({"eptest_fake_optional_dep"}),
        )

        keys = PluginLoader().load_entry_points()

        good_key = f"{good_pkg}.manifest:EpFeatureGroup"
        assert good_key in keys
        assert not any(key.startswith(f"{broken_pkg}.") for key in keys)


class TestLoadEntryPointsIdempotencyAndCollisions:
    def test_double_load_registers_nothing_new_and_raises_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_idempotent_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        loader = PluginLoader()
        loader.load_entry_points()
        registry = PluginRegistry.default()
        first_snapshot = registry.snapshot()
        assert f"{pkg}.manifest:EpFeatureGroup" in first_snapshot

        loader.load_entry_points()

        assert registry.snapshot() == first_snapshot

    def test_different_class_already_under_key_raises_collision_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_collision_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        conflicting_key = f"{pkg}.manifest:EpFeatureGroup"
        register_plugin(_PreexistingConflictFG, name=conflicting_key)

        with pytest.raises(PluginRegistryCollisionError):
            PluginLoader().load_entry_points()
        assert PluginRegistry.default().get(conflicting_key) is _PreexistingConflictFG


_DUP_FG_MANIFEST = """
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


    class DupEpFeatureGroup(FeatureGroup):
        pass


    FEATURE_GROUPS = [DupEpFeatureGroup, DupEpFeatureGroup]
"""


class TestLoadEntryPointsDeduplication:
    def test_manifest_listing_same_class_twice_yields_key_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_dupclass_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _DUP_FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        keys = PluginLoader().load_entry_points()

        expected_key = f"{pkg}.manifest:DupEpFeatureGroup"
        assert keys.count(expected_key) == 1, "a class listed twice in one manifest must yield its key once"
        assert keys == sorted(set(keys)), "load_entry_points must return a sorted, deduplicated key list"
        assert PluginRegistry.default().get_entry(expected_key).cls is _manifest_class(pkg, "DupEpFeatureGroup")


class TestEntryPointNameIsLabelOnly:
    def test_two_labels_for_same_manifest_register_class_once_under_module_qualname(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = "eptest_labels_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            label_one = {pkg}.manifest:FEATURE_GROUPS
            label_two = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        PluginLoader().load_entry_points()

        registry = PluginRegistry.default()
        cls = _manifest_class(pkg, "EpFeatureGroup")
        keys_for_class = [key for key, entry in registry.snapshot().items() if entry.cls is cls]
        assert keys_for_class == [f"{pkg}.manifest:EpFeatureGroup"]
        assert registry.get("label_one") is None
        assert registry.get("label_two") is None


class TestPluginLoaderAllLoadsEntryPoints:
    def test_all_folds_in_entry_points_after_module_scan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = "eptest_all_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            _FG_MANIFEST,
            f"""
            [mloda.feature_groups]
            demo = {pkg}.manifest:FEATURE_GROUPS
            """,
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        PluginLoader.all(force_reload=True)

        registry = PluginRegistry.default()
        key = f"{pkg}.manifest:EpFeatureGroup"
        assert registry.is_registered(_manifest_class(pkg, "EpFeatureGroup"))
        assert registry.get_entry(key).source == PluginSource.ENTRY_POINT
