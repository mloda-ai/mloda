"""Tests for entry-point based plugin discovery (PluginLoader.load_entry_points).

Contract under test:
- ENTRY_POINT_GROUPS maps "mloda.feature_groups" / "mloda.compute_frameworks" / "mloda.extenders"
  to FeatureGroup / ComputeFramework / Extender.
- PluginLoader.load_entry_points(group=None) discovers installed distributions' entry points via
  importlib.metadata, loads each manifest attribute (a sequence of plugin classes), registers the
  classes into PluginRegistry.default() with source=PluginSource.ENTRY_POINT under module:qualname
  keys, and returns the sorted list of registered keys. The entry-point NAME is a label, never a key.
- Validation is loud (non-sequence manifests, wrong base types), abstract classes are skipped,
  missing optional dependencies skip only the affected entry point, key conflicts raise
  PluginRegistryCollisionError, double loads are idempotent, and PluginLoader.all() folds
  entry points in after the mloda_plugins scan.

Each test builds real on-disk distributions (package + dist-info) in tmp_path with a unique
package name, so importlib.metadata discovery is exercised for real and tests stay xdist-safe.
"""

import importlib
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
    @pytest.mark.parametrize(
        ("slug", "manifest_value"),
        [
            ("int", "42"),
            ("strings", '["not_a_class"]'),
        ],
    )
    def test_manifest_not_a_sequence_of_classes_raises_naming_entry_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slug: str, manifest_value: str
    ) -> None:
        pkg = f"eptest_badmanifest_{slug}_pkg"
        _build_distribution(
            tmp_path,
            pkg,
            f"FEATURE_GROUPS = {manifest_value}\n",
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

        PluginLoader.all()

        registry = PluginRegistry.default()
        key = f"{pkg}.manifest:EpFeatureGroup"
        assert registry.is_registered(_manifest_class(pkg, "EpFeatureGroup"))
        assert registry.get_entry(key).source == PluginSource.ENTRY_POINT
