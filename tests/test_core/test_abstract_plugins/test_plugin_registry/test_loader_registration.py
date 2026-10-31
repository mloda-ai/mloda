"""Tests for the loader -> registry discovery funnel (issue #526, work item 3).

Contract: PluginLoader loads feed the default PluginRegistry. Every non-abstract
FeatureGroup, ComputeFramework, or Extender subclass DEFINED in a loaded module
(cls.__module__ equals the loaded module name) is registered exactly once under its
defining module, with source="loader". Abstract classes (inspect.isabstract) are
infrastructure, never plugins, and stay out. Registration is idempotent, happens on
the sys.modules cache path too (import-order and import-history independent), and
never picks up ad-hoc subclasses defined outside loaded plugin modules.

The scopes used here (python_dict, sqlite, node_centrality) need no optional dependency beyond pandas and numpy.
The autouse conftest fixture restores the default registry around every test, so clearing it inside a test is safe.
"""

import abc
import gc
import inspect
import pathlib
import sys
import types
from abc import abstractmethod
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    _PLUGIN_BASE_TYPES,
    PluginRegistry,
    register_module_plugins,
    register_plugin,
)
from mloda.user import PluginLoader

PYTHON_DICT_FRAMEWORK_MODULE = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework"


def _default_key(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _python_dict_framework_cls() -> type[Any]:
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework,
    )

    return PythonDictFramework


def _sqlite_framework_cls() -> type[Any]:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

    return SqliteFramework


def _load_python_dict_compute_scope() -> PluginLoader:
    """Small, dependency-free scope: the python_dict compute framework files only."""
    loader = PluginLoader()
    loader.load_matching("compute_framework", "*python_dict*")
    return loader


class TestLoaderRegistersLoadedPlugins:
    def test_loader_registers_classes_defined_in_loaded_modules(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        assert registry.is_registered(python_dict_framework), (
            "PluginLoader.load_matching must register plugin classes defined in loaded modules "
            "in the default PluginRegistry"
        )

        module = sys.modules[PYTHON_DICT_FRAMEWORK_MODULE]
        defined_plugin_classes = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, _PLUGIN_BASE_TYPES) and obj.__module__ == module.__name__
        ]
        assert defined_plugin_classes, "sanity: the loaded module defines at least one plugin class"
        for cls in defined_plugin_classes:
            assert registry.is_registered(cls), f"{cls!r} is defined in a loaded module but is not registered"

    def test_loader_registers_implementations_defined_in_loaded_modules(self) -> None:
        from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvPyArrowTable

        registry = PluginRegistry.default()
        registry.clear()

        PluginLoader().load_matching("feature_group", "*csv_fg*")

        assert registry.is_registered(CsvPyArrowTable)

    def test_loader_entries_carry_loader_provenance(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        key = _default_key(python_dict_framework)
        assert key in registry.snapshot(), f"loader must register '{key}' in the default registry"

        entry = registry.get_entry(key)
        assert entry.cls is python_dict_framework
        assert entry.source == "loader"
        assert entry.source_module == python_dict_framework.__module__


class TestLoaderRegistrationKeying:
    def test_imported_classes_register_once_under_their_defining_module(self) -> None:
        """A class imported into other loaded modules must not get a second entry there.

        Both mloda_plugins.feature_group.experimental.text_cleaning.python_dict and
        mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict
        import PythonDictFramework; only the defining module produces an entry.
        """
        registry = PluginRegistry.default()
        registry.clear()

        loader = PluginLoader()
        loader.load_matching("compute_framework", "*python_dict*")
        loader.load_matching("feature_group", "*python_dict*")

        python_dict_framework = _python_dict_framework_cls()
        entries = registry.snapshot()
        keys_for_class = [key for key, entry in entries.items() if entry.cls is python_dict_framework]
        assert keys_for_class == [_default_key(python_dict_framework)], (
            "PythonDictFramework must be registered exactly once, keyed by its own defining module, "
            f"got keys: {keys_for_class}"
        )

        for entry in entries.values():
            assert entry.source_module == entry.cls.__module__
            assert entry.name == _default_key(entry.cls)


class TestLoaderRegistrationIdempotency:
    def test_repeated_load_registers_nothing_new(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()
        first_snapshot = registry.snapshot()
        assert first_snapshot, "loader run must populate the cleared default registry"

        _load_python_dict_compute_scope()
        assert registry.snapshot() == first_snapshot, "loading the same scope twice must not add or change entries"

    def test_load_of_already_imported_modules_reregisters(self) -> None:
        """Registration must also happen on the sys.modules cache path of _load_plugin."""
        registry = PluginRegistry.default()
        registry.clear()

        _load_python_dict_compute_scope()
        first_snapshot = registry.snapshot()
        assert first_snapshot, "loader run must populate the cleared default registry"
        assert PYTHON_DICT_FRAMEWORK_MODULE in sys.modules, "sanity: module is cached in sys.modules"

        registry.clear()
        _load_python_dict_compute_scope()

        python_dict_framework = _python_dict_framework_cls()
        assert registry.is_registered(python_dict_framework), (
            "loading a scope whose modules are already in sys.modules must still register the plugin classes"
        )
        assert registry.snapshot() == first_snapshot


class TestLoaderRegistrationOrderIndependence:
    def test_load_order_does_not_change_registry_contents(self) -> None:
        registry = PluginRegistry.default()

        registry.clear()
        loader_ab = PluginLoader()
        loader_ab.load_matching("compute_framework", "*python_dict*")
        loader_ab.load_matching("compute_framework", "*sqlite*")
        result_ab = registry.list_registered(ComputeFramework)

        registry.clear()
        loader_ba = PluginLoader()
        loader_ba.load_matching("compute_framework", "*sqlite*")
        loader_ba.load_matching("compute_framework", "*python_dict*")
        result_ba = registry.list_registered(ComputeFramework)

        assert result_ab, "loading two compute framework scopes must register compute frameworks"
        assert _python_dict_framework_cls() in result_ab
        assert _sqlite_framework_cls() in result_ab
        assert result_ab == result_ba, "registry contents must not depend on load order"


class TestLoaderIgnoresTransientSubclasses:
    def test_local_subclass_stays_out_but_explicit_register_opts_in(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _AdHocFeatureGroup(FeatureGroup):
            """Simulates a test double or notebook-cell class; must never auto-register."""

        _load_python_dict_compute_scope()

        assert not registry.is_registered(_AdHocFeatureGroup), (
            "ad-hoc subclasses defined outside loaded plugin modules must stay out of the registry"
        )

        key = register_plugin(_AdHocFeatureGroup)
        assert registry.is_registered(_AdHocFeatureGroup), "explicit register_plugin() must opt a class in"
        assert registry.get_entry(key).source == "manual"

        assert registry.is_registered(_python_dict_framework_cls()), (
            "loader-run plugins must be registered while ad-hoc subclasses are not"
        )


class TestLoaderRegistrationSkipsAbstractClasses:
    """Abstract helper base classes are infrastructure, never plugins; they must not auto-register."""

    def test_register_module_plugins_skips_abstract_classes(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        module = types.ModuleType("fake_registry_abstract_mod")

        class _AbstractHelperBaseFG(FeatureGroup):
            @abstractmethod
            def _abstract_registry_probe_hook(self) -> None: ...

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                # Inert for other tests' feature resolution in this worker; the default
                # matching falls back to cls(), which raises TypeError on abstract classes.
                return False

        class _ConcreteLeafProbeFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return False

        _AbstractHelperBaseFG.__module__ = module.__name__
        _ConcreteLeafProbeFG.__module__ = module.__name__
        setattr(module, "_AbstractHelperBaseFG", _AbstractHelperBaseFG)
        setattr(module, "_ConcreteLeafProbeFG", _ConcreteLeafProbeFG)

        try:
            assert inspect.isabstract(_AbstractHelperBaseFG), "sanity: the helper base double is abstract"
            assert not inspect.isabstract(_ConcreteLeafProbeFG), "sanity: the leaf double is concrete"

            register_module_plugins(module)

            assert registry.is_registered(_ConcreteLeafProbeFG), (
                "concrete plugin classes defined in a loaded module must be registered"
            )
            assert not registry.is_registered(_AbstractHelperBaseFG), (
                "abstract helper base classes must not be auto-registered by register_module_plugins"
            )
        finally:
            # Drop every strong ref to the fake-module doubles so they cannot poison
            # other tests in this worker via FeatureGroup.__subclasses__(): their fake
            # __module__ has no importable file, so version()/class_source_hash raises
            # TypeError for them. The conftest fixture restores the registry snapshot
            # after the test, so clearing the loader-registered entry here is safe.
            registry.clear()
            delattr(module, "_AbstractHelperBaseFG")
            delattr(module, "_ConcreteLeafProbeFG")
            del _AbstractHelperBaseFG, _ConcreteLeafProbeFG, module
            gc.collect()

    def test_loaded_scope_registers_concrete_classes(self) -> None:
        """End to end through PluginLoader: the node_centrality scope needs only pandas and numpy and its
        operation group is concrete, so loading the scope registers it."""
        registry = PluginRegistry.default()
        registry.clear()

        loader = PluginLoader()
        loader.load_matching("feature_group", "node_centrality/*")

        from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup

        assert not inspect.isabstract(NodeCentralityFeatureGroup), "sanity: the scope has a concrete class"

        registered_feature_groups = registry.list_registered(FeatureGroup)
        assert NodeCentralityFeatureGroup in registered_feature_groups, (
            "loading a scope must register its concrete plugin classes"
        )


def _spy_on_plugin_subclasschecks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap ABCMeta.__subclasscheck__; the returned list collects calls whose cls is a plugin base, FeatureGroup, or Implementation."""
    original = abc.ABCMeta.__subclasscheck__
    counted: list[str] = []

    def spy(cls: abc.ABCMeta, subclass: type[Any]) -> bool:
        if FeatureGroup in cls.__mro__ or Implementation in cls.__mro__ or cls in _PLUGIN_BASE_TYPES:
            counted.append(cls.__qualname__)
        return original(cls, subclass)

    monkeypatch.setattr(abc.ABCMeta, "__subclasscheck__", spy)
    return counted


def _invalidate_abc_caches() -> None:
    class _Throwaway(abc.ABC):
        pass

    class _Registered:
        pass

    _Throwaway.register(_Registered)


class TestPluginDiscoveryAvoidsAbcWalk:
    """Plugin discovery must not pay the ABC subclass-tree walk for classes it will not register."""

    def test_register_module_plugins_skips_abc_walk_for_unregistered_classes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        module = types.ModuleType("fake_registry_abc_walk_mod")

        class _OwnConcreteFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return False

        class _OwnPlainClass:
            pass

        _OwnConcreteFG.__module__ = module.__name__
        _OwnPlainClass.__module__ = module.__name__
        setattr(module, "_OwnConcreteFG", _OwnConcreteFG)
        setattr(module, "_OwnPlainClass", _OwnPlainClass)
        setattr(module, "ImportedFramework", _python_dict_framework_cls())
        setattr(module, "ImportedPath", pathlib.Path)
        setattr(module, "alias", tuple[str, ...])

        try:
            counted = _spy_on_plugin_subclasschecks(monkeypatch)

            keys = register_module_plugins(module)

            assert keys == [_default_key(_OwnConcreteFG)]
            assert counted == [], f"ABC subclass walk ran {len(counted)} times for classes it would not register"
        finally:
            monkeypatch.undo()
            registry.clear()
            for name in ("_OwnConcreteFG", "_OwnPlainClass", "ImportedFramework", "ImportedPath", "alias"):
                delattr(module, name)
            del _OwnConcreteFG, _OwnPlainClass, module
            gc.collect()

    def test_cold_plugin_loader_cost_does_not_grow_with_live_feature_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The call count stands in for wall time to stay stable under xdist load.
        PluginLoader.all(force_reload=True)
        counted = _spy_on_plugin_subclasschecks(monkeypatch)

        _invalidate_abc_caches()
        counted.clear()
        PluginLoader.all(force_reload=True)
        baseline = len(counted)

        extras: list[type[Any]] = []
        try:
            for i in range(300):
                extras.append(
                    type(
                        f"_ColdLoadExtraFG{i}",
                        (FeatureGroup,),
                        {
                            "__module__": __name__,
                            "__qualname__": f"TestPluginDiscoveryAvoidsAbcWalk.<locals>._ColdLoadExtraFG{i}",
                            "match_feature_group_criteria": classmethod(lambda cls, *a, **k: False),
                        },
                    )
                )
            _invalidate_abc_caches()
            counted.clear()
            PluginLoader.all(force_reload=True)
            with_extra = len(counted)
        finally:
            monkeypatch.undo()
            PluginRegistry.default().clear()
            extras.clear()
            del extras
            gc.collect()

        assert with_extra - baseline <= 10, (
            f"cold load did {with_extra - baseline} extra ABC subclass checks for 300 extra FeatureGroups "
            f"(baseline {baseline}, with extras {with_extra})"
        )

    def test_prepare_triggers_no_abc_subclass_check_against_plugin_classes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from collections import Counter

        from mloda.core.abstract_plugins.components.feature import Feature
        from mloda.core.abstract_plugins.components.feature_name import FeatureName
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.provider import ApiDataFG, DataCreatorFeatureGroup, FeatureSet
        from mloda.user import Index, JoinSpec, Link, PluginCollector, mloda
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group import (
            OPTOY_CHAIN_SRC,
            OPTOY_CHAIN_SUFFIX,
            OpToyChainFG,
            OpToyChainPandas,
            OpToyChainSourceFG,
        )

        class _PrepareWalkDerivedFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return str(feature_name) == "prepare_walk_derived"

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature(OPTOY_CHAIN_SRC + OPTOY_CHAIN_SUFFIX)}

        class _PrepareWalkApiFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return str(feature_name) == "prepare_walk_api"

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature(name="walk_id", index=Index(("walk_id",)))}

        class _PrepareWalkLeftFG(DataCreatorFeatureGroup):
            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"pw_key", "pw_left_val"}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"pw_key": [1, 2], "pw_left_val": [10, 20]}

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        class _PrepareWalkRightFG(DataCreatorFeatureGroup):
            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"pw_rkey", "pw_right_val"}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"pw_rkey": [1, 2], "pw_right_val": [1, 2]}

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        class _PrepareWalkJoinedFG(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return str(feature_name) == "prepare_walk_joined"

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature("pw_left_val"), Feature("pw_right_val")}

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        assert OpToyChainPandas.feature_group is OpToyChainFG
        link_collector = PluginCollector.enabled_feature_groups(
            {_PrepareWalkLeftFG, _PrepareWalkRightFG, _PrepareWalkJoinedFG}
        )
        prepare_link = Link.inner(JoinSpec(_PrepareWalkLeftFG, "pw_key"), JoinSpec(_PrepareWalkRightFG, "pw_rkey"))
        chain_collector = PluginCollector.enabled_feature_groups(
            {OpToyChainSourceFG, OpToyChainFG, _PrepareWalkDerivedFG}
        )
        api_collector = PluginCollector.enabled_feature_groups({ApiDataFG, _PrepareWalkApiFG})

        def prepare_all(
            chain: PluginCollector = chain_collector,
            api: PluginCollector = api_collector,
            joined: PluginCollector = link_collector,
            link: Link = prepare_link,
        ) -> None:
            mloda.prepare(
                [
                    Feature(OPTOY_CHAIN_SRC + OPTOY_CHAIN_SUFFIX, implementation=OpToyChainPandas),
                    "prepare_walk_derived",
                ],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=chain,
            )
            mloda.prepare(
                ["prepare_walk_api"],
                compute_frameworks=[PandasDataFrame],
                api_data={"WalkSet": {"walk_id": [1, 2]}},
                plugin_collector=api,
            )
            mloda.prepare(
                ["prepare_walk_joined"],
                compute_frameworks=[PandasDataFrame],
                links={link},
                plugin_collector=joined,
            )

        try:
            prepare_all()  # warm-up so the result does not depend on test order
            counted = _spy_on_plugin_subclasschecks(monkeypatch)
            prepare_all()
            assert counted == [], f"prepare ran ABC subclass checks against plugin classes: {Counter(counted)}"
        finally:
            monkeypatch.undo()
            del chain_collector, api_collector, link_collector, prepare_all, prepare_link
            del _PrepareWalkDerivedFG, _PrepareWalkApiFG, _PrepareWalkLeftFG, _PrepareWalkRightFG, _PrepareWalkJoinedFG

    def test_virtual_abc_subclasses_are_not_plugins(self) -> None:
        class _VirtualHostFG(FeatureGroup):
            @abstractmethod
            def _virtual_host_probe_hook(self) -> None: ...

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                return False

        class _VirtualX:
            pass

        try:
            _VirtualHostFG.register(_VirtualX)
            assert issubclass(_VirtualX, FeatureGroup), "sanity: X is a virtual FeatureGroup subclass"

            assert not is_nominal_subclass(_VirtualX, FeatureGroup)

            with pytest.raises(ValueError, match="is not a subclass of FeatureGroup"):
                PluginRegistry().register(_VirtualX)
            with pytest.raises(TypeError, match="requires subclasses of"):
                PluginLoader()._register_manifest("label", "mloda.feature_groups", FeatureGroup, (_VirtualX,))
        finally:
            del _VirtualHostFG, _VirtualX
            gc.collect()
