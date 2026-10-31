"""SourceFeatureGroup loaders, neutral fallback and the describe_columns/count_rows tools."""

import gc
import importlib
import types
import typing
import weakref
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.implementation import (
    Implementation,
    loaded_implementations,
)
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.provider import SourceFeatureGroup
from mloda.user import DataAccessCollection, Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.subclass_walks import collect_until_dead
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group import (
    OpToyBareFG,
    OpToyFG,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation import (
    WORKER_COLUMN,
    WORKER_VALUES,
    ToyWorkerFG,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation_impl import (
    ToyWorkerPythonDict,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToyDeclaredFG,
    ToyBoundedNeutralDictFG,
    ToySourceFG,
    column_values,
    neutral_csv_group,
    toy_dac,
)

SENTINEL = [{"toyfmt_ld": 111}, {"toyfmt_ld": 222}]


_T = TypeVar("_T")
_G = TypeVar("_G", bound=FeatureGroup)
_G2 = TypeVar("_G2", bound=FeatureGroup)

_RELOADED_IMPL_SOURCE = """
from mloda.core.abstract_plugins.implementation import Implementation


class ReloadedGroup(_SOURCE_GROUP_):
    pass


class ReloadedMid(Implementation[_G_]):
    pass


class ReloadedImpl({base}[ReloadedGroup]):
    feature_group = ReloadedGroup
    compute_framework = _FRAMEWORK_

    @classmethod
    def calculate(cls, *args):
        return None
"""


def _release_subscripted_generics() -> None:
    """Drop typing's alias cache, which keeps subscripted local generic classes alive."""
    for cleanup in getattr(typing, "_cleanups", []):
        cleanup()


def _toy_operation() -> Any:
    """Imported lazily so a missing OperationFeatureGroup fails only the tests that need it."""
    return importlib.import_module(
        "tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group"
    )


def _run(group: type, frameworks: Any = None, column: str = "toyfmt_ld") -> list[Any]:
    return mloda.run_all(
        [column],
        compute_frameworks=frameworks or [PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        data_access_collection=toy_dac(h1={column: [1, 2]}),
    )


def _sentinel(serving: Any, match: SourceMatch, features: Any) -> Any:
    return SENTINEL


_MATCH = SourceMatch(source="toyfmt_src", access="toyfmt_src")


class _LoadCapture(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            self.contexts.append(context)
        return result


class _BlindDeclaredFG(ToyDeclaredFG):
    @classmethod
    def columns(cls, match: SourceMatch) -> None:
        return None


class TestImplementation:
    def test_implementation_result_is_the_data(self, tmp_path: Path, make_implementation: Any) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = make_implementation(group, PythonDictFramework, _sentinel)

        assert impl in loaded_implementations()
        assert column_values(_run(group), "toyfmt_ld") == [111, 222]

    def test_implementation_only_applies_to_its_framework(self, tmp_path: Path, make_implementation: Any) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = make_implementation(group, PyArrowTable, _sentinel)

        assert column_values(_run(group), "toyfmt_ld") == [1, 2]
        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is None
        assert group.select_implementation(PyArrowTable, _MATCH, [impl]) is impl

    def test_implementation_on_a_subclass_group_wins(self, tmp_path: Path, make_implementation: Any) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        parent_impl = make_implementation(parent, PythonDictFramework, lambda g, m, f: [{"toyfmt_ld": 1}])

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        child_impl = make_implementation(_Child, PythonDictFramework, _sentinel)

        picked = _Child.select_implementation(PythonDictFramework, _MATCH, [parent_impl, child_impl])
        assert picked is child_impl
        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]

    def test_subclass_of_an_implementation_replaces_it(self, tmp_path: Path, make_implementation: Any) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        parent_impl = make_implementation(group, PythonDictFramework, lambda g, m, f: [{"toyfmt_ld": 1}])
        child_impl: type[Implementation[Any]] = make_implementation(
            group, PythonDictFramework, _sentinel, base=parent_impl
        )

        assert group.select_implementation(PythonDictFramework, _MATCH, [parent_impl, child_impl]) is child_impl
        assert group.select_implementation(PythonDictFramework, _MATCH, [child_impl, parent_impl]) is child_impl
        assert column_values(_run(group), "toyfmt_ld") == [111, 222]

    def test_inherited_implementation_is_found_through_the_mro(self, tmp_path: Path, make_implementation: Any) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        groups: list[Any] = []

        def recording(group: Any, match: SourceMatch, features: Any) -> Any:
            groups.append(group)
            return SENTINEL

        impl = make_implementation(parent, PythonDictFramework, recording)

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        assert _Child.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl
        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]
        assert groups == [_Child]

    def test_implementation_of_an_unavailable_framework_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_implementation: Any
    ) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl: type[Implementation[Any]] = make_implementation(group, PythonDictFramework, _sentinel)
        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl
        monkeypatch.setattr(PythonDictFramework, "is_available", staticmethod(lambda: False))

        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is None

    def test_load_neutral_override_cuts_inherited_implementations(
        self, tmp_path: Path, make_implementation: Any
    ) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        impl = make_implementation(parent, PythonDictFramework, _sentinel)

        class _Cut(parent):  # type: ignore[valid-type, misc]
            @classmethod
            def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
                return super().load_neutral(match, features)

        class _Below(_Cut):
            pass

        assert _Below.select_implementation(PythonDictFramework, _MATCH, [impl]) is None
        assert parent.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl

    def test_implementation_is_stored_on_the_feature_set_at_plan_time(
        self, tmp_path: Path, make_implementation: Any
    ) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        seen: list[Any] = []

        class _Spy(group):  # type: ignore[valid-type, misc]
            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                seen.append(features.implementation)
                return super().calculate_feature(data, features)

        spy_impl = make_implementation(_Spy, PythonDictFramework, _sentinel)
        assert column_values(_run(_Spy), "toyfmt_ld") == [111, 222]
        assert seen == [spy_impl]


class TestImplementationReceivesFramework:
    def _run_capturing(self, tmp_path: Path, extenders: Any = None) -> list[tuple[Any, Any]]:
        group = neutral_csv_group(tmp_path / "n.csv")
        seen: list[tuple[Any, Any]] = []

        class _FrameworkImpl(Implementation[Any]):
            feature_group = group
            compute_framework = PythonDictFramework

            @classmethod
            def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
                seen.append((framework, ComputeFramework.current()))
                return SENTINEL

        result = mloda.run_all(
            ["toyfmt_ld"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            data_access_collection=toy_dac(h1={"toyfmt_ld": [1, 2]}),
            function_extender=extenders,
        )
        assert column_values(result, "toyfmt_ld") == [111, 222]
        del _FrameworkImpl
        return seen

    def test_framework_instance_is_passed_without_extender(self, tmp_path: Path) -> None:
        seen = self._run_capturing(tmp_path)

        assert len(seen) == 1
        assert isinstance(seen[0][0], PythonDictFramework)

    def test_framework_instance_is_the_one_running_the_step(self, tmp_path: Path) -> None:
        capture = _LoadCapture()
        seen = self._run_capturing(tmp_path, {capture})

        assert len(capture.contexts) == 1
        assert len(seen) == 1
        framework, running = seen[0]
        assert isinstance(framework, PythonDictFramework)
        assert framework is running


class TestImplementationSeesDacConnection:
    """A DuckDB Implementation reads the DataAccessCollection connection from the framework, implicit or named."""

    @pytest.mark.parametrize("named", [False, True], ids=["implicit", "connection_handle"])
    def test_framework_connection_object_is_the_dac_connection(self, tmp_path: Path, named: bool) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        group = neutral_csv_group(tmp_path / "n.csv")
        conn = duckdb.connect()
        seen: list[Any] = []

        class _DuckImpl(Implementation[Any]):
            feature_group = group
            compute_framework = DuckDBFramework

            @classmethod
            def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
                seen.append(framework.framework_connection_object)
                conn = framework.framework_connection_object or framework.fallback_connection_object
                return DuckdbRelation.from_dict(conn or duckdb.connect(), {"toyfmt_ld": [1, 2]})

        options = {"connection_handle": "w"} if named else {}
        dac = DataAccessCollection(
            credentials={"h1": {"toyfmt_columns": {"toyfmt_ld": [1, 2]}}}, connections={"w": conn}
        )
        mloda.run_all(
            [Feature("toyfmt_ld", options=options)],
            compute_frameworks=[DuckDBFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            data_access_collection=dac,
        )
        del _DuckImpl

        assert len(seen) == 1
        assert seen[0] is conn


class TestCalculateFeatureWithoutMarker:
    def test_a_missing_load_marker_passes_no_framework(self, tmp_path: Path) -> None:
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet

        group = neutral_csv_group(tmp_path / "n.csv")
        frameworks: list[Any] = []

        class _NoneImpl(Implementation[Any]):
            feature_group = group
            compute_framework = PythonDictFramework

            @classmethod
            def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
                frameworks.append(framework)
                return SENTINEL

        feature = Feature("toyfmt_ld")
        feature.input_data_match = (group, _MATCH)
        features = FeatureSet([feature])
        features.implementation = _NoneImpl

        group.calculate_feature(None, features)
        del _NoneImpl

        assert frameworks == [None]


@pytest.fixture
def gc_disabled() -> Iterator[None]:
    gc.disable()
    yield
    gc.enable()


class TestImplementationDefinition:
    @pytest.mark.parametrize("case", ["missing_feature_group", "mismatched_feature_group"])
    def test_a_rejected_subclass_is_not_a_loaded_implementation_before_it_is_collected(
        self, case: str, gc_disabled: None
    ) -> None:
        with pytest.raises(TypeError, match="feature_group"):
            if case == "missing_feature_group":

                class _RejectedImpl(Implementation[Any]):
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            else:

                class _RejectedImpl(Implementation[OpToyFG]):  # type: ignore[no-redef]
                    feature_group = ToySourceFG
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

        assert all(impl.__name__ != "_RejectedImpl" for impl in loaded_implementations())
        assert PreFilterPlugins({PythonDictFramework}).implementations is not None

    @pytest.mark.parametrize(
        "case",
        [
            "inherited",
            "other_generic_base",
            "walk_past_orig_bases",
            "generic_intermediate",
            "reparameterized",
            "two_level_generic",
        ],
    )
    def test_feature_group_must_match_the_parameterized_group(self, case: str) -> None:
        class _Mixin(Generic[_T]):
            pass

        class _BaseImpl(Implementation[OpToyFG]):
            feature_group = OpToyFG
            compute_framework = PythonDictFramework

            @classmethod
            def calculate(cls, *args: Any) -> Any:
                return None

        if case == "inherited":
            with pytest.raises(TypeError, match="feature_group") as excinfo:

                class _Child(_BaseImpl):
                    feature_group = ToySourceFG  # type: ignore[assignment]

            assert "OpToyFG" in str(excinfo.value)
        elif case == "other_generic_base":
            with pytest.raises(TypeError, match="feature_group"):

                class _Multi(_Mixin[int], Implementation[OpToyFG]):
                    feature_group = ToySourceFG
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

        elif case == "generic_intermediate":

            class _GenericMid(Implementation[_G]):
                pass

            with pytest.raises(TypeError, match="feature_group") as excinfo:

                class _GLeaf(_GenericMid[OpToyFG]):
                    feature_group = ToySourceFG
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            assert "OpToyFG" in str(excinfo.value)
            del _GenericMid
        elif case == "reparameterized":
            with pytest.raises(TypeError, match="feature_group") as excinfo:

                class _Bar(_BaseImpl, Implementation[ToySourceFG]):
                    feature_group = ToySourceFG  # type: ignore[assignment]

            assert "OpToyFG" in str(excinfo.value)
        elif case == "two_level_generic":

            class _Outer(Implementation[_G]):
                pass

            class _Inner(_Outer[_G2]):
                pass

            with pytest.raises(TypeError, match="feature_group") as excinfo:

                class _Deep(_Inner[OpToyFG]):
                    feature_group = ToySourceFG
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            assert "OpToyFG" in str(excinfo.value)
            del _Outer
            del _Inner
        else:

            class _Mid(Implementation[OpToyFG], _Mixin[int]):
                pass

            with pytest.raises(TypeError, match="feature_group"):

                class _Leaf(_Mid, _Mixin[int]):
                    feature_group = ToySourceFG
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            del _Mid
        del _BaseImpl
        _release_subscripted_generics()

    @pytest.mark.parametrize("base", ["Implementation", "ReloadedMid"])
    def test_a_reloaded_group_is_reclaimed_while_its_module_lives(self, base: str) -> None:
        module = types.ModuleType(__name__)
        module.__dict__.update(_SOURCE_GROUP_=ToySourceFG, _G_=_G, _FRAMEWORK_=PythonDictFramework)
        source = _RELOADED_IMPL_SOURCE.format(base=base)
        exec(source, module.__dict__)  # nosec B102
        stale = weakref.ref(module.ReloadedGroup)
        exec(source, module.__dict__)  # nosec B102

        reclaimed = collect_until_dead(stale)
        _release_subscripted_generics()

        assert reclaimed

    @pytest.mark.parametrize(
        "case",
        [
            "free_typevar",
            "unsubscripted_generic",
            "bare",
            "resolved_match",
            "narrowing",
            "namedtuple_mixin",
            "generic_alias_argument",
        ],
    )
    def test_unresolved_or_matching_parameter_is_accepted(self, case: str) -> None:
        created: Any
        helper: Any = None
        if case == "free_typevar":

            class _Free(Implementation[_G]):
                feature_group = ToySourceFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            created = _Free
        elif case == "unsubscripted_generic":

            class _GenericMid(Implementation[_G]):
                pass

            class _Open(_GenericMid):  # type: ignore[type-arg]
                feature_group = ToySourceFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            helper = _GenericMid
            created = _Open
        elif case == "bare":

            class _Bare(Implementation):  # type: ignore[type-arg]
                feature_group = ToySourceFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            created = _Bare
        elif case == "resolved_match":

            class _SubOp(OpToyFG):
                pass

            class _GenericBase(Implementation[_G]):
                pass

            class _Leaf(_GenericBase[OpToyFG]):
                feature_group = _SubOp
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            helper = _GenericBase
            created = _Leaf
        elif case == "narrowing":

            class _SubFG(OpToyFG):
                pass

            class _BaseImpl(Implementation[OpToyFG]):
                feature_group = OpToyFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            class _Narrow(_BaseImpl, Implementation[_SubFG]):
                feature_group = _SubFG

            helper = _BaseImpl
            created = _Narrow
        elif case == "namedtuple_mixin":

            class _Point(NamedTuple):
                x: int

            class _PointMixin(_Point):
                pass

            class _WithTuple(_PointMixin, Implementation[OpToyFG]):
                feature_group = OpToyFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            helper = (_Point, _PointMixin)
            created = _WithTuple
        else:

            class _Listed(Implementation[list[OpToyFG]]):  # type: ignore[type-var]
                feature_group = OpToyFG
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

            created = _Listed

        assert created in loaded_implementations()
        del created
        del helper
        _release_subscripted_generics()

    @pytest.mark.parametrize("missing", ["feature_group", "compute_framework", "both"])
    def test_concrete_subclass_missing_an_attribute_raises(self, missing: str) -> None:
        attributes: dict[str, Any] = {"calculate": classmethod(lambda cls, *args: None)}
        if missing == "compute_framework":
            attributes["feature_group"] = ToySourceFG
        if missing == "feature_group":
            attributes["compute_framework"] = PythonDictFramework

        with pytest.raises(TypeError, match="feature_group" if missing != "compute_framework" else "compute_framework"):
            type("_Missing", (Implementation,), attributes)

    @pytest.mark.parametrize("kind", ["plain", "source", "operation"])
    def test_feature_group_must_be_a_source_or_operation_feature_group(
        self, kind: str, make_implementation: Any
    ) -> None:
        if kind == "plain":
            with pytest.raises(TypeError, match="feature_group"):

                class _PlainGroup(Implementation[Any]):
                    feature_group = FeatureGroup
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            return
        group = ToySourceFG if kind == "source" else _toy_operation().OpToyFG
        impl = make_implementation(group, PythonDictFramework, _sentinel)
        assert impl in loaded_implementations()

    def test_the_implementation_base_is_not_a_loaded_implementation(self) -> None:
        assert Implementation not in loaded_implementations()
        assert getattr(Implementation.calculate, "__isabstractmethod__", False)

    def test_calculate_feature_passes_group_data_features_framework(self) -> None:
        class _OpGroup(OpToyBareFG):
            pass

        seen: list[tuple[Any, ...]] = []

        class _RecordingImpl(Implementation[OpToyBareFG]):
            feature_group = _OpGroup
            compute_framework = PythonDictFramework

            @classmethod
            def calculate(
                cls, group: type[OpToyBareFG], data: Any, features: FeatureSet, framework: ComputeFramework | None
            ) -> Any:
                seen.append((group, data, features, framework))
                return "done"

        features = FeatureSet([Feature("optoy_bare_value")])
        features.implementation = _RecordingImpl
        data = object()

        assert _OpGroup.calculate_feature(data, features) == "done"
        assert seen == [(_OpGroup, data, features, None)]
        del _RecordingImpl, _OpGroup

    def test_the_typed_signature_is_enforced_by_mypy(self) -> None:
        class _OpGroup(OpToyFG):
            pass

        # mypy --strict (warn_unused_ignores) fails the gate if this wrong override stops being flagged.
        class _WrongArity(Implementation[OpToyFG]):
            feature_group = _OpGroup
            compute_framework = PythonDictFramework

            @classmethod
            def calculate(  # type: ignore[override]
                cls, data: Any, features: FeatureSet, framework: ComputeFramework | None
            ) -> Any:
                return None

        assert issubclass(_WrongArity, Implementation)
        del _WrongArity, _OpGroup

    def test_compute_framework_must_be_a_compute_framework(self) -> None:
        with pytest.raises(TypeError, match="compute_framework"):

            class _NotAFramework(Implementation[Any]):
                feature_group = ToySourceFG
                compute_framework = dict  # type: ignore[assignment]

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_subclass_leaving_calculate_abstract_is_not_validated(self, make_implementation: Any) -> None:
        loaded_implementations()

        class _IntermediateBase(Implementation[Any]):
            pass

        sibling = make_implementation(ToySourceFG, PythonDictFramework, _sentinel)

        class _HelperBase(Implementation[Any]):
            @classmethod
            def calculate(cls, *args: Any) -> Any:
                return None

            @classmethod
            @abstractmethod
            def hook(cls) -> Any: ...

        class _HelperPythonDict(_HelperBase):
            feature_group = ToyDeclaredFG
            compute_framework = PythonDictFramework

            @classmethod
            def hook(cls) -> Any:
                return None

        class _HelperPyArrow(_HelperBase):
            feature_group = ToyDeclaredFG
            compute_framework = PyArrowTable

            @classmethod
            def hook(cls) -> Any:
                return None

        assert sibling in loaded_implementations()
        assert _IntermediateBase not in loaded_implementations()
        assert _HelperBase not in loaded_implementations()
        assert {_HelperPythonDict, _HelperPyArrow} <= set(loaded_implementations())
        assert PreFilterPlugins({PythonDictFramework, PyArrowTable}).implementations is not None
        del _HelperPythonDict, _HelperPyArrow, _HelperBase

    def test_loaded_implementations_does_not_rescan_known_classes(
        self, monkeypatch: pytest.MonkeyPatch, make_implementation: Any
    ) -> None:
        import mloda.core.abstract_plugins.implementation as impl_module

        original = impl_module._has_abstract_member
        calls: list[type] = []
        original_pair = impl_module._names_a_pair
        pair_calls: list[str] = []

        def counting(cls: type) -> bool:
            calls.append(cls)
            return bool(original(cls))

        def counting_pair(cls: Any) -> Any:
            pair_calls.append(cls.__name__)
            return original_pair(cls)

        monkeypatch.setattr(impl_module, "_has_abstract_member", counting)
        monkeypatch.setattr(impl_module, "_names_a_pair", counting_pair)
        loaded_implementations()
        calls.clear()
        pair_calls.clear()
        loaded_implementations()
        assert calls == []
        assert pair_calls == []

        new_impl = make_implementation(ToySourceFG, PythonDictFramework, _sentinel)
        calls.clear()
        pair_calls.clear()
        assert new_impl in loaded_implementations()
        assert pair_calls == [new_impl.__name__]
        pair_calls.clear()
        loaded_implementations()
        assert pair_calls == []
        calls.clear()
        del new_impl

    def test_class_made_concrete_by_a_decorator_is_loaded(self) -> None:
        def add_calculate(cls: Any) -> Any:
            cls.calculate = classmethod(lambda c, serving, data, features, framework: None)
            return cls

        @add_calculate
        class _Decorated(Implementation[Any]):
            feature_group = ToySourceFG
            compute_framework = PyArrowTable

        assert _Decorated in loaded_implementations()
        del _Decorated


def _run_operation(
    feature_name: str,
    framework: Any = None,
    group: Any = None,
    disabled: Any = (),
    kind: str = "sum",
) -> list[Any]:
    toy = _toy_operation()
    feature = Feature(
        feature_name,
        {toy.OPTOY_KEY: kind},
        compute_framework=None if framework is None else framework.get_class_name(),
    )
    collector = PluginCollector.enabled_feature_groups({group or toy.OpToyFG})
    collector.add_disabled_implementation_classes(set(disabled))
    return mloda.run_all(
        [feature],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=collector,
    )


class TestOperationGroupImplementation:
    def test_unpinned_run_with_two_frameworks_resolves_to_one_group(self) -> None:
        toy = _toy_operation()

        result = _run_operation(toy.OPTOY_FEATURE)

        assert column_values(result, toy.OPTOY_FEATURE) == [toy.PYARROW_MARK]

    def test_a_matching_implementation_is_preferred_to_a_divergent_one_listed_first(self) -> None:
        toy = _toy_operation()

        steps = mloda.explain(
            [Feature(toy.OPTOY_FEATURE, {toy.OPTOY_KEY: "sum"})],
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({toy.OpToyFG}),
        )
        step = next(s for s in steps if s.step_kind == "compute")

        assert step.compute_framework_name == "PyArrowTable"
        assert step.compute_framework_reason == "prefers a matching version"

    def test_a_subclass_group_is_served_by_the_parent_implementation_with_the_subclass_as_group(self) -> None:
        toy = _toy_operation()
        toy.OpToyPandas.served_by.clear()

        result = _run_operation(toy.OPTOY_CHILD_FEATURE, PandasDataFrame, group=toy.OpToyChildFG)

        assert column_values(result, toy.OPTOY_CHILD_FEATURE) == [toy.PANDAS_MARK]
        assert toy.OpToyPandas.served_by == [toy.OpToyChildFG]

    def test_an_operation_group_without_an_implementation_is_never_a_candidate(self) -> None:
        toy = _toy_operation()

        assert toy.OpToyBareFG.compute_framework_definition() == set()
        with pytest.raises(ValueError, match="no Implementation of it is loaded; import its implementation modules"):
            _run_operation(toy.OPTOY_BARE_FEATURE, group=toy.OpToyBareFG)

    def test_a_framework_without_an_implementation_is_not_in_the_definition(self, make_implementation: Any) -> None:
        toy = _toy_operation()

        definition = toy.OpToyFG.compute_framework_definition()
        assert {PandasDataFrame, PyArrowTable} <= definition
        assert PythonDictFramework not in definition
        dict_impl = make_implementation(toy.OpToyFG, PythonDictFramework, _sentinel)
        assert PythonDictFramework in toy.OpToyFG.compute_framework_definition()
        assert dict_impl.feature_group is toy.OpToyFG

    def test_a_disabled_implementation_removes_its_framework_from_the_run(self) -> None:
        toy = _toy_operation()

        result = _run_operation(toy.OPTOY_FEATURE, disabled={toy.OpToyPyArrow})

        assert column_values(result, toy.OPTOY_FEATURE) == [toy.PANDAS_MARK]
        with pytest.raises(ValueError):
            _run_operation(toy.OPTOY_FEATURE, PyArrowTable, disabled={toy.OpToyPyArrow})

    def test_disabling_every_implementation_reports_disabled_or_denied(self) -> None:
        toy = _toy_operation()

        with pytest.raises(ValueError, match="its implementations are all disabled or denied for this run"):
            _run_operation(toy.OPTOY_FEATURE, disabled={toy.OpToyPandas, toy.OpToyPyArrow})

    def test_divergence_reports_the_declared_reason_for_its_subtype_only(self) -> None:
        toy = _toy_operation()
        sum_options = Options({toy.OPTOY_KEY: "sum"})
        median_options = Options({toy.OPTOY_KEY: "median"})

        assert toy.OpToyPandas.divergence(toy.OpToyFG, toy.OPTOY_FEATURE, sum_options) == "skips NaN in sum"
        assert toy.OpToyPandas.divergence(toy.OpToyFG, toy.OPTOY_FEATURE, median_options) is None


INNER = "optoy_chain_src__optoy_chain"
OUTER = "optoy_chain_src__optoy_chain__optoy_chain"


def _run_pinned(
    features: list[Feature],
    groups: Any = None,
    frameworks: Any = (PandasDataFrame, PyArrowTable),
    disabled: Any = (),
) -> list[Any]:
    toy = _toy_operation()
    collector = PluginCollector.enabled_feature_groups(groups or {toy.OpToyFG})
    collector.add_disabled_implementation_classes(set(disabled))
    return mloda.run_all(features, compute_frameworks=list(frameworks), plugin_collector=collector)


def _run_chain(features: list[Feature], **kwargs: Any) -> list[Any]:
    toy = _toy_operation()
    toy.CHAIN_LOG.clear()
    return _run_pinned(features, groups={toy.OpToyChainSourceFG, toy.OpToyChainFG}, **kwargs)


def _op_feature(pin: Any, **kwargs: Any) -> Feature:
    toy = _toy_operation()
    return Feature(toy.OPTOY_FEATURE, {toy.OPTOY_KEY: "sum"}, implementation=pin, **kwargs)


class TestImplementationPin:
    @pytest.mark.parametrize(
        ("pin", "framework", "mark"),
        [
            ("OpToyPyArrow", None, 2),
            ("OpToyPandas", None, 1),
            ("toy_class_pyarrow", None, 2),
            ("toy_class_pandas", None, 1),
            (None, PandasDataFrame, 1),
            (None, PyArrowTable, 2),
        ],
    )
    def test_the_pin_selects_the_implementation_that_runs(self, pin: str | None, framework: Any, mark: int) -> None:
        toy = _toy_operation()
        classes: dict[str | None, Any] = {"toy_class_pyarrow": toy.OpToyPyArrow, "toy_class_pandas": toy.OpToyPandas}

        if pin is None:
            result = _run_operation(toy.OPTOY_FEATURE, framework)
        else:
            result = _run_pinned([_op_feature(classes.get(pin, pin))])

        assert column_values(result, toy.OPTOY_FEATURE) == [mark]

    def test_naming_a_parent_implementation_lets_a_subclass_implementation_serve(
        self, make_implementation: Any
    ) -> None:
        toy = _toy_operation()
        plus = make_implementation(
            toy.OpToyFG,
            PandasDataFrame,
            lambda g, m, f: pd.DataFrame({toy.OPTOY_FEATURE: [77]}),
            base=toy.OpToyPandas,
            name="OpToyPandasPlus",
        )

        result = _run_pinned([_op_feature("OpToyPandas")])

        assert column_values(result, toy.OPTOY_FEATURE) == [77]
        assert issubclass(plus, toy.OpToyPandas)

    @pytest.mark.parametrize(
        ("pin", "expected"),
        [
            ("OpToyChainPyArrow", [("INNER", "OpToyChainPyArrow"), ("OUTER", "OpToyChainPyArrow")]),
            (
                ["OpToyChainPandas", "OpToyChainPyArrow"],
                [("INNER", "OpToyChainPandas"), ("OUTER", "OpToyChainPyArrow")],
            ),
            (
                ["OpToyChainPyArrow", "OpToyChainPandas"],
                [("INNER", "OpToyChainPyArrow"), ("OUTER", "OpToyChainPandas")],
            ),
        ],
        ids=["one_name_pins_both", "inner_pandas_outer_pyarrow", "inner_pyarrow_outer_pandas"],
    )
    def test_a_chain_pins_by_input_ancestry_order(self, pin: Any, expected: list[tuple[str, str]]) -> None:
        _run_chain([Feature(OUTER, implementation=pin)])

        toy = _toy_operation()
        names = {"INNER": INNER, "OUTER": OUTER}
        assert toy.CHAIN_LOG == [(names[step], impl) for step, impl in expected]

    @pytest.mark.parametrize(
        ("case", "needle"),
        [
            ("unknown", "NoSuchImpl"),
            ("disabled", "OpToyPyArrow"),
            ("framework_not_enabled", "OpToyPyArrow"),
            ("keys_no_group_of_chain", "OpToyChainPandas"),
            ("conflicts_with_compute_framework", "OpToyPyArrow"),
            ("too_many_names", "OpToyChainPandas"),
            ("source_group_overrides_load_neutral", "FmtParentImpl"),
        ],
    )
    def test_a_bad_pin_raises_implementation_pin_error_naming_step_and_class(
        self, case: str, needle: str, tmp_path: Path, make_implementation: Any
    ) -> None:
        from mloda.provider import ImplementationPinError

        toy = _toy_operation()
        assert issubclass(ImplementationPinError, ValueError)
        feature_name = toy.OPTOY_FEATURE
        with pytest.raises(ImplementationPinError) as caught:
            if case == "unknown":
                _run_pinned([_op_feature("NoSuchImpl")])
            elif case == "disabled":
                _run_pinned([_op_feature("OpToyPyArrow")], disabled={toy.OpToyPyArrow})
            elif case == "framework_not_enabled":
                _run_pinned([_op_feature("OpToyPyArrow")], frameworks=(PandasDataFrame,))
            elif case == "keys_no_group_of_chain":
                _run_pinned([_op_feature("OpToyChainPandas")])
            elif case == "conflicts_with_compute_framework":
                _run_pinned([_op_feature("OpToyPyArrow", compute_framework="PandasDataFrame")])
            elif case == "source_group_overrides_load_neutral":
                feature_name = "toyfmt_ld"
                parent = neutral_csv_group(tmp_path / "n.csv")
                parent_impl = make_implementation(parent, PythonDictFramework, _sentinel, name="FmtParentImpl")

                class _Child(parent):  # type: ignore[valid-type, misc]
                    @classmethod
                    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
                        return super().load_neutral(match, features)

                mloda.run_all(
                    [Feature("toyfmt_ld", implementation=parent_impl)],
                    compute_frameworks=[PythonDictFramework],
                    plugin_collector=PluginCollector.enabled_feature_groups({_Child}),
                    data_access_collection=toy_dac(h1={"toyfmt_ld": [1, 2]}),
                )
            else:
                feature_name = OUTER
                _run_chain([Feature(OUTER, implementation=["OpToyChainPandas"] * 3)])

        assert feature_name in str(caught.value)
        assert needle in str(caught.value)

    def test_two_requests_pinning_a_shared_step_differently_raise_naming_both(self) -> None:
        from mloda.provider import ImplementationPinError

        features = [
            Feature(OUTER, implementation="OpToyChainPandas"),
            Feature(INNER, implementation="OpToyChainPyArrow"),
        ]

        with pytest.raises(ImplementationPinError) as caught:
            _run_chain(features)

        assert OUTER in str(caught.value)
        assert INNER in str(caught.value)

    def test_a_parent_and_a_subclass_pin_resolving_to_one_serving_class_do_not_conflict(
        self, make_implementation: Any
    ) -> None:
        toy = _toy_operation()

        def serve(group: Any, data: Any, features: Any) -> Any:
            for name in sorted(features.get_all_names()):
                toy.CHAIN_LOG.append((name, "OpToyChainPandasPlus"))
                data[name] = toy.PANDAS_MARK
            return data

        plus = make_implementation(
            toy.OpToyChainFG, PandasDataFrame, serve, base=toy.OpToyChainPandas, name="OpToyChainPandasPlus"
        )

        _run_chain(
            [
                Feature(OUTER, implementation="OpToyChainPandas"),
                Feature(INNER, implementation="OpToyChainPandasPlus"),
            ]
        )

        assert toy.CHAIN_LOG == [(INNER, "OpToyChainPandasPlus"), (OUTER, "OpToyChainPandasPlus")]
        assert plus.__name__ == "OpToyChainPandasPlus"

    def test_a_class_pin_resolves_although_another_loaded_implementation_shares_its_name(
        self, make_implementation: Any
    ) -> None:
        toy = _toy_operation()
        twin = make_implementation(
            toy.OpToyFG,
            PandasDataFrame,
            lambda g, m, f: pd.DataFrame({toy.OPTOY_FEATURE: [77]}),
            base=toy.OpToyPandas,
            name="OpToyPyArrow",
        )

        result = _run_pinned([_op_feature(toy.OpToyPyArrow)])

        assert column_values(result, toy.OPTOY_FEATURE) == [toy.PYARROW_MARK]
        assert twin is not toy.OpToyPyArrow

    def test_the_pin_does_not_reach_the_input_or_a_non_pinned_sibling(self) -> None:
        toy = _toy_operation()
        sibling = Feature(toy.OPTOY_CHAIN_SRC)

        result = _run_chain([Feature(INNER, implementation="OpToyChainPyArrow"), sibling])

        assert toy.CHAIN_LOG == [(INNER, "OpToyChainPyArrow")]
        assert sibling.implementation_pin is None
        assert column_values(result, toy.OPTOY_CHAIN_SRC) == [1, 2, 3]


class TestFrameworkSubclassImplementation:
    def test_implementation_for_a_framework_serves_its_subclass(self, tmp_path: Path, make_implementation: Any) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = make_implementation(group, PythonDictFramework, _sentinel)

        class _SubDictFramework(PythonDictFramework):
            pass

        assert group.select_implementation(_SubDictFramework, _MATCH, [impl]) is impl

    def test_load_hook_names_the_implementation_not_the_subclass_framework(self) -> None:
        class _SubDictFramework(PythonDictFramework):
            pass

        capture = _LoadCapture()
        result = mloda.run_all(
            [WORKER_COLUMN],
            compute_frameworks=[_SubDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyWorkerFG}),
            data_access_collection=toy_dac(h1={WORKER_COLUMN: [1, 2]}),
            function_extender={capture},
        )

        assert column_values(result, WORKER_COLUMN) == WORKER_VALUES
        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        expected = f"{ToyWorkerPythonDict.__module__}:{ToyWorkerPythonDict.__qualname__}"
        assert context.data_access_loader == expected
        assert context.implementation == expected
        assert context.implementation_version == ToyWorkerPythonDict.version()


@pytest.mark.timeout(30)
class TestImplementationInASpawnWorker:
    @pytest.mark.subprocess
    def test_implementation_from_the_groups_helper_module_serves_sync_and_multiprocessing(
        self, flight_server: Any
    ) -> None:
        def run(mode: ParallelizationMode) -> Any:
            return mloda.run_all(
                [WORKER_COLUMN],
                compute_frameworks=[PythonDictFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({ToyWorkerFG}),
                data_access_collection=toy_dac(h1={WORKER_COLUMN: [1, 2]}),
                parallelization_modes={mode},
                flight_server=flight_server,
            )

        assert ToyWorkerPythonDict in loaded_implementations()
        sync = column_values(run(ParallelizationMode.SYNC), WORKER_COLUMN)
        multiprocessing = column_values(run(ParallelizationMode.MULTIPROCESSING), WORKER_COLUMN)

        assert sync == WORKER_VALUES
        assert multiprocessing == WORKER_VALUES


class TestNeutralFallback:
    def test_neutral_form_is_converted_by_the_existing_transformer(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")

        assert column_values(_run(group), "toyfmt_ld") == [1, 2]

    def test_a_default_run_loads_a_neutral_group_natively_for_mixed_types(self) -> None:
        result = mloda.run_all(
            ["toyfmt_mixed"],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyBoundedNeutralDictFG}),
            data_access_collection=toy_dac(h1={"toyfmt_mixed": [1, "a"]}),
        )

        assert isinstance(result[0], dict)
        assert result[0]["toyfmt_mixed"] == [1, "a"]


class TestTools:
    def _step_group_and_match(self, group: type, feature: str) -> tuple[Any, Any]:
        requested = Feature(feature)
        resolve_or_raise(
            requested,
            {group: {PythonDictFramework}},
            None,
            toy_dac(h1={"toyfmt_a": [1], "toyfmt_b": [2]}),
        )
        assert requested.input_data_match is not None
        return group, requested.input_data_match[1]

    def test_describe_columns_defaults_to_columns_with_none_types(self) -> None:
        fg, match = self._step_group_and_match(ToySourceFG, "toyfmt_a")

        assert fg.describe_columns(match) == {"toyfmt_a": None, "toyfmt_b": None}

    def test_describe_columns_raises_when_columns_cannot_enumerate(self) -> None:
        fg, match = self._step_group_and_match(_BlindDeclaredFG, "toyfmt_declared_col")

        with pytest.raises(NotImplementedError):
            fg.describe_columns(match)

    def test_count_rows_defaults_to_none(self) -> None:
        fg, match = self._step_group_and_match(ToySourceFG, "toyfmt_a")

        assert fg.count_rows(match, PythonDictFramework) is None

    def test_match_connection_defaults_to_none(self) -> None:
        assert SourceFeatureGroup.match_connection(_MATCH) is None

    def test_neutral_type_defaults_to_none(self) -> None:
        assert SourceFeatureGroup.neutral_type(_MATCH) is None
