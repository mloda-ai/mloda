"""FormatFeatureGroup loaders, neutral fallback and the describe_columns/count_rows tools."""

import gc
import importlib
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation import (
    WORKER_COLUMN,
    WORKER_VALUES,
    ToyWorkerFG,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation_impl import (
    ToyWorkerPythonDict,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyDeclaredFG,
    ToyFormatFG,
    column_values,
    neutral_csv_group,
    toy_dac,
)

SENTINEL = [{"toyfmt_ld": 111}, {"toyfmt_ld": 222}]


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
        child_impl: type[Implementation] = make_implementation(group, PythonDictFramework, _sentinel, base=parent_impl)

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
        impl: type[Implementation] = make_implementation(group, PythonDictFramework, _sentinel)
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


@pytest.fixture
def gc_disabled() -> Iterator[None]:
    gc.disable()
    yield
    gc.enable()


class TestImplementationDefinition:
    def test_a_rejected_subclass_is_not_a_loaded_implementation_before_it_is_collected(self, gc_disabled: None) -> None:
        with pytest.raises(TypeError, match="feature_group"):

            class _RejectedImpl(Implementation):
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

        assert all(impl.__name__ != "_RejectedImpl" for impl in loaded_implementations())
        assert PreFilterPlugins({PythonDictFramework}).implementations is not None

    @pytest.mark.parametrize("missing", ["feature_group", "compute_framework", "both"])
    def test_concrete_subclass_missing_an_attribute_raises(self, missing: str) -> None:
        attributes: dict[str, Any] = {"calculate": classmethod(lambda cls, *args: None)}
        if missing == "compute_framework":
            attributes["feature_group"] = ToyFormatFG
        if missing == "feature_group":
            attributes["compute_framework"] = PythonDictFramework

        with pytest.raises(TypeError, match="feature_group" if missing != "compute_framework" else "compute_framework"):
            type("_Missing", (Implementation,), attributes)

    @pytest.mark.parametrize("kind", ["plain", "format", "operation"])
    def test_feature_group_must_be_a_format_or_operation_feature_group(
        self, kind: str, make_implementation: Any
    ) -> None:
        if kind == "plain":
            with pytest.raises(TypeError, match="feature_group"):

                class _PlainGroup(Implementation):
                    feature_group = FeatureGroup
                    compute_framework = PythonDictFramework

                    @classmethod
                    def calculate(cls, *args: Any) -> Any:
                        return None

            return
        group = ToyFormatFG if kind == "format" else _toy_operation().OpToyFG
        impl = make_implementation(group, PythonDictFramework, _sentinel)
        assert impl in loaded_implementations()

    def test_compute_framework_must_be_a_compute_framework(self) -> None:
        with pytest.raises(TypeError, match="compute_framework"):

            class _NotAFramework(Implementation):
                feature_group = ToyFormatFG
                compute_framework = dict  # type: ignore[assignment]

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_subclass_leaving_calculate_abstract_is_not_validated(self, make_implementation: Any) -> None:
        class _IntermediateBase(Implementation):
            pass

        sibling = make_implementation(ToyFormatFG, PythonDictFramework, _sentinel)

        class _HelperBase(Implementation):
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
        gc.collect()


@pytest.fixture(autouse=True, scope="module")
def _collect_local_classes() -> Iterator[None]:
    yield
    gc.collect()


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


class TestOperationImplementation:
    @pytest.mark.parametrize("framework, mark", [(PandasDataFrame, 1), (PyArrowTable, 2)])
    def test_one_group_runs_the_implementation_of_the_chosen_framework(self, framework: Any, mark: int) -> None:
        toy = _toy_operation()

        result = _run_operation(toy.OPTOY_FEATURE, framework)

        assert column_values(result, toy.OPTOY_FEATURE) == [mark]

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
        with pytest.raises(ValueError):
            _run_operation(toy.OPTOY_BARE_FEATURE, group=toy.OpToyBareFG)

    def test_a_framework_without_an_implementation_is_not_in_the_definition(self, make_implementation: Any) -> None:
        toy = _toy_operation()

        definition = toy.OpToyFG.compute_framework_definition()
        assert {PandasDataFrame, PyArrowTable} <= definition
        assert PythonDictFramework not in definition
        make_implementation(toy.OpToyFG, PythonDictFramework, _sentinel)
        assert PythonDictFramework in toy.OpToyFG.compute_framework_definition()

    def test_a_disabled_implementation_removes_its_framework_from_the_run(self) -> None:
        toy = _toy_operation()

        result = _run_operation(toy.OPTOY_FEATURE, disabled={toy.OpToyPyArrow})

        assert column_values(result, toy.OPTOY_FEATURE) == [toy.PANDAS_MARK]
        with pytest.raises(ValueError):
            _run_operation(toy.OPTOY_FEATURE, PyArrowTable, disabled={toy.OpToyPyArrow})

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
        ("pin", "mark"),
        [("OpToyPyArrow", 2), ("OpToyPandas", 1), ("toy_class_pyarrow", 2), ("toy_class_pandas", 1)],
    )
    def test_the_pin_selects_the_implementation_that_runs(self, pin: str, mark: int) -> None:
        toy = _toy_operation()
        pinned: Any = {"toy_class_pyarrow": toy.OpToyPyArrow, "toy_class_pandas": toy.OpToyPandas}.get(pin, pin)

        result = _run_pinned([_op_feature(pinned)])

        assert column_values(result, toy.OPTOY_FEATURE) == [mark]

    def test_naming_a_parent_implementation_lets_a_subclass_implementation_serve(
        self, make_implementation: Any
    ) -> None:
        toy = _toy_operation()
        make_implementation(
            toy.OpToyFG,
            PandasDataFrame,
            lambda g, m, f: pd.DataFrame({toy.OPTOY_FEATURE: [77]}),
            base=toy.OpToyPandas,
            name="OpToyPandasPlus",
        )

        result = _run_pinned([_op_feature("OpToyPandas")])

        assert column_values(result, toy.OPTOY_FEATURE) == [77]

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
        ],
    )
    def test_a_bad_pin_raises_implementation_pin_error_naming_step_and_class(self, case: str, needle: str) -> None:
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
        fg, match = self._step_group_and_match(ToyFormatFG, "toyfmt_a")

        assert fg.describe_columns(match) == {"toyfmt_a": None, "toyfmt_b": None}

    def test_describe_columns_raises_when_columns_cannot_enumerate(self) -> None:
        fg, match = self._step_group_and_match(_BlindDeclaredFG, "toyfmt_declared_col")

        with pytest.raises(NotImplementedError):
            fg.describe_columns(match)

    def test_count_rows_defaults_to_none(self) -> None:
        fg, match = self._step_group_and_match(ToyFormatFG, "toyfmt_a")

        assert fg.count_rows(match, PythonDictFramework) is None
