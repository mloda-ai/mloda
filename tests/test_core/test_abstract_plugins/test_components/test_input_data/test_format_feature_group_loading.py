"""FormatFeatureGroup loaders, neutral fallback and the describe_columns/count_rows tools."""

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation import (
    WORKER_COLUMN,
    WORKER_VALUES,
    ToyWorkerFG,
)
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyDeclaredFG,
    ToyFormatFG,
    column_values,
    neutral_csv_group,
    toy_dac,
)

SENTINEL = [{"toyfmt_ld": 111}, {"toyfmt_ld": 222}]


def _run(group: type, frameworks: Any = None, column: str = "toyfmt_ld") -> list[Any]:
    return mloda.run_all(
        [column],
        compute_frameworks=frameworks or [PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        data_access_collection=toy_dac(h1={column: [1, 2]}),
    )


def _impl(
    group: Any, framework: Any, fn: Any = None, base: type[Implementation] = Implementation
) -> type[Implementation]:
    """A fresh Implementation class for (group, framework); callers pass test-only groups, never a stock pair."""

    class _Impl(base):  # type: ignore[misc, valid-type]
        feature_group = group
        compute_framework = framework

        @classmethod
        def calculate(cls, serving: Any, match: SourceMatch, features: Any) -> Any:
            return SENTINEL if fn is None else fn(serving, match, features)

    return _Impl


_MATCH = SourceMatch(source="toyfmt_src", access="toyfmt_src")


class _BlindDeclaredFG(ToyDeclaredFG):
    @classmethod
    def columns(cls, match: SourceMatch) -> None:
        return None


class TestImplementation:
    def test_implementation_result_is_the_data(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        _impl(group, PythonDictFramework)

        assert column_values(_run(group), "toyfmt_ld") == [111, 222]

    def test_implementation_only_applies_to_its_framework(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = _impl(group, PyArrowTable)

        assert column_values(_run(group), "toyfmt_ld") == [1, 2]
        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is None
        assert group.select_implementation(PyArrowTable, _MATCH, [impl]) is impl

    def test_implementation_on_a_subclass_group_wins(self, tmp_path: Path) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        parent_impl = _impl(parent, PythonDictFramework, lambda g, m, f: [{"toyfmt_ld": 1}])

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        child_impl = _impl(_Child, PythonDictFramework)

        picked = _Child.select_implementation(PythonDictFramework, _MATCH, [parent_impl, child_impl])
        assert picked is child_impl
        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]

    def test_subclass_of_an_implementation_replaces_it(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        parent_impl = _impl(group, PythonDictFramework, lambda g, m, f: [{"toyfmt_ld": 1}])
        child_impl = _impl(group, PythonDictFramework, base=parent_impl)

        assert group.select_implementation(PythonDictFramework, _MATCH, [parent_impl, child_impl]) is child_impl
        assert group.select_implementation(PythonDictFramework, _MATCH, [child_impl, parent_impl]) is child_impl
        assert column_values(_run(group), "toyfmt_ld") == [111, 222]

    def test_inherited_implementation_is_found_through_the_mro(self, tmp_path: Path) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        groups: list[Any] = []

        def recording(group: Any, match: SourceMatch, features: Any) -> Any:
            groups.append(group)
            return SENTINEL

        impl = _impl(parent, PythonDictFramework, recording)

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        assert _Child.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl
        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]
        assert groups == [_Child]

    def test_implementation_of_an_unavailable_framework_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = _impl(group, PythonDictFramework)
        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl
        monkeypatch.setattr(PythonDictFramework, "is_available", staticmethod(lambda: False))

        assert group.select_implementation(PythonDictFramework, _MATCH, [impl]) is None

    def test_load_neutral_override_cuts_inherited_implementations(self, tmp_path: Path) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        impl = _impl(parent, PythonDictFramework)

        class _Cut(parent):  # type: ignore[valid-type, misc]
            @classmethod
            def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
                return super().load_neutral(match, features)

        class _Below(_Cut):
            pass

        assert _Below.select_implementation(PythonDictFramework, _MATCH, [impl]) is None
        assert parent.select_implementation(PythonDictFramework, _MATCH, [impl]) is impl

    def test_implementation_is_stored_on_the_feature_set_at_plan_time(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        seen: list[Any] = []

        class _Spy(group):  # type: ignore[valid-type, misc]
            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                seen.append(features.implementation)
                return super().calculate_feature(data, features)

        spy_impl = _impl(_Spy, PythonDictFramework)
        assert column_values(_run(_Spy), "toyfmt_ld") == [111, 222]
        assert seen == [spy_impl]


class TestImplementationDefinition:
    def test_concrete_subclass_without_attributes_raises(self) -> None:
        with pytest.raises(TypeError):

            class _NoAttrs(Implementation):
                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_concrete_subclass_without_compute_framework_raises(self) -> None:
        with pytest.raises(TypeError):

            class _NoFramework(Implementation):
                feature_group = ToyFormatFG

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_concrete_subclass_without_feature_group_raises(self) -> None:
        with pytest.raises(TypeError):

            class _NoGroup(Implementation):
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_feature_group_must_be_a_format_feature_group(self) -> None:
        with pytest.raises(TypeError):

            class _PlainGroup(Implementation):
                feature_group = FeatureGroup
                compute_framework = PythonDictFramework

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_compute_framework_must_be_a_compute_framework(self) -> None:
        with pytest.raises(TypeError):

            class _NotAFramework(Implementation):
                feature_group = ToyFormatFG
                compute_framework = dict  # type: ignore[assignment]

                @classmethod
                def calculate(cls, *args: Any) -> Any:
                    return None

    def test_subclass_leaving_calculate_abstract_is_not_validated(self) -> None:
        class _IntermediateBase(Implementation):
            pass

        assert _IntermediateBase not in loaded_implementations()


class TestFrameworkSubclassImplementation:
    def test_implementation_for_a_framework_serves_its_subclass(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = _impl(group, PythonDictFramework)

        class _SubDictFramework(PythonDictFramework):
            pass

        assert group.select_implementation(_SubDictFramework, _MATCH, [impl]) is impl


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
