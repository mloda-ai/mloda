"""A link-driven framework rewrite rebinds queue Features, not the SingleFilters GlobalFilter stores."""

from collections import defaultdict
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.core.engine import Engine
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import (
    Feature,
    FeatureName,
    Features,
    GlobalFilter,
    Index,
    JoinSpec,
    Link,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner
from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw


class CfwStrandLeftSrc(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"cfwstr_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"cfwstr_left": [1, 2, 3], "cfwstr_idx": ["a", "b", "c"]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SecondCfw}


class CfwStrandRightSrc(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"cfwstr_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"cfwstr_right": [4, 5, 6], "cfwstr_idx": ["a", "b", "c"]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SecondCfw}


class CfwStrandConsumer(FeatureGroup):
    """Link child whose frameworks get narrowed by the link resolution."""

    SUPPORTED = frozenset({"cfwstr_a", "cfwstr_b", "cfwstr_ts"})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) in cls.SUPPORTED

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("cfwstr_left"), Feature("cfwstr_right")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"cfwstr_a": [1, 2, 3], "cfwstr_b": [4, 5, 6], "cfwstr_ts": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, SecondCfw}


_ENABLED = PluginCollector.enabled_feature_groups({CfwStrandLeftSrc, CfwStrandRightSrc, CfwStrandConsumer})


def test_link_cfw_rewrite_keeps_stored_filters_usable() -> None:
    """The run must succeed and apply the filter."""
    idx = Index(("cfwstr_idx",))
    links = {Link("inner", JoinSpec(CfwStrandLeftSrc, idx), JoinSpec(CfwStrandRightSrc, idx))}

    global_filter = GlobalFilter()
    global_filter.add_filter("cfwstr_ts", "min", {"value": 15})

    features = Features([Feature("cfwstr_a"), Feature("cfwstr_b")])

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks=[PyArrowTable, SecondCfw],
        parallelization_modes={ParallelizationMode.SYNC},
        global_filter=global_filter,
        links=links,
        plugin_collector=_ENABLED,
    )

    merged: dict[str, list[Any]] = {}
    for res in result.results:
        merged.update(res.to_pydict())

    assert merged["cfwstr_a"] == [2, 3]
    assert merged["cfwstr_b"] == [5, 6]

    # Every stored SingleFilter must still be findable in its own hash-keyed set.
    for stored_set in global_filter.collection.values():
        for single_filter in stored_set:
            assert single_filter in stored_set

    # Stored filter features are GlobalFilter's own, so the planner's narrowing never reaches them.
    candidate_sets: set[frozenset[type[ComputeFramework]]] = {
        frozenset(single_filter.filter_feature.compute_frameworks or ())
        for stored_set in global_filter.collection.values()
        for single_filter in stored_set
    }
    assert candidate_sets == {frozenset({PyArrowTable, SecondCfw})}, (
        f"the planner must not narrow a stored filter feature: {candidate_sets!r}"
    )


class PinHostFG(FeatureGroup):
    """Unrestricted root hosting a filterable column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"pinhost_val", "pinhost_ts"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"pinhost_val": [1, 2, 3], "pinhost_ts": [10, 20, 30]})


_PIN_ENABLED = PluginCollector.enabled_feature_groups({PinHostFG})


def _pinned_filter() -> GlobalFilter:
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature("pinhost_ts", compute_framework="PyArrowTable"), "min", {"value": 15})
    return global_filter


def _host_frameworks(global_filter: GlobalFilter | None) -> list[str | None]:
    steps = mloda.explain(
        ["pinhost_val"],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        global_filter=global_filter,
        plugin_collector=_PIN_ENABLED,
    )
    return [step.compute_framework_name for step in steps if step.step_kind == "compute"]


def test_a_filter_pinned_to_a_framework_moves_its_host_onto_the_pin() -> None:
    assert _host_frameworks(None) == ["PandasDataFrame"]
    assert _host_frameworks(_pinned_filter()) == ["PyArrowTable"]


def test_a_pinned_filter_is_applied_on_the_pinned_host() -> None:
    result = mloda.run_all(
        ["pinhost_val"],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        global_filter=_pinned_filter(),
        plugin_collector=_PIN_ENABLED,
    )

    assert [res.to_pydict()["pinhost_val"] for res in result] == [[2, 3]]


def _pinned_self_filter() -> GlobalFilter:
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature("pinhost_val", compute_framework="PyArrowTable"), "min", {"value": 2})
    return global_filter


def test_a_filter_pinned_on_the_requested_column_plans_and_applies() -> None:
    frameworks: list[type[ComputeFramework]] = [PandasDataFrame, PyArrowTable]

    result = mloda.run_all(
        ["pinhost_val"],
        compute_frameworks=frameworks,
        global_filter=_pinned_self_filter(),
        plugin_collector=_PIN_ENABLED,
    )

    assert [res.to_pydict()["pinhost_val"] for res in result] == [[2, 3]]


def test_a_filter_pinned_on_the_requested_column_runs_its_host_on_the_pin() -> None:
    steps = mloda.explain(
        ["pinhost_val"],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        global_filter=_pinned_self_filter(),
        plugin_collector=_PIN_ENABLED,
    )

    assert [s.compute_framework_name for s in steps if s.step_kind == "compute"] == ["PyArrowTable"]


@pytest.mark.parametrize("pinned_first", [True, False], ids=["pinned_first", "plain_first"])
def test_a_pinned_twin_of_a_requested_column_survives_a_pinned_filter(pinned_first: bool) -> None:
    twins: list[Feature | str] = [Feature("pinhost_val", compute_framework="PyArrowTable"), Feature("pinhost_val")]
    result = mloda.run_all(
        twins if pinned_first else twins[::-1],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        global_filter=_pinned_filter(),
        plugin_collector=_PIN_ENABLED,
    )

    assert [res.to_pydict()["pinhost_val"] for res in result] == [[2, 3]]


def _bare_engine() -> Engine:
    engine = Engine.__new__(Engine)
    engine.feature_group_collection = defaultdict(set)
    engine.specialized_from = {}
    engine.resolved_input_feature_names = {}
    engine._declared_options_by_uuid = {}
    engine.feature_link_parents = defaultdict(set)
    return engine


def _unpinned_host(name: str) -> Feature:
    host = Feature(name)
    host.compute_frameworks = {PandasDataFrame, PyArrowTable}
    return host


def test_narrowing_a_host_onto_a_pinned_filter_flags_it_pinned() -> None:
    engine = _bare_engine()
    host = _unpinned_host("narrow_flag_host")
    engine.feature_group_collection[PinHostFG].add(host)
    assert host.framework_pinned is False

    returned = engine._narrow_host_to_pin(PinHostFG, host, Feature("narrow_flag_ts", compute_framework="PyArrowTable"))

    assert returned is host
    assert returned.compute_frameworks == {PyArrowTable}
    assert returned.framework_pinned is True


def test_merging_a_narrowed_host_into_its_twin_flags_the_survivor_pinned() -> None:
    engine = _bare_engine()
    host = _unpinned_host("narrow_merge_host")
    twin = Feature("narrow_merge_host")
    twin.compute_frameworks = {PyArrowTable}
    engine.feature_group_collection[PinHostFG].update({host, twin})
    assert twin.framework_pinned is False

    returned = engine._narrow_host_to_pin(PinHostFG, host, Feature("narrow_merge_ts", compute_framework="PyArrowTable"))

    assert returned is twin
    assert returned.framework_pinned is True
