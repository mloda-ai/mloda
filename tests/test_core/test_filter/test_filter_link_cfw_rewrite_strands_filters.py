"""A link-driven framework rewrite rebinds queue Features, not the SingleFilters GlobalFilter stores."""

from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
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
)
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
        compute_frameworks={PyArrowTable, SecondCfw},
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
