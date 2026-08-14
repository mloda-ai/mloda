"""Groups whose trekked links resolve to different compute frameworks: a link with equal
frameworks and no child on that framework is rejected, the distinct-framework shape here still runs."""

from typing import Any

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import (
    Feature,
    FeatureName,
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

# Import the transformer so the pandas/pyarrow hop is registered.
import mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer  # noqa: F401

from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw


MLG_INDEX = Index(("mlg_idx",))


def _joined_columns(data: Any, expected: set[str]) -> str:
    """Report which of the expected input columns actually reached the child."""
    return "|".join(sorted(expected.intersection(data.columns)))


class MultiLinkRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mlg_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a": [1, 2, 3], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class MultiLinkRootBSame(FeatureGroup):
    """Root B on the same framework as root A, so the A-B link joins in one framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mlg_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_b": [10, 20, 30], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class MultiLinkRootBDistinct(FeatureGroup):
    """Root B on a second framework, so the A-B link joins across two distinct frameworks."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mlg_bd"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_bd": [10, 20, 30], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SecondCfw}


class MultiLinkRootC(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mlg_c"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_c": [100, 200, 300], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class MultiLinkChildSame(FeatureGroup):
    """Child of both links, restricted to a framework neither A-B side supports."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_b"), Feature("mlg_c")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [_joined_columns(data, {"mlg_a", "mlg_b", "mlg_c"})] * len(data)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class MultiLinkChildDistinct(FeatureGroup):
    """Same shape, but the A-B link joins across two distinct frameworks."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_bd"), Feature("mlg_c")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [_joined_columns(data, {"mlg_a", "mlg_bd", "mlg_c"})] * len(data)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkSharedParent(FeatureGroup):
    """Read as the join source by both spokes below; every feature group here stays on one framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sfw_p"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sfw_p": [1, 2, 3], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkSpokeD1(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sfw_d1"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sfw_d1": [10, 20, 30], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkSpokeD2(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sfw_d2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sfw_d2": [100, 200, 300], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkConsumer(FeatureGroup):
    """Needs the shared parent through both spokes; all four feature groups share one framework."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sfw_p"), Feature("sfw_d1"), Feature("sfw_d2")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [_joined_columns(data, {"sfw_p", "sfw_d1", "sfw_d2"})] * len(data)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_ENABLED_SAME = PluginCollector.enabled_feature_groups(
    {MultiLinkRootA, MultiLinkRootBSame, MultiLinkRootC, MultiLinkChildSame}
)
_ENABLED_DISTINCT = PluginCollector.enabled_feature_groups(
    {MultiLinkRootA, MultiLinkRootBDistinct, MultiLinkRootC, MultiLinkChildDistinct}
)
_ENABLED_SAME_FRAMEWORK = PluginCollector.enabled_feature_groups(
    {SameFrameworkSharedParent, SameFrameworkSpokeD1, SameFrameworkSpokeD2, SameFrameworkConsumer}
)


def test_link_joining_in_a_framework_no_member_supports_raises() -> None:
    """Both sides of the A-B join would resolve to the same input, so planning must stop."""
    links = {
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBSame, MLG_INDEX)),
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootC, MLG_INDEX)),
    }

    with pytest.raises(ValueError) as excinfo:
        mloda.run_all(
            [Feature(MultiLinkChildSame.get_class_name())],
            links=links,
            compute_frameworks={PyArrowTable, PandasDataFrame},
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_ENABLED_SAME,
        )

    message = str(excinfo.value)
    assert f"joins in {PyArrowTable.__name__}" in message
    assert "Both join sides would resolve to the same input" in message


def test_link_joining_across_distinct_frameworks_delivers_every_parent_column() -> None:
    """The dropped trekker keeps distinguishable sides, so the chained join stays correct."""
    links = {
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBDistinct, MLG_INDEX)),
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootC, MLG_INDEX)),
    }

    results = mloda.run_all(
        [Feature(MultiLinkChildDistinct.get_class_name())],
        links=links,
        compute_frameworks={PyArrowTable, PandasDataFrame, SecondCfw},
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_ENABLED_DISTINCT,
    )

    seen = {value for result in results for value in result[MultiLinkChildDistinct.get_class_name()]}
    assert seen == {"mlg_a|mlg_bd|mlg_c"}


def test_link_joining_across_distinct_frameworks_with_the_shared_parent_swapped_must_raise() -> None:
    """Same shape as the test above with the A-B link's sides swapped: no join writes back into
    the shared parent A, so the branches never reunite."""
    links = {
        Link.inner(JoinSpec(MultiLinkRootBDistinct, MLG_INDEX), JoinSpec(MultiLinkRootA, MLG_INDEX)),
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootC, MLG_INDEX)),
    }

    with pytest.raises(ValueError, match="is read as the join source"):
        mloda.run_all(
            [Feature(MultiLinkChildDistinct.get_class_name())],
            links=links,
            compute_frameworks={PyArrowTable, PandasDataFrame, SecondCfw},
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_ENABLED_DISTINCT,
        )


def test_link_joining_a_shared_parent_twice_within_one_framework_must_not_raise() -> None:
    """The shared parent and both spokes all stay on PandasDataFrame: add_tfs reunites the two
    joins through add_value_to_children_if_root bookkeeping, not uuid-slot rewriting, so the
    orphaned-join-source guard must not treat the shared parent as lost."""
    links = {
        Link.inner(JoinSpec(SameFrameworkSpokeD1, MLG_INDEX), JoinSpec(SameFrameworkSharedParent, MLG_INDEX)),
        Link.inner(JoinSpec(SameFrameworkSpokeD2, MLG_INDEX), JoinSpec(SameFrameworkSharedParent, MLG_INDEX)),
    }

    results = mloda.run_all(
        [Feature(SameFrameworkConsumer.get_class_name())],
        links=links,
        compute_frameworks={PandasDataFrame},
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_ENABLED_SAME_FRAMEWORK,
    )

    seen = {value for result in results for value in result[SameFrameworkConsumer.get_class_name()]}
    assert seen == {"sfw_d1|sfw_d2|sfw_p"}
