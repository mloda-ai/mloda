"""Groups whose trekked links resolve to different compute frameworks: a link with equal
frameworks and no child on that framework is rejected, the distinct-framework shape here still runs."""

from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

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
    """Both sides of the A-B join would resolve to the same input, so the chooser finds no assignment."""
    links = {
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBSame, MLG_INDEX)),
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootC, MLG_INDEX)),
    }

    with pytest.raises(ValueError) as excinfo:
        mloda.run_all(
            [Feature(MultiLinkChildSame.get_class_name())],
            links=links,
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_ENABLED_SAME,
        )

    message = str(excinfo.value)
    assert "No compute framework assignment satisfies the hard rules" in message
    assert MultiLinkChildSame.get_class_name() in message
    assert "join inner" in message


def test_link_joining_across_distinct_frameworks_delivers_every_parent_column() -> None:
    """The dropped trekker keeps distinguishable sides, so the chained join stays correct."""
    links = {
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBDistinct, MLG_INDEX)),
        Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootC, MLG_INDEX)),
    }

    results = mloda.run_all(
        [Feature(MultiLinkChildDistinct.get_class_name())],
        links=links,
        compute_frameworks=[PandasDataFrame, PyArrowTable, SecondCfw],
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
            compute_frameworks=[PandasDataFrame, PyArrowTable, SecondCfw],
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
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_ENABLED_SAME_FRAMEWORK,
    )

    seen = {value for result in results for value in result[SameFrameworkConsumer.get_class_name()]}
    assert seen == {"sfw_d1|sfw_d2|sfw_p"}


class DescLinkChild(FeatureGroup):
    """PyArrow consumer of both join sides (roots A and BSame)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data.column("mlg_a"), data.column("mlg_b")))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class DescLinkGrandchild(FeatureGroup):
    """Pandas-only consumer of the link child, further down than the join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("DescLinkChild")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        assert not isinstance(data, pa.Table)
        data[cls.get_class_name()] = data["DescLinkChild"] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_ENABLED_DESC = PluginCollector.enabled_feature_groups(
    {MultiLinkRootA, MultiLinkRootBSame, DescLinkChild, DescLinkGrandchild}
)


def _desc_args() -> dict[str, Any]:
    return {
        "links": {Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBSame, MLG_INDEX))},
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "parallelization_modes": {ParallelizationMode.SYNC},
        "plugin_collector": _ENABLED_DESC,
    }


def test_a_consumer_below_a_link_child_hops_once_from_the_link_childs_framework() -> None:
    session = mloda.prepare([Feature(DescLinkGrandchild.get_class_name())], **_desc_args())

    assert session.engine is not None
    steps = list(session.engine.execution_planner)
    hops = [step for step in steps if isinstance(step, TransformFrameworkStep)]
    assert [(hop.from_framework, hop.to_framework) for hop in hops] == [(PyArrowTable, PandasDataFrame)]
    grandchild_step = next(
        step for step in steps if isinstance(step, FeatureGroupStep) and step.feature_group is DescLinkGrandchild
    )
    assert hops[0].uuid in grandchild_step.required_uuids


def test_a_consumer_below_a_link_child_reads_the_joined_values() -> None:
    results = mloda.run_all([Feature(DescLinkGrandchild.get_class_name())], **_desc_args())

    values = [sorted(result[DescLinkGrandchild.get_class_name()]) for result in results]
    assert values == [[22, 44, 66]]


class TwoPathP(FeatureGroup):
    """PyArrow consumer of root A only."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.multiply(data.column("mlg_a"), 100))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TwoPathQ(FeatureGroup):
    """PyArrow link child of roots A and B."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data.column("mlg_a"), data.column("mlg_b")))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TwoPathC(FeatureGroup):
    """Pandas-only consumer reaching A through P and through the A-B link child Q."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("TwoPathP"), Feature("TwoPathQ")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["TwoPathP"] + data["TwoPathQ"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


@pytest.mark.parametrize("extra_request", [[], ["mlg_a"]])
@pytest.mark.parametrize("swap_link_sides", [False, True])
def test_a_consumer_reaching_a_link_root_by_two_paths_is_correct(
    extra_request: list[str], swap_link_sides: bool
) -> None:
    """The outcome must not depend on set order, so the values are always right."""
    left, right = (MultiLinkRootBSame, MultiLinkRootA) if swap_link_sides else (MultiLinkRootA, MultiLinkRootBSame)
    kwargs: dict[str, Any] = {
        "links": {Link.inner(JoinSpec(left, MLG_INDEX), JoinSpec(right, MLG_INDEX))},
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "parallelization_modes": {ParallelizationMode.SYNC},
        "plugin_collector": PluginCollector.enabled_feature_groups(
            {MultiLinkRootA, MultiLinkRootBSame, TwoPathP, TwoPathQ, TwoPathC}
        ),
    }
    features: list[Feature | str] = [Feature(TwoPathC.get_class_name()), *extra_request]

    results = mloda.prepare(features, **kwargs).run()

    values = [sorted(result[TwoPathC.get_class_name()]) for result in results if TwoPathC.get_class_name() in result]
    assert values == [[111, 222, 333]]


_SIDE_PATH_P_NAMES = ("SidePathPandasP", "SidePathArrowP", "SidePathFreeP")


def _side_path_p_column(data: Any) -> str:
    return next(name for name in _SIDE_PATH_P_NAMES if name in data.column_names)


def _side_path_p(data: Any, name: str) -> Any:
    if isinstance(data, pa.Table):
        return data.append_column(name, pc.multiply(data.column("mlg_a"), 100))
    data[name] = data["mlg_a"] * 100
    return data


class SidePathPandasP(FeatureGroup):
    """Pandas-only reader of root A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _side_path_p(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SidePathArrowP(FeatureGroup):
    """PyArrow-only reader of root A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _side_path_p(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SidePathFreeP(FeatureGroup):
    """Reader of root A that runs on either framework."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _side_path_p(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame, PyArrowTable}


class SidePathQ(FeatureGroup):
    """PyArrow link child reading P (named by option), root B, and root A (directly when asked)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        direct = {Feature("mlg_a")} if options.get("sp_direct") else set()
        return {Feature(options.get("sp_p")), Feature("mlg_b"), *direct}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data.column(_side_path_p_column(data)), data.column("mlg_b"))
        return data.append_column(cls.get_class_name(), pc.add(total, data.column("mlg_a")))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SidePathC(FeatureGroup):
    """PyArrow consumer of P and Q."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        q_options = {"sp_p": options.get("sp_p"), "sp_direct": options.get("sp_direct")}
        return {Feature(options.get("sp_p")), Feature("SidePathQ", options=q_options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data.column(_side_path_p_column(data)), data.column("SidePathQ"))
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _side_path_prepare(p_group: type[FeatureGroup], direct: bool) -> Any:
    options = {"sp_p": p_group.get_class_name(), "sp_direct": direct}
    return mloda.prepare(
        [Feature(SidePathC.get_class_name(), options=options)],
        links={Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBSame, MLG_INDEX))},
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {MultiLinkRootA, MultiLinkRootBSame, p_group, SidePathQ, SidePathC}
        ),
    )


def test_a_pinned_framework_mid_between_a_link_side_and_its_consumer_raises_at_plan_time() -> None:
    with pytest.raises(ValueError, match="SidePathPandasP.*must run on"):
        _side_path_prepare(SidePathPandasP, direct=False)


def test_a_pinned_framework_mid_next_to_a_direct_side_read_raises_at_plan_time() -> None:
    with pytest.raises(ValueError, match="SidePathPandasP.*must run on"):
        _side_path_prepare(SidePathPandasP, direct=True)


@pytest.mark.parametrize("p_group", [SidePathArrowP, SidePathFreeP], ids=["pyarrow_mid", "free_mid"])
def test_a_mid_on_the_side_framework_between_a_link_side_and_its_consumer_is_correct(
    p_group: type[FeatureGroup],
) -> None:
    results = _side_path_prepare(p_group, direct=False).run()

    c_name = SidePathC.get_class_name()
    values = [sorted(result[c_name].to_pylist()) for result in results if c_name in result.column_names]
    assert values == [[211, 422, 633]]


TIE_SCALE = {"a": 1, "b": 10}
TIE_INDEX = Index(("tie_jid",))


def _tie_root_data(features: FeatureSet, name: str, base: list[int]) -> dict[str, list[int]]:
    scale = TIE_SCALE[features.get_options_key("tie_tag")]
    return {"tie_jid": [1, 2, 3], name: [value * scale for value in base]}


class TieLeftPd(FeatureGroup):
    """Pandas root whose values depend on the tie_tag option."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tie_left_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(_tie_root_data(features, "tie_left_val", [1, 2, 3]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [TIE_INDEX]


class TieLeftPa(FeatureGroup):
    """PyArrow root whose values depend on the tie_tag option."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tie_left_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(_tie_root_data(features, "tie_left_val", [1, 2, 3]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [TIE_INDEX]


class TieRightPd(FeatureGroup):
    """Pandas root whose values depend on the tie_tag option."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tie_right_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(_tie_root_data(features, "tie_right_val", [100, 200, 300]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [TIE_INDEX]


class TieRightPa(FeatureGroup):
    """PyArrow root whose values depend on the tie_tag option."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tie_right_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(_tie_root_data(features, "tie_right_val", [100, 200, 300]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [TIE_INDEX]


def _tie_parents(options: Options) -> set[Feature]:
    tag = options.get("tie_tag")
    return {Feature("tie_left_val", options={"tie_tag": tag}), Feature("tie_right_val", options={"tie_tag": tag})}


class TieConsumerPd(FeatureGroup):
    """Pandas consumer pulling both roots with its own tie_tag."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return _tie_parents(options)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["tie_left_val"] + data["tie_right_val"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class TieConsumerPa(FeatureGroup):
    """PyArrow consumer pulling both roots with its own tie_tag."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return _tie_parents(options)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["tie_left_val"], data["tie_right_val"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


_TIE_LEFT: dict[str, type[FeatureGroup]] = {"pd": TieLeftPd, "pa": TieLeftPa}
_TIE_RIGHT: dict[str, type[FeatureGroup]] = {"pd": TieRightPd, "pa": TieRightPa}
_TIE_CONSUMER: dict[str, type[FeatureGroup]] = {"pd": TieConsumerPd, "pa": TieConsumerPa}

_TIE_FRAMEWORK_MIXES = [
    pytest.param("pd", "pd", "pd", id="pd_pd_pd"),
    pytest.param("pd", "pa", "pd", id="pd_pa_pd"),
    pytest.param("pa", "pd", "pd", id="pa_pd_pd"),
    pytest.param("pd", "pa", "pa", id="pd_pa_pa"),
    pytest.param("pa", "pa", "pa", id="pa_pa_pa"),
]


def _tie_args(left: str, right: str, consumer: str) -> dict[str, Any]:
    left_group, right_group = _TIE_LEFT[left], _TIE_RIGHT[right]
    return {
        "links": {Link.inner(JoinSpec(left_group, "tie_jid"), JoinSpec(right_group, "tie_jid"))},
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "plugin_collector": PluginCollector.enabled_feature_groups({left_group, right_group, _TIE_CONSUMER[consumer]}),
    }


def _tie_features(consumer: str, tags: list[str]) -> list[Feature | str]:
    name = _TIE_CONSUMER[consumer].get_class_name()
    return [Feature(name, options={"tie_tag": tag}) for tag in tags]


def _tie_values(results: list[Any], consumer: str) -> list[list[int]]:
    name = _TIE_CONSUMER[consumer].get_class_name()
    return sorted(
        sorted(int(v) for v in (r[name].tolist() if hasattr(r, "iloc") else r[name].to_pylist())) for r in results
    )


def _tie_expected(tags: list[str]) -> list[list[int]]:
    return sorted([101 * TIE_SCALE[tag], 202 * TIE_SCALE[tag], 303 * TIE_SCALE[tag]] for tag in tags)


@pytest.mark.parametrize("left, right, consumer", _TIE_FRAMEWORK_MIXES)
def test_one_consumer_requested_with_two_option_variants_over_one_link_joins_each_variant(
    left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}

    results = mloda.run_all(_tie_features(consumer, ["a", "b"]), **kwargs)

    assert _tie_values(results, consumer) == _tie_expected(["a", "b"])


@pytest.mark.parametrize("left, right, consumer", _TIE_FRAMEWORK_MIXES)
def test_one_consumer_requested_with_two_option_variants_over_one_link_plans_one_join_per_variant(
    left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}

    session = mloda.prepare(_tie_features(consumer, ["a", "b"]), **kwargs)

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert len(join_steps) == 2


@pytest.mark.parametrize("left, right, consumer", _TIE_FRAMEWORK_MIXES)
def test_one_consumer_requested_with_one_option_variant_over_one_link_joins_it(
    left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}

    results = mloda.run_all(_tie_features(consumer, ["a"]), **kwargs)

    assert _tie_values(results, consumer) == _tie_expected(["a"])


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(30)
def test_two_option_variants_over_one_link_join_each_variant_with_multiprocessing(flight_server: Any) -> None:
    kwargs = _tie_args("pd", "pd", "pd")
    kwargs["parallelization_modes"] = {ParallelizationMode.MULTIPROCESSING}
    kwargs["flight_server"] = flight_server

    results = mloda.run_all(_tie_features("pd", ["a", "b"]), **kwargs)

    assert _tie_values(results, "pd") == _tie_expected(["a", "b"])
