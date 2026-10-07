"""Groups whose trekked links resolve to different compute frameworks: a link with equal
frameworks and no child on that framework is rejected, the distinct-framework shape here still runs."""

from typing import Any, ClassVar

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

from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw, ThirdCfw


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


class MultiLinkRootAPartial(MultiLinkRootA):
    """Root A with an extra row (w) and without root B's rows (v, u)."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a": [1, 2, 3, 4], "mlg_idx": ["x", "y", "z", "w"]}


class MultiLinkRootBDistinctPartial(MultiLinkRootBDistinct):
    """Root B with extra rows (v, u) and without root A's row (w)."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_bd": [10, 20, 30, 50, 60], "mlg_idx": ["x", "y", "z", "v", "u"]}


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


class _SidePathP(FeatureGroup):
    """Reader of root A on the frameworks in FRAMEWORKS."""

    FRAMEWORKS: ClassVar[set[type[ComputeFramework]]] = set()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _side_path_p(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return cls.FRAMEWORKS


def _side_path_group(name: str, frameworks: set[type[ComputeFramework]]) -> type[FeatureGroup]:
    return type(name, (_SidePathP,), {"FRAMEWORKS": frameworks, "__module__": __name__})


SidePathPandasP = _side_path_group("SidePathPandasP", {PandasDataFrame})
SidePathArrowP = _side_path_group("SidePathArrowP", {PyArrowTable})
SidePathFreeP = _side_path_group("SidePathFreeP", {PandasDataFrame, PyArrowTable})


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


def _side_path_prepare(
    p_group: type[FeatureGroup],
    direct: bool,
    swap_link_sides: bool = False,
    mode: ParallelizationMode = ParallelizationMode.SYNC,
    extra_groups: set[type[FeatureGroup]] | None = None,
    consumer: type[FeatureGroup] = SidePathC,
    root_b: type[FeatureGroup] = MultiLinkRootBSame,
    compute_frameworks: list[type[ComputeFramework]] | None = None,
    extra_options: dict[str, Any] | None = None,
    features: list[Feature | str] | None = None,
    link_factory: Any = Link.inner,
    root_a: type[FeatureGroup] = MultiLinkRootA,
) -> Any:
    options = {"sp_p": p_group.get_class_name(), "sp_direct": direct, **(extra_options or {})}
    left, right = (root_b, root_a) if swap_link_sides else (root_a, root_b)
    return mloda.prepare(
        features or [Feature(consumer.get_class_name(), options=options)],
        links={link_factory(JoinSpec(left, MLG_INDEX), JoinSpec(right, MLG_INDEX))},
        compute_frameworks=compute_frameworks or [PandasDataFrame, PyArrowTable],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {root_a, root_b, p_group, SidePathQ, consumer, *(extra_groups or set())}
        ),
    )


def _side_path_values(results: Any, consumer: type[FeatureGroup] = SidePathC) -> list[list[int]]:
    c_name = consumer.get_class_name()
    return [sorted(result[c_name].to_pylist()) for result in results if c_name in result.column_names]


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize("direct", [False, True], ids=["indirect", "direct"])
@pytest.mark.parametrize(
    "p_group",
    [SidePathArrowP, SidePathFreeP, SidePathPandasP],
    ids=["pyarrow_mid", "free_mid", "pandas_mid"],
)
def test_a_mid_on_the_side_framework_between_a_link_side_and_its_consumer_is_correct(
    p_group: type[FeatureGroup], direct: bool, swap_link_sides: bool
) -> None:
    results = _side_path_prepare(p_group, direct, swap_link_sides).run()

    assert _side_path_values(results) == [[211, 422, 633]]


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize("direct", [False, True], ids=["indirect", "direct"])
@pytest.mark.parametrize("mode", [ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING])
def test_a_pandas_mid_between_a_link_side_and_its_consumer_is_correct_in_parallel_modes(
    flight_server: Any, mode: ParallelizationMode, direct: bool, swap_link_sides: bool
) -> None:
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None
    results = _side_path_prepare(SidePathPandasP, direct, swap_link_sides, mode).run(
        parallelization_modes={mode}, flight_server=server
    )

    assert _side_path_values(results) == [[211, 422, 633]]


class SidePathR(_SidePathP):
    """PyArrow-only second reader of root A."""

    FRAMEWORKS: ClassVar[set[type[ComputeFramework]]] = {PyArrowTable}


class SidePathBelowR(FeatureGroup):
    """PyArrow reader of the Pandas mid, standing between it and the link child."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("SidePathPandasP")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), data.column("SidePathPandasP"))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SidePathQBelow(FeatureGroup):
    """PyArrow link child reading A directly and through Pandas P then PyArrow R."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("SidePathBelowR"), Feature("mlg_a"), Feature("mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), data.column("SidePathBelowR"))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SidePathQTwoFrames(FeatureGroup):
    """PyArrow link child reading A through Pandas P and through PyArrow R."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("SidePathPandasP"), Feature("SidePathR"), Feature("mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), data.column("SidePathR"))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


@pytest.mark.parametrize(
    "consumer, extra_groups, names",
    [
        pytest.param(
            SidePathQBelow,
            {SidePathBelowR},
            ("SidePathPandasP",),
            id="mid_below_a_side_framework_parent",
        ),
        pytest.param(
            SidePathQTwoFrames,
            {SidePathPandasP, SidePathR},
            ("SidePathPandasP",),
            id="two_frames_of_one_side",
        ),
    ],
)
def test_a_side_path_through_two_frameworks_into_one_consumer_raises_at_plan_time(
    consumer: type[FeatureGroup], extra_groups: set[type[FeatureGroup]], names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError) as error:
        _side_path_prepare(SidePathPandasP, False, extra_groups=extra_groups, consumer=consumer)

    message = str(error.value)
    assert all(name in message for name in names)
    assert "missing Links" not in message
    assert "unlinked sources" not in message


def _side_path_b_column(data: Any) -> str:
    return next(name for name in ("mlg_bd", "mlg_b") if name in data.column_names)


class _SidePathBConsumer(FeatureGroup):
    """Consumer of the mid named by READS and root B's column (option sp_b, default mlg_b)."""

    READS: ClassVar[str] = ""
    FRAMEWORKS: ClassVar[set[type[ComputeFramework]]] = {PyArrowTable}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(self.READS), Feature(options.get("sp_b") or "mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data.column(cls.READS), data.column(_side_path_b_column(data)))
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return cls.FRAMEWORKS


class _SidePathDirectConsumer(_SidePathBConsumer):
    """Consumer of root A and root B's column directly."""

    READS: ClassVar[str] = "mlg_a"


def _side_path_consumer(
    name: str, reads: str, frameworks: set[type[ComputeFramework]], base: type[FeatureGroup] = _SidePathBConsumer
) -> type[FeatureGroup]:
    return type(name, (base,), {"READS": reads, "FRAMEWORKS": frameworks, "__module__": __name__})


SidePathTwoConsumerPandas = _side_path_consumer("SidePathTwoConsumerPandas", "SidePathPandasP", {PyArrowTable})
SidePathTwoConsumerArrow = _side_path_consumer("SidePathTwoConsumerArrow", "SidePathR", {PyArrowTable})
SidePathCarrierArrow = _side_path_consumer("SidePathCarrierArrow", "SidePathPandasP", {PyArrowTable})
SidePathCarrierSecond = _side_path_consumer("SidePathCarrierSecond", "SidePathPandasP", {SecondCfw})
SidePathDirectArrow = _side_path_consumer("SidePathDirectArrow", "mlg_a", {PyArrowTable}, base=_SidePathDirectConsumer)
SidePathDirectSecond = _side_path_consumer("SidePathDirectSecond", "mlg_a", {SecondCfw}, base=_SidePathDirectConsumer)


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize(
    "mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]
)
def test_two_consumers_of_one_link_reading_a_side_through_different_mids_join_each_side_frame(
    flight_server: Any, mode: ParallelizationMode, swap_link_sides: bool
) -> None:
    groups: set[type[FeatureGroup]] = {
        MultiLinkRootA,
        MultiLinkRootBSame,
        SidePathPandasP,
        SidePathR,
        SidePathTwoConsumerPandas,
        SidePathTwoConsumerArrow,
    }
    left, right = (MultiLinkRootBSame, MultiLinkRootA) if swap_link_sides else (MultiLinkRootA, MultiLinkRootBSame)
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    session = mloda.prepare(
        ["SidePathTwoConsumerPandas", "SidePathTwoConsumerArrow"],
        links={Link.inner(JoinSpec(left, MLG_INDEX), JoinSpec(right, MLG_INDEX))},
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )

    assert session.engine is not None
    steps = list(session.engine.execution_planner)
    join_steps = [step for step in steps if isinstance(step, JoinStep)]
    assert len(join_steps) == 2
    pandas_mid_uuids = {
        uuid
        for step in steps
        if isinstance(step, FeatureGroupStep) and issubclass(step.feature_group, SidePathPandasP)
        for uuid in step.get_uuids()
    }
    carried = [step for step in join_steps if step.carriers]
    direct = [step for step in join_steps if not step.carriers]
    assert len(carried) == 1
    assert len(direct) == 1
    assert carried[0].carriers & pandas_mid_uuids
    # The direct join must not merge into the frame the carried join shares.
    assert direct[0].shared_destination
    assert direct[0].destination_hop_uuid is not None
    _assert_consumers_wait_only_on_their_joins(session, (SidePathTwoConsumerPandas, SidePathTwoConsumerArrow))

    results = session.run(parallelization_modes={mode}, flight_server=server)

    for consumer in (SidePathTwoConsumerPandas, SidePathTwoConsumerArrow):
        name = consumer.get_class_name()
        assert [sorted(r[name].to_pylist()) for r in results if name in r.column_names] == [[110, 220, 330]]


class DownstreamHopQ(FeatureGroup):
    """PyArrow link child reading both sides of the A-B link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data.column("mlg_a"), data.column("mlg_b")))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class DownstreamHopC(FeatureGroup):
    """PyArrow consumer of the link child Q and a one-sided reader P (named by option)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(options.get("sp_p")), Feature("DownstreamHopQ")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data.column(_side_path_p_column(data)), data.column("DownstreamHopQ"))
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _downstream_hop_prepare(p_group: type[FeatureGroup]) -> Any:
    return mloda.prepare(
        [Feature(DownstreamHopC.get_class_name(), options={"sp_p": p_group.get_class_name()})],
        links={Link.inner(JoinSpec(MultiLinkRootA, MLG_INDEX), JoinSpec(MultiLinkRootBSame, MLG_INDEX))},
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {MultiLinkRootA, MultiLinkRootBSame, p_group, DownstreamHopQ, DownstreamHopC}
        ),
    )


def test_a_hop_of_one_link_side_on_the_joined_framework_reaching_a_join_consumers_child_is_correct() -> None:
    results = _downstream_hop_prepare(SidePathArrowP).run()

    c_name = DownstreamHopC.get_class_name()
    values = [sorted(result[c_name].to_pylist()) for result in results if c_name in result.column_names]
    assert values == [[111, 222, 333]]


TIE_SCALE = {"a": 1, "b": 10}
TIE_INDEX = Index(("tie_jid",))


def _tie_root_data(features: FeatureSet, name: str, base: list[int]) -> dict[str, list[int]]:
    scale = TIE_SCALE[features.get_options_key("tie_tag")]
    return {"tie_jid": list(range(1, len(base) + 1)), name: [value * scale for value in base]}


class _TieRoot(FeatureGroup):
    """Root whose values depend on the tie_tag option, built in FRAMEWORK's native type."""

    COLUMN: ClassVar[str] = ""
    BASE: ClassVar[list[int]] = []
    FRAMEWORK: ClassVar[type[ComputeFramework]] = PyArrowTable

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({cls.COLUMN})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        built = _tie_root_data(features, cls.COLUMN, cls.BASE)
        return pd.DataFrame(built) if cls.FRAMEWORK is PandasDataFrame else pa.table(built)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {cls.FRAMEWORK}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [TIE_INDEX]


def _tie_root(name: str, column: str, base: list[int], framework: type[ComputeFramework]) -> type[FeatureGroup]:
    return type(name, (_TieRoot,), {"COLUMN": column, "BASE": base, "FRAMEWORK": framework, "__module__": __name__})


TieLeftPd = _tie_root("TieLeftPd", "tie_left_val", [1, 2, 3], PandasDataFrame)
TieLeftPa = _tie_root("TieLeftPa", "tie_left_val", [1, 2, 3], PyArrowTable)
TieRightPd = _tie_root("TieRightPd", "tie_right_val", [100, 200, 300], PandasDataFrame)
TieRightPa = _tie_root("TieRightPa", "tie_right_val", [100, 200, 300], PyArrowTable)
TieRightExtraPd = _tie_root("TieRightExtraPd", "tie_right_val", [100, 200, 300, 400], PandasDataFrame)
TieRightExtraPa = _tie_root("TieRightExtraPa", "tie_right_val", [100, 200, 300, 400], PyArrowTable)


def _tie_parents(options: Options) -> set[Feature]:
    tag = options.get("tie_tag")
    return {Feature("tie_left_val", options={"tie_tag": tag}), Feature("tie_right_val", options={"tie_tag": tag})}


class _TieConsumer(FeatureGroup):
    """Consumer pulling both roots with its own tie_tag, computed in the data's own type."""

    FRAMEWORKS: ClassVar[set[type[ComputeFramework]]] = {PyArrowTable}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return _tie_parents(options)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if isinstance(data, pd.DataFrame):
            data[cls.get_class_name()] = data["tie_left_val"] + data["tie_right_val"]
            return data
        return data.append_column(cls.get_class_name(), pc.add(data["tie_left_val"], data["tie_right_val"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return cls.FRAMEWORKS


def _tie_consumer(name: str, frameworks: set[type[ComputeFramework]]) -> type[FeatureGroup]:
    return type(name, (_TieConsumer,), {"FRAMEWORKS": frameworks, "__module__": __name__})


TieConsumerPd = _tie_consumer("TieConsumerPd", {PandasDataFrame})
TieConsumerPa = _tie_consumer("TieConsumerPa", {PyArrowTable})
TieConsumerAny = _tie_consumer("TieConsumerAny", {PandasDataFrame, PyArrowTable})


_TIE_LEFT: dict[str, type[FeatureGroup]] = {"pd": TieLeftPd, "pa": TieLeftPa}
_TIE_RIGHT: dict[str, type[FeatureGroup]] = {
    "pd": TieRightPd,
    "pa": TieRightPa,
    "pd_extra": TieRightExtraPd,
    "pa_extra": TieRightExtraPa,
}
_TIE_CONSUMER: dict[str, type[FeatureGroup]] = {"pd": TieConsumerPd, "pa": TieConsumerPa}

_TIE_FRAMEWORK_MIXES = [
    pytest.param("pd", "pd", "pd", id="pd_pd_pd"),
    pytest.param("pd", "pa", "pd", id="pd_pa_pd"),
    pytest.param("pa", "pd", "pd", id="pa_pd_pd"),
    pytest.param("pd", "pa", "pa", id="pd_pa_pa"),
    pytest.param("pa", "pa", "pa", id="pa_pa_pa"),
    pytest.param("pa", "pd", "pa", id="pa_pd_pa"),
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


def _assert_consumers_wait_only_on_their_joins(session: Any, consumer_types: tuple[type[FeatureGroup], ...]) -> None:
    """Each consumer step waits on exactly the JoinSteps whose resolved record lists it as a consumer."""
    planner = session.engine.execution_planner
    steps = list(planner)
    join_uuids = {step.uuid for step in steps if isinstance(step, JoinStep)}
    consumer_steps = [
        s for s in steps if isinstance(s, FeatureGroupStep) and issubclass(s.feature_group, consumer_types)
    ]
    assert consumer_steps
    for step in consumer_steps:
        own = {r.token for r in planner.resolved_join_plan.records if r.consumers & step.get_uuids()}
        assert own, "the consumer must be recorded by a join"
        assert step.required_uuids & join_uuids == own


@pytest.mark.parametrize("left, right, consumer", _TIE_FRAMEWORK_MIXES)
def test_one_consumer_requested_with_two_option_variants_over_one_link_joins_each_variant(
    left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}

    session = mloda.prepare(_tie_features(consumer, ["a", "b"]), **kwargs)

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert len(join_steps) == 2
    _assert_consumers_wait_only_on_their_joins(session, (_TieConsumer,))
    assert _tie_values(session.run(), consumer) == _tie_expected(["a", "b"])


@pytest.mark.parametrize("left, right, consumer", _TIE_FRAMEWORK_MIXES)
def test_one_consumer_requested_with_one_option_variant_over_one_link_joins_it(
    left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}

    results = mloda.run_all(_tie_features(consumer, ["a"]), **kwargs)

    assert _tie_values(results, consumer) == _tie_expected(["a"])


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "left, right, consumer",
    [pytest.param("pd", "pd", "pd", id="pd_pd_pd"), pytest.param("pa", "pd", "pd", id="pa_pd_pd")],
)
def test_two_option_variants_over_one_link_join_each_variant_with_multiprocessing(
    flight_server: Any, left: str, right: str, consumer: str
) -> None:
    kwargs = _tie_args(left, right, consumer)
    kwargs["parallelization_modes"] = {ParallelizationMode.MULTIPROCESSING}
    kwargs["flight_server"] = flight_server

    results = mloda.run_all(_tie_features(consumer, ["a", "b"]), **kwargs)

    assert _tie_values(results, consumer) == _tie_expected(["a", "b"])


class TieOneSidedConsumer(FeatureGroup):
    """PyArrow consumer reading a fixed variant on one join side and a variant chosen by tie_tag on the other."""

    VARY: ClassVar[str] = "right"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        tag = options.get("tie_tag")
        fixed = {"tie_tag": "a"}
        shared, varying = (
            ("tie_left_val", "tie_right_val") if self.VARY == "right" else ("tie_right_val", "tie_left_val")
        )
        return {
            Feature(shared, options=fixed, forward_group_exclude=frozenset({"tie_tag"})),
            Feature(varying, options={"tie_tag": tag}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["tie_left_val"], data["tie_right_val"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TieOneSidedLeftConsumer(TieOneSidedConsumer):
    """Same consumer with the left side varying and the right side shared."""

    VARY: ClassVar[str] = "left"


class TieOneSidedChild(FeatureGroup):
    """PyArrow child of a one-sided consumer named by the tie_consumer option."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(options.get("tie_consumer"), options={"tie_tag": options.get("tie_tag")})}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        consumer = features.get_options_key("tie_consumer")
        return data.append_column(cls.get_class_name(), pc.multiply(data[consumer], 2))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TieBothVaryConsumer(FeatureGroup):
    """PyArrow consumer reading its left variant from tie_l and its right variant from tie_r."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("tie_left_val", options={"tie_tag": options.get("tie_l")}),
            Feature("tie_right_val", options={"tie_tag": options.get("tie_r")}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["tie_left_val"], data["tie_right_val"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


_TIE_ONE_SIDED_SHAPES = [
    pytest.param(TieOneSidedConsumer, "pa", "pa", id="same_framework_right_varies"),
    pytest.param(TieOneSidedLeftConsumer, "pa", "pa", id="same_framework_left_varies"),
    pytest.param(TieOneSidedConsumer, "pd", "pa", id="varying_side_on_consumer_framework"),
    pytest.param(TieOneSidedConsumer, "pa", "pd", id="shared_side_on_consumer_framework"),
]


def _tie_one_sided_expected(consumer: type[FeatureGroup], tag: str) -> list[int]:
    left_scale, right_scale = (1, TIE_SCALE[tag]) if consumer is TieOneSidedConsumer else (TIE_SCALE[tag], 1)
    return [i * left_scale + 100 * i * right_scale for i in (1, 2, 3)]


def _tie_one_sided_kwargs(consumer: type[FeatureGroup], left: str, right: str, join_type: str) -> dict[str, Any]:
    left_group, right_group = _TIE_LEFT[left], _TIE_RIGHT[right]
    make = {"inner": Link.inner, "left": Link.left, "right": Link.right, "outer": Link.outer}[join_type]
    return {
        "links": {make(JoinSpec(left_group, "tie_jid"), JoinSpec(right_group, "tie_jid"))},
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "plugin_collector": PluginCollector.enabled_feature_groups(
            {left_group, right_group, consumer, TieOneSidedChild, TieBothVaryConsumer}
        ),
    }


def _tie_one_sided_features(consumer: type[FeatureGroup], tags: list[str]) -> list[Feature | str]:
    return [Feature(consumer.get_class_name(), options={"tie_tag": tag}) for tag in tags]


def _column_values(results: list[Any], name: str) -> list[list[int]]:
    found = [r[name] for r in results if name in (r.columns if hasattr(r, "iloc") else r.column_names)]
    return sorted(sorted(int(v) for v in (c.tolist() if hasattr(c, "iloc") else c.to_pylist())) for c in found)


def _tie_one_sided_run(
    tags: list[str],
    consumer: type[FeatureGroup] = TieOneSidedConsumer,
    left: str = "pa",
    right: str = "pa",
    join_type: str = "inner",
) -> list[list[int]]:
    kwargs = _tie_one_sided_kwargs(consumer, left, right, join_type)
    results = mloda.run_all(
        _tie_one_sided_features(consumer, tags), parallelization_modes={ParallelizationMode.SYNC}, **kwargs
    )
    return _column_values(results, consumer.get_class_name())


@pytest.mark.parametrize(("tag", "expected"), [("a", [101, 202, 303]), ("b", [1001, 2002, 3003])])
def test_one_sided_option_variant_requested_alone_over_one_link_joins_it(tag: str, expected: list[int]) -> None:
    assert _tie_one_sided_run([tag]) == [expected]


def _nullable_column_values(results: list[Any], name: str) -> list[list[int | None]]:
    found = [r[name] for r in results if name in (r.columns if hasattr(r, "iloc") else r.column_names)]
    columns = [
        [None if pd.isna(v) else int(v) for v in (c.tolist() if hasattr(c, "iloc") else c.to_pylist())] for c in found
    ]
    return sorted((sorted(c, key=lambda v: (v is None, v or 0)) for c in columns), key=lambda c: [v or 0 for v in c])


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
@pytest.mark.parametrize("right", ["pa_extra", "pd_extra"])
def test_a_right_join_keeps_the_right_row_the_left_lacks_for_each_variant(
    flight_server: Any, mode: ParallelizationMode, right: str
) -> None:
    kwargs = _tie_one_sided_kwargs(TieOneSidedConsumer, "pa", right, "right")
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    results = mloda.run_all(
        _tie_one_sided_features(TieOneSidedConsumer, ["a", "b"]),
        parallelization_modes={mode},
        flight_server=server,
        **kwargs,
    )

    values = _nullable_column_values(results, TieOneSidedConsumer.get_class_name())
    assert values == [[101, 202, 303, None], [1001, 2002, 3003, None]]


def test_two_consumers_of_one_right_link_on_the_left_and_the_right_framework_raise_a_clear_error() -> None:
    link = Link.right(JoinSpec(TieLeftPa, "tie_jid"), JoinSpec(TieRightPd, "tie_jid"))

    with pytest.raises(ValueError, match="Request them separately or restrict their compute frameworks"):
        mloda.run_all(
            [*_tie_features("pa", ["a"]), *_tie_features("pd", ["a"])],
            links={link},
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {TieLeftPa, TieRightPd, TieConsumerPa, TieConsumerPd}
            ),
            parallelization_modes={ParallelizationMode.SYNC},
        )


def _right_link_sibling_run(mode: ParallelizationMode, flight_server: Any = None) -> dict[str, Any]:
    return {
        "features": [*_tie_features("pa", ["a"]), Feature("TieConsumerAny", options={"tie_tag": "a"})],
        "links": {Link.right(JoinSpec(TieLeftPa, "tie_jid"), JoinSpec(TieRightExtraPd, "tie_jid"))},
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "parallelization_modes": {mode},
        "plugin_collector": PluginCollector.enabled_feature_groups(
            {TieLeftPa, TieRightExtraPd, TieConsumerPa, TieConsumerAny}
        ),
    }


def test_a_flexible_and_a_left_only_consumer_of_one_right_link_both_plan_on_the_left_framework() -> None:
    kwargs = _right_link_sibling_run(ParallelizationMode.SYNC)
    session = mloda.prepare(kwargs.pop("features"), **kwargs)

    assert session.engine is not None
    steps = [s for s in session.engine.execution_planner if isinstance(s, FeatureGroupStep)]
    consumers = [s for s in steps if issubclass(s.feature_group, (TieConsumerPa, TieConsumerAny))]
    assert len(consumers) == 2
    assert {s.compute_framework for s in consumers} == {PyArrowTable}


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
def test_a_flexible_and_a_left_only_consumer_of_one_right_link_return_the_right_join_values(
    flight_server: Any, mode: ParallelizationMode
) -> None:
    kwargs = _right_link_sibling_run(mode)
    features = kwargs.pop("features")
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    results = mloda.run_all(features, flight_server=server, **kwargs)

    assert _nullable_column_values(results, "TieConsumerPa") == [[101, 202, 303, None]]
    assert _nullable_column_values(results, "TieConsumerAny") == [[101, 202, 303, None]]


def _tie_child_features(consumer: type[FeatureGroup], tags: list[str]) -> list[Feature | str]:
    options = {"tie_consumer": consumer.get_class_name()}
    return [Feature("TieOneSidedChild", options={**options, "tie_tag": tag}) for tag in tags]


@pytest.mark.parametrize("requested", ["consumer", "child", "both"])
@pytest.mark.parametrize("join_type", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("consumer, left, right", _TIE_ONE_SIDED_SHAPES)
def test_option_variants_reading_one_join_side_alike_and_the_other_differently_join_each_variant(
    consumer: type[FeatureGroup], left: str, right: str, join_type: str, requested: str
) -> None:
    kwargs = _tie_one_sided_kwargs(consumer, left, right, join_type)
    kwargs["parallelization_modes"] = {ParallelizationMode.SYNC}
    features: list[Feature | str] = []
    if requested != "child":
        features += _tie_one_sided_features(consumer, ["a", "b"])
    if requested != "consumer":
        features += _tie_child_features(consumer, ["a", "b"])

    session = mloda.prepare(features, **kwargs)

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    if requested == "consumer":
        assert len(join_steps) == 2
        assert {step.destination_framework for step in join_steps} == {PyArrowTable}
        _assert_consumers_wait_only_on_their_joins(session, (TieOneSidedConsumer,))
    results = session.run()
    expected = sorted(_tie_one_sided_expected(consumer, tag) for tag in ("a", "b"))
    if requested != "child":
        assert _column_values(results, consumer.get_class_name()) == expected
    if requested != "consumer":
        assert _column_values(results, "TieOneSidedChild") == sorted([v * 2 for v in e] for e in expected)


def test_option_variants_reading_differing_variants_of_both_join_sides_join_each_variant() -> None:
    kwargs = _tie_one_sided_kwargs(TieOneSidedConsumer, "pa", "pa", "inner")
    pairs = (("a", "a"), ("a", "b"), ("b", "a"))
    features: list[Feature | str] = [
        Feature("TieBothVaryConsumer", options={"tie_l": left, "tie_r": right}) for left, right in pairs
    ]

    results = mloda.run_all(features, parallelization_modes={ParallelizationMode.SYNC}, **kwargs)

    expected = sorted([i * TIE_SCALE[left] + 100 * i * TIE_SCALE[right] for i in (1, 2, 3)] for left, right in pairs)
    assert _column_values(results, "TieBothVaryConsumer") == expected


def test_one_sided_variants_requested_next_to_their_shared_side_feature_return_both() -> None:
    consumer = TieOneSidedConsumer
    kwargs = _tie_one_sided_kwargs(consumer, "pa", "pa", "inner")
    shared = Feature("tie_left_val", options={"tie_tag": "a"})

    results = mloda.run_all(
        [*_tie_one_sided_features(consumer, ["a", "b"]), shared],
        parallelization_modes={ParallelizationMode.SYNC},
        **kwargs,
    )

    assert _column_values(results, consumer.get_class_name()) == [[101, 202, 303], [1001, 2002, 3003]]
    assert [1, 2, 3] in _column_values(results, "tie_left_val")


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", [ParallelizationMode.MULTIPROCESSING, ParallelizationMode.THREADING])
@pytest.mark.parametrize(
    "consumer, left, right",
    [_TIE_ONE_SIDED_SHAPES[0], _TIE_ONE_SIDED_SHAPES[1], _TIE_ONE_SIDED_SHAPES[3]],
)
@pytest.mark.parametrize("join_type", ["inner", "right"])
def test_one_sided_option_variants_over_one_link_join_each_variant_in_parallel_modes(
    flight_server: Any, mode: ParallelizationMode, consumer: type[FeatureGroup], left: str, right: str, join_type: str
) -> None:
    kwargs = _tie_one_sided_kwargs(consumer, left, right, join_type)
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    results = mloda.run_all(
        _tie_one_sided_features(consumer, ["a", "b"]), parallelization_modes={mode}, flight_server=server, **kwargs
    )

    assert _column_values(results, consumer.get_class_name()) == sorted(
        _tie_one_sided_expected(consumer, tag) for tag in ("a", "b")
    )


_CARRIER_FRAMEWORKS = [PandasDataFrame, PyArrowTable, SecondCfw]
_CARRIER_CONSUMERS = {
    "carried_framework": (SidePathCarrierArrow, SidePathDirectArrow),
    "other_side_framework": (SidePathCarrierSecond, SidePathDirectSecond),
}
_CARRIER_KIND_PARAMS = pytest.mark.parametrize("consumer_kind", list(_CARRIER_CONSUMERS))


def _carrier_alone(
    carrier: type[FeatureGroup],
    swap_link_sides: bool,
    mode: ParallelizationMode = ParallelizationMode.SYNC,
    link_factory: Any = Link.inner,
    root_a: type[FeatureGroup] = MultiLinkRootA,
    root_b: type[FeatureGroup] = MultiLinkRootBDistinct,
) -> Any:
    return _side_path_prepare(
        SidePathPandasP,
        False,
        swap_link_sides,
        mode,
        consumer=carrier,
        root_a=root_a,
        root_b=root_b,
        compute_frameworks=_CARRIER_FRAMEWORKS,
        extra_options={"sp_b": "mlg_bd"},
        link_factory=link_factory,
    )


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "mode, swap_link_sides, consumer_kind",
    [
        pytest.param(ParallelizationMode.SYNC, False, "carried_framework", id="sync_a_b_carried_framework"),
        pytest.param(ParallelizationMode.SYNC, True, "carried_framework", id="sync_b_a_carried_framework"),
        pytest.param(ParallelizationMode.SYNC, False, "other_side_framework", id="sync_a_b_other_side_framework"),
        pytest.param(ParallelizationMode.SYNC, True, "other_side_framework", id="sync_b_a_other_side_framework"),
        pytest.param(ParallelizationMode.THREADING, False, "carried_framework", id="threading_a_b_carried_framework"),
        pytest.param(ParallelizationMode.THREADING, True, "carried_framework", id="threading_b_a_carried_framework"),
        pytest.param(
            ParallelizationMode.THREADING, False, "other_side_framework", id="threading_a_b_other_side_framework"
        ),
        pytest.param(
            ParallelizationMode.MULTIPROCESSING, False, "carried_framework", id="multiprocessing_a_b_carried_framework"
        ),
    ],
)
def test_a_carrier_consumer_over_a_link_across_distinct_frameworks_is_correct(
    flight_server: Any, mode: ParallelizationMode, swap_link_sides: bool, consumer_kind: str
) -> None:
    carrier = _CARRIER_CONSUMERS[consumer_kind][0]
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    results = _carrier_alone(carrier, swap_link_sides, mode).run(parallelization_modes={mode}, flight_server=server)

    assert _side_path_values(results, carrier) == [[110, 220, 330]]


def _carrier_with_direct(
    carrier: type[FeatureGroup], direct: type[FeatureGroup], swap_link_sides: bool, mode: ParallelizationMode
) -> Any:
    options = {"sp_b": "mlg_bd"}
    return _side_path_prepare(
        SidePathPandasP,
        False,
        swap_link_sides,
        mode,
        extra_groups={direct},
        consumer=carrier,
        root_b=MultiLinkRootBDistinct,
        compute_frameworks=_CARRIER_FRAMEWORKS,
        features=[
            Feature(carrier.get_class_name(), options=options),
            Feature(direct.get_class_name(), options=options),
        ],
    )


# Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "mode, swap_link_sides, consumer_kind",
    [
        pytest.param(mode, swap, kind, id=f"{name}-{swap_id}")
        for name, mode, kind in (
            ("sync_carried_framework", ParallelizationMode.SYNC, "carried_framework"),
            ("sync_other_side_framework", ParallelizationMode.SYNC, "other_side_framework"),
            ("threading", ParallelizationMode.THREADING, "carried_framework"),
        )
        for swap, swap_id in ((False, "a_b"), (True, "b_a"))
    ]
    + [pytest.param(ParallelizationMode.MULTIPROCESSING, False, "carried_framework", id="multiprocessing-a_b")],
)
def test_a_carrier_consumer_next_to_a_direct_consumer_of_one_link_waits_only_on_its_join(
    flight_server: Any, mode: ParallelizationMode, swap_link_sides: bool, consumer_kind: str
) -> None:
    carrier, direct = _CARRIER_CONSUMERS[consumer_kind]
    server = flight_server if mode == ParallelizationMode.MULTIPROCESSING else None

    session = _carrier_with_direct(carrier, direct, swap_link_sides, mode)

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert len(join_steps) == 2
    assert len([step for step in join_steps if step.carriers]) == 1
    _assert_consumers_wait_only_on_their_joins(session, (carrier, direct))

    results = session.run(parallelization_modes={mode}, flight_server=server)

    assert _side_path_values(results, carrier) == [[110, 220, 330]]
    assert _side_path_values(results, direct) == [[11, 22, 33]]


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@_CARRIER_KIND_PARAMS
def test_the_carrier_join_is_planned_on_the_consumers_framework(consumer_kind: str, swap_link_sides: bool) -> None:
    carrier = _CARRIER_CONSUMERS[consumer_kind][0]
    session = _carrier_alone(carrier, swap_link_sides)

    assert session.engine is not None
    planner = session.engine.execution_planner
    carried = [step for step in planner if isinstance(step, JoinStep) and step.carriers]
    assert len(carried) == 1
    frameworks = carrier.compute_framework_rule()
    assert frameworks is not None
    consumer_framework = next(iter(frameworks))
    assert carried[0].destination_framework is consumer_framework
    records = [record for record in planner.resolved_join_plan.records if record.token == carried[0].uuid]
    assert [record.destination_framework for record in records] == [consumer_framework]
    sides = [(record.left, record.right) for record in records]
    for left, right in sides:
        for side in (left, right):
            expected = PyArrowTable if side.feature_group is MultiLinkRootA else SecondCfw
            assert side.declared_frameworks == {expected}
    assert all(record.destination_uuids.isdisjoint(record.source_uuids) for record in records)
    assert all(record.destination_uuids and record.source_uuids for record in records)


_A_KEPT = [110, 220, 330, None]
_B_KEPT = [110, 220, 330, None, None]
_FULL_MATCH = [110, 220, 330]


@pytest.mark.parametrize(
    "root_a, root_b, partial",
    [
        pytest.param(MultiLinkRootA, MultiLinkRootBDistinct, False, id="full_match"),
        pytest.param(MultiLinkRootAPartial, MultiLinkRootBDistinctPartial, True, id="partial_match"),
    ],
)
@pytest.mark.parametrize("link_factory", [Link.left, Link.right], ids=["left", "right"])
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@_CARRIER_KIND_PARAMS
def test_a_carrier_consumer_over_a_left_or_right_link_keeps_the_link_rows(
    consumer_kind: str, swap_link_sides: bool, link_factory: Any, root_a: Any, root_b: Any, partial: bool
) -> None:
    carrier = _CARRIER_CONSUMERS[consumer_kind][0]
    a_side_kept = (link_factory == Link.left) != swap_link_sides
    expected = ([_A_KEPT] if a_side_kept else [_B_KEPT]) if partial else [_FULL_MATCH]

    results = _carrier_alone(carrier, swap_link_sides, link_factory=link_factory, root_a=root_a, root_b=root_b).run()

    assert _nullable_column_values(results, carrier.get_class_name()) == expected


class RemedyRootA1(MultiLinkRootA):
    """Polymorphic root A variant on a third framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"mlg_a1"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a1": [7, 8, 9], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


class RemedyThirdConsumer(FeatureGroup):
    """Consumer on the third framework reading both root A variants and root B."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("SidePathArrowP"), Feature("mlg_a1"), Feature("mlg_bd")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_consumer_on_a_framework_of_a_polymorphic_link_side_but_neither_nearest_side_is_rejected(
    swap_link_sides: bool,
) -> None:
    left, right = (
        (MultiLinkRootBDistinct, MultiLinkRootA) if swap_link_sides else (MultiLinkRootA, MultiLinkRootBDistinct)
    )

    with pytest.raises(ValueError, match="read a link side through"):
        mloda.prepare(
            [RemedyThirdConsumer.get_class_name()],
            links={Link.inner(JoinSpec(left, MLG_INDEX), JoinSpec(right, MLG_INDEX))},
            compute_frameworks=[PandasDataFrame, PyArrowTable, SecondCfw, ThirdCfw],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {MultiLinkRootA, RemedyRootA1, MultiLinkRootBDistinct, SidePathArrowP, RemedyThirdConsumer}
            ),
        )


class SidePathPandasBP(_SidePathP):
    """Pandas mid reading root B (mlg_bd)."""

    FRAMEWORKS: ClassVar[set[type[ComputeFramework]]] = {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_bd")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["mlg_bd"] * 100
        return data


class SidePathCarrierFromB(FeatureGroup):
    """PyArrow consumer of the B-side Pandas mid and root A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("SidePathPandasBP"), Feature("mlg_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data.column("SidePathPandasBP"), data.column("mlg_a"))
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_carrier_descending_from_the_b_side_is_correct(swap_link_sides: bool) -> None:
    results = _side_path_prepare(
        SidePathPandasBP,
        False,
        swap_link_sides,
        consumer=SidePathCarrierFromB,
        root_b=MultiLinkRootBDistinct,
        compute_frameworks=_CARRIER_FRAMEWORKS,
    ).run()

    assert _side_path_values(results, SidePathCarrierFromB) == [[1001, 2002, 3003]]
