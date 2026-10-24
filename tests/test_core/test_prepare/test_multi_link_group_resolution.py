"""Groups whose trekked links resolve to different compute frameworks: a link with equal
frameworks and no child on that framework is rejected, the distinct-framework shape here still runs."""

from collections.abc import Callable
from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, GeneratedFeatureGroup
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


class MultiLinkRootA(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a": [1, 2, 3], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class MultiLinkRootBSame(GeneratedFeatureGroup):
    """Root B on the same framework as root A, so the A-B link joins in one framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_b"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_b": [10, 20, 30], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class MultiLinkRootBDistinct(GeneratedFeatureGroup):
    """Root B on a second framework, so the A-B link joins across two distinct frameworks."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_bd"}

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


class MultiLinkRootC(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_c"}

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


class SameFrameworkSharedParent(GeneratedFeatureGroup):
    """Read as the join source by both spokes below; every feature group here stays on one framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sfw_p"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sfw_p": [1, 2, 3], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkSpokeD1(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sfw_d1"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sfw_d1": [10, 20, 30], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SameFrameworkSpokeD2(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sfw_d2"}

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
    joins through add_join_token bookkeeping, not uuid-slot rewriting, so the
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


class _TieRoot(GeneratedFeatureGroup):
    """Root whose values depend on the tie_tag option, built in FRAMEWORK's native type."""

    COLUMN: ClassVar[str] = ""
    BASE: ClassVar[list[int]] = []
    FRAMEWORK: ClassVar[type[ComputeFramework]] = PyArrowTable

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.COLUMN}

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
    hops = [
        step
        for step in planner
        if isinstance(step, TransformFrameworkStep) and step.uuid == carried[0].destination_hop_uuid
    ]
    assert len(hops) == 1
    assert hops[0].to_framework is records[0].destination_framework
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
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_a1"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a1": [7, 8, 9], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


class _RemedyConsumer(FeatureGroup):
    """Consumer reading both root A variants and root B on a ClassVar framework."""

    FRAMEWORK: ClassVar[type[ComputeFramework]] = ThirdCfw
    INPUTS: ClassVar[tuple[str, ...]] = ("SidePathArrowP", "mlg_a1", "mlg_bd")

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name) for name in self.INPUTS}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {cls.FRAMEWORK}


def _remedy_consumer(name: str, framework: type[ComputeFramework], direct: bool) -> type[FeatureGroup]:
    inputs = ("mlg_a" if direct else "SidePathArrowP", "mlg_a1", "mlg_bd")
    return type(name, (_RemedyConsumer,), {"FRAMEWORK": framework, "INPUTS": inputs, "__module__": __name__})


RemedyThirdConsumer = _remedy_consumer("RemedyThirdConsumer", ThirdCfw, False)
RemedyThirdDirectConsumer = _remedy_consumer("RemedyThirdDirectConsumer", ThirdCfw, True)
RemedyArrowConsumer = _remedy_consumer("RemedyArrowConsumer", PyArrowTable, False)
RemedyArrowDirectConsumer = _remedy_consumer("RemedyArrowDirectConsumer", PyArrowTable, True)
RemedySecondMidConsumer = _remedy_consumer("RemedySecondMidConsumer", SecondCfw, False)


class RemedySumConsumer(_RemedyConsumer):
    """Third-framework consumer summing mlg_a1 and mlg_bd."""

    INPUTS: ClassVar[tuple[str, ...]] = ("mlg_a1", "mlg_bd")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): pc.add(data["mlg_a1"], data["mlg_bd"]).to_pylist()}


_REMEDY_KWARGS: dict[str, Any] = {
    "root_b": MultiLinkRootBDistinct,
    "extra_groups": {RemedyRootA1},
    "compute_frameworks": [PandasDataFrame, PyArrowTable, SecondCfw, ThirdCfw],
}


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize(
    "consumer",
    [
        pytest.param(RemedyThirdConsumer, id="third_mid"),
        pytest.param(RemedyThirdDirectConsumer, id="third_direct"),
        pytest.param(RemedyArrowConsumer, id="arrow_mid"),
        pytest.param(RemedyArrowDirectConsumer, id="arrow_direct"),
        pytest.param(RemedySecondMidConsumer, id="b_framework_mid"),
    ],
)
def test_a_consumer_joining_two_members_of_one_link_side_on_different_frameworks_is_rejected(
    consumer: type[FeatureGroup], swap_link_sides: bool
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=consumer,
            features=[consumer.get_class_name()],
            **_REMEDY_KWARGS,
        )

    message = str(exc_info.value)
    assert "members of one link side" in message
    assert "MultiLinkRootA" in message
    assert "RemedyRootA1" in message
    assert "one member of the link side per consumer" in message
    assert "only when both joins merge into the other link side's frame" in message
    assert "reads both members directly" in message
    assert "runs on neither" not in message


_BACKSTOP_INNER = "one consumer cannot join members of one link side on different compute frameworks"


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_the_planner_backstop_messages_for_members_of_one_link_side_on_different_frameworks(
    monkeypatch: pytest.MonkeyPatch, swap_link_sides: bool
) -> None:
    # Empty chooser rules reach the planner's backstop.
    monkeypatch.setattr(ChooseComputeFrameworks, "_side_member_rules", lambda self, *args, **kwargs: [])

    with pytest.raises(ValueError) as inner_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=RemedyThirdConsumer,
            features=[RemedyThirdConsumer.get_class_name()],
            **_REMEDY_KWARGS,
        )
    with pytest.raises(ValueError) as outer_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=RemedyArrowConsumer,
            features=[RemedyArrowConsumer.get_class_name()],
            **_REMEDY_KWARGS,
        )

    inner, outer = str(inner_info.value), str(outer_info.value)
    assert "RemedyRootA1" in inner
    assert _BACKSTOP_INNER in inner.lower()
    assert "third compute framework" not in inner
    assert ThirdCfw.get_class_name() in outer
    assert SecondCfw.get_class_name() in outer
    assert "SidePathArrowP" in outer
    assert "RemedyRootA1" in outer
    assert "third compute framework" not in outer
    assert "Read one member of the link side per consumer" in outer
    assert "Compute the consumer on one of the join's frameworks" not in outer
    assert inner != outer


class RemedyRootBThird(MultiLinkRootBSame):
    """Root B variant on the third framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


def _remedy_sum_consumer(name: str, framework: type[ComputeFramework], inputs: tuple[str, ...]) -> type[FeatureGroup]:
    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        total = data[inputs[0]]
        for column in inputs[1:]:
            total = pc.add(total, data[column])
        # Pass the frame through so sibling consumers share it.
        return data.append_column(cls.get_class_name(), total)

    namespace = {
        "FRAMEWORK": framework,
        "INPUTS": inputs,
        "__module__": __name__,
        "calculate_feature": classmethod(calculate_feature),
    }
    return type(name, (_RemedyConsumer,), namespace)


_REMEDY_DIRECT_INPUTS = ("mlg_a", "mlg_a1", "mlg_b")
RemedyArrowSameBConsumer = _remedy_sum_consumer("RemedyArrowSameBConsumer", PyArrowTable, _REMEDY_DIRECT_INPUTS)
RemedyArrowSameBMidConsumer = _remedy_sum_consumer(
    "RemedyArrowSameBMidConsumer", PyArrowTable, ("SidePathArrowP", "mlg_a1", "mlg_b")
)
RemedyThirdBThirdConsumer = _remedy_sum_consumer("RemedyThirdBThirdConsumer", ThirdCfw, _REMEDY_DIRECT_INPUTS)
RemedyThirdBThirdMidConsumer = _remedy_sum_consumer(
    "RemedyThirdBThirdMidConsumer", ThirdCfw, ("SidePathArrowP", "mlg_a1", "mlg_b")
)
RemedyThirdPairConsumer = _remedy_sum_consumer("RemedyThirdPairConsumer", ThirdCfw, ("mlg_a1", "mlg_b"))
_REMEDY_SECOND_INPUTS = ("mlg_a", "mlg_a1", "mlg_bd")
RemedySecondBConsumer = _remedy_sum_consumer("RemedySecondBConsumer", SecondCfw, _REMEDY_SECOND_INPUTS)


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize(
    "consumer, root_b, expected",
    [
        pytest.param(RemedySumConsumer, MultiLinkRootBDistinct, [[17, 28, 39]], id="third_distinct_b"),
        pytest.param(RemedyArrowSameBConsumer, MultiLinkRootBSame, [[18, 30, 42]], id="arrow_direct_b_on_arrow"),
        pytest.param(RemedyArrowSameBMidConsumer, MultiLinkRootBSame, [[117, 228, 339]], id="arrow_mid_b_on_arrow"),
        pytest.param(RemedyThirdBThirdConsumer, RemedyRootBThird, [[18, 30, 42]], id="third_direct_b_on_third"),
        pytest.param(RemedyThirdBThirdMidConsumer, RemedyRootBThird, [[117, 228, 339]], id="third_mid_b_on_third"),
    ],
)
def test_a_consumer_reading_members_of_one_link_side_is_correct_when_the_other_side_shares_a_framework(
    consumer: type[FeatureGroup],
    root_b: type[FeatureGroup],
    expected: list[list[int]],
    swap_link_sides: bool,
    mode: ParallelizationMode,
) -> None:
    session = _side_path_prepare(
        SidePathArrowP,
        False,
        swap_link_sides,
        mode,
        consumer=consumer,
        features=[consumer.get_class_name()],
        **{**_REMEDY_KWARGS, "root_b": root_b},
    )

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert not any(step.carriers for step in join_steps)

    results = session.run(parallelization_modes={mode})

    assert _side_path_values(results, consumer) == expected


@pytest.mark.parametrize("link_factory", [Link.inner, Link.outer], ids=["inner", "outer"])
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param(RemedyThirdBThirdMidConsumer, RemedyThirdPairConsumer, id="mid_and_pair"),
        pytest.param(RemedyThirdBThirdConsumer, RemedyThirdPairConsumer, id="direct_and_pair"),
    ],
)
def test_a_pair_consumer_missing_a_join_of_the_polymorphic_link_layout_is_rejected(
    first: type[FeatureGroup], second: type[FeatureGroup], swap_link_sides: bool, link_factory: Any
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            ParallelizationMode.SYNC,
            consumer=first,
            features=[first.get_class_name(), second.get_class_name()],
            link_factory=link_factory,
            **{**_REMEDY_KWARGS, "root_b": RemedyRootBThird, "extra_groups": {*_REMEDY_KWARGS["extra_groups"], second}},
        )

    message = str(exc_info.value)
    assert first.get_class_name() in message
    assert second.get_class_name() in message
    assert "Resolution:" in message


def _a_without_x(cls: Any, data: Any, features: FeatureSet) -> Any:
    return {"mlg_a": [2, 3], "mlg_idx": ["y", "z"]}


# Mid is mlg_a * 100 + mlg_a1 + mlg_b, direct is mlg_a + mlg_a1 + mlg_b. Without x in A, the inner joins
# drop row x for both consumers; the outer joins keep x (A1 and B have it) with the missing mlg_a as null.
_POLYMORPHIC_EXPECTED = {
    (Link.inner, False): ([117, 228, 339], [18, 30, 42]),
    (Link.outer, False): ([117, 228, 339], [18, 30, 42]),
    (Link.inner, True): ([228, 339], [30, 42]),
    (Link.outer, True): ([228, 339, None], [30, 42, None]),
}


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize("link_factory", [Link.inner, Link.outer], ids=["inner", "outer"])
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize("a_lacks_x", [False, True], ids=["full_data", "a_lacks_row_x"])
def test_a_mid_and_a_direct_consumer_of_one_polymorphic_link_layout_return_their_values_together(
    monkeypatch: pytest.MonkeyPatch,
    a_lacks_x: bool,
    swap_link_sides: bool,
    link_factory: Any,
    mode: ParallelizationMode,
) -> None:
    if a_lacks_x:
        monkeypatch.setattr(MultiLinkRootA, "calculate_feature", classmethod(_a_without_x))
    first, second = RemedyThirdBThirdMidConsumer, RemedyThirdBThirdConsumer
    session = _side_path_prepare(
        SidePathArrowP,
        False,
        swap_link_sides,
        mode,
        consumer=first,
        features=[first.get_class_name(), second.get_class_name()],
        link_factory=link_factory,
        **{**_REMEDY_KWARGS, "root_b": RemedyRootBThird, "extra_groups": {*_REMEDY_KWARGS["extra_groups"], second}},
    )

    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert all(step.split_consumers is None for step in join_steps)

    results = session.run(parallelization_modes={mode})

    first_expected, second_expected = _POLYMORPHIC_EXPECTED[(link_factory, a_lacks_x)]
    assert _nullable_column_values(results, first.get_class_name()) == [first_expected]
    assert _nullable_column_values(results, second.get_class_name()) == [second_expected]


RemedyArrowPairConsumer = _remedy_sum_consumer("RemedyArrowPairConsumer", PyArrowTable, ("mlg_a1", "mlg_b"))


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_direct_consumer_and_a_pair_consumer_on_the_framework_of_b_are_rejected(swap_link_sides: bool) -> None:
    first, second = RemedyArrowSameBConsumer, RemedyArrowPairConsumer
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            ParallelizationMode.SYNC,
            consumer=first,
            features=[first.get_class_name(), second.get_class_name()],
            **{**_REMEDY_KWARGS, "root_b": MultiLinkRootBSame, "extra_groups": {RemedyRootA1, second}},
        )

    message = str(exc_info.value)
    assert first.get_class_name() in message
    assert second.get_class_name() in message
    assert "Resolution:" in message


class RemedyRootBArrowX(RemedyRootBThird):
    """Root B variant on PyArrow providing mlg_bx."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_bx"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_bx": [1000, 2000, 3000], "mlg_idx": ["x", "y", "z"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RemedyBothSidesMid(FeatureGroup):
    """PyArrow mid reading members of both link sides."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mlg_a"), Feature("mlg_bx")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["mlg_a"], data["mlg_bx"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


RemedyBothSidesConsumer = _remedy_sum_consumer(
    "RemedyBothSidesConsumer", ThirdCfw, ("RemedyBothSidesMid", "mlg_a1", "mlg_b")
)


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_mid_over_both_link_sides_beside_a_same_framework_join_is_rejected(swap_link_sides: bool) -> None:
    with pytest.raises(ValueError, match="read both sides of a link"):
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=RemedyBothSidesConsumer,
            features=[RemedyBothSidesConsumer.get_class_name()],
            **{
                **_REMEDY_KWARGS,
                "root_b": RemedyRootBThird,
                "extra_groups": {*_REMEDY_KWARGS["extra_groups"], RemedyRootBArrowX, RemedyBothSidesMid},
            },
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


_B_SIDE_KEPT_SHAPES = [
    pytest.param(Link.inner, False, id="inner_a_b"),
    pytest.param(Link.inner, True, id="inner_b_a"),
    pytest.param(Link.outer, False, id="outer_a_b"),
    pytest.param(Link.outer, True, id="outer_b_a"),
    pytest.param(Link.left, True, id="left_keeps_b"),
    pytest.param(Link.right, False, id="right_keeps_b"),
]


def _remedy_second_b_prepare(
    link_factory: Any,
    swap_link_sides: bool,
    mode: ParallelizationMode,
    consumer: type[FeatureGroup] = RemedySecondBConsumer,
    **overrides: Any,
) -> Any:
    return _side_path_prepare(
        SidePathArrowP,
        False,
        swap_link_sides,
        mode,
        consumer=consumer,
        features=[consumer.get_class_name()],
        link_factory=link_factory,
        **{**_REMEDY_KWARGS, **overrides},
    )


def _remedy_second_b_run(
    link_factory: Any, swap_link_sides: bool, mode: ParallelizationMode, flight_server: Any = None
) -> list[list[int]]:
    session = _remedy_second_b_prepare(link_factory, swap_link_sides, mode)
    results = session.run(parallelization_modes={mode}, flight_server=flight_server)
    return _side_path_values(results, RemedySecondBConsumer)


def _assert_no_join_step_has_carriers(session: Any) -> None:
    assert session.engine is not None
    join_steps = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert not any(step.carriers for step in join_steps)


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize("link_factory, swap_link_sides", _B_SIDE_KEPT_SHAPES)
def test_a_consumer_on_the_other_sides_framework_joins_both_members_when_the_join_keeps_the_other_side(
    link_factory: Any, swap_link_sides: bool, mode: ParallelizationMode
) -> None:
    assert _remedy_second_b_run(link_factory, swap_link_sides, mode) == [[18, 30, 42]]


class RemedyRootA1Partial(MultiLinkRootAPartial):
    """Member of the partial root A on the third framework, keys y, z, q."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_a1"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a1": [7, 8, 9], "mlg_idx": ["y", "z", "q"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


class RemedyRootBFourKeys(MultiLinkRootBDistinct):
    """Root B with keys x, y, z, v."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_bd": [10, 20, 30, 50], "mlg_idx": ["x", "y", "z", "v"]}


def _remedy_null_safe_consumer(name: str, framework: type[ComputeFramework], inputs: tuple[str, ...]) -> Any:
    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        total = pc.fill_null(data[inputs[0]], 0)
        for column in inputs[1:]:
            total = pc.add(total, pc.fill_null(data[column], 0))
        return data.append_column(cls.get_class_name(), total)

    namespace = {
        "FRAMEWORK": framework,
        "INPUTS": inputs,
        "__module__": __name__,
        "calculate_feature": classmethod(calculate_feature),
    }
    return type(name, (_RemedyConsumer,), namespace)


RemedySecondNullSafeConsumer = _remedy_null_safe_consumer(
    "RemedySecondNullSafeConsumer", SecondCfw, _REMEDY_SECOND_INPUTS
)

# A {x,y,z,w}=1..4, A1 {y,z,q}=7,8,9, B {x,y,z,v}=10,20,30,50; a missing member counts as 0.
_B_SIDE_PARTIAL_ROWS = [
    pytest.param(Link.inner, False, [29, 41], id="inner_a_b"),
    pytest.param(Link.inner, True, [29, 41], id="inner_b_a"),
    pytest.param(Link.outer, False, [4, 9, 11, 29, 41, 50], id="outer_a_b"),
    pytest.param(Link.outer, True, [4, 9, 11, 29, 41, 50], id="outer_b_a"),
    pytest.param(Link.left, True, [11, 29, 41, 50], id="left_keeps_b"),
    pytest.param(Link.right, False, [11, 29, 41, 50], id="right_keeps_b"),
]


@pytest.mark.parametrize("link_factory, swap_link_sides, expected", _B_SIDE_PARTIAL_ROWS)
def test_a_consumer_on_the_other_sides_framework_keeps_the_rows_of_the_join_shape_with_partial_keys(
    link_factory: Any, swap_link_sides: bool, expected: list[int]
) -> None:
    session = _remedy_second_b_prepare(
        link_factory,
        swap_link_sides,
        ParallelizationMode.SYNC,
        consumer=RemedySecondNullSafeConsumer,
        root_a=MultiLinkRootAPartial,
        root_b=RemedyRootBFourKeys,
        extra_groups={RemedyRootA1Partial},
    )
    _assert_no_join_step_has_carriers(session)

    results = session.run(parallelization_modes={ParallelizationMode.SYNC})

    assert _side_path_values(results, RemedySecondNullSafeConsumer) == [expected]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_consumer_on_the_other_sides_framework_joins_both_members_with_multiprocessing(
    flight_server: Any, swap_link_sides: bool
) -> None:
    values = _remedy_second_b_run(Link.inner, swap_link_sides, ParallelizationMode.MULTIPROCESSING, flight_server)

    assert values == [[18, 30, 42]]


@pytest.mark.parametrize(
    "link_factory, swap_link_sides",
    [pytest.param(Link.left, False, id="left_keeps_members"), pytest.param(Link.right, True, id="right_keeps_members")],
)
def test_a_join_keeping_the_member_side_of_two_members_on_different_frameworks_is_rejected(
    link_factory: Any, swap_link_sides: bool
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=RemedySecondBConsumer,
            features=[RemedySecondBConsumer.get_class_name()],
            link_factory=link_factory,
            **_REMEDY_KWARGS,
        )

    message = str(exc_info.value)
    _assert_star_keeping_two_groups_reason(
        message, RemedySecondBConsumer.get_class_name(), ("MultiLinkRootA", "RemedyRootA1"), "MultiLinkRootBDistinct"
    )
    assert "reads both members directly" not in message


class RemedyAsofRootA(GeneratedFeatureGroup):
    """Time-stamped root A for ASOF links."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_asof_a", "mlg_ts_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_asof_a": [1, 2, 3, 4], "mlg_idx": ["x", "x", "y", "w"], "mlg_ts_a": [1, 5, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class RemedyAsofRootA1(RemedyAsofRootA):
    """Member of the ASOF root A on the third framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_asof_a1", "mlg_ts_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_asof_a1": [7, 8, 9], "mlg_idx": ["x", "y", "q"], "mlg_ts_a": [3, 9, 1]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ThirdCfw}


class RemedyAsofRootB(GeneratedFeatureGroup):
    """Time-stamped root B for ASOF links on the second framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_asof_b", "mlg_ts_b"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_asof_b": [10, 20, 30, 50], "mlg_idx": ["x", "x", "y", "v"], "mlg_ts_b": [0, 4, 1, 1]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SecondCfw}


RemedyAsofConsumer = _remedy_sum_consumer("RemedyAsofConsumer", SecondCfw, ("mlg_asof_a", "mlg_asof_a1", "mlg_asof_b"))


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["members_left", "members_right"])
def test_an_asof_join_of_two_members_on_different_frameworks_is_rejected_in_either_orientation(
    swap_link_sides: bool,
) -> None:
    left_time, right_time = ("mlg_ts_b", "mlg_ts_a") if swap_link_sides else ("mlg_ts_a", "mlg_ts_b")

    def asof_link(left: JoinSpec, right: JoinSpec) -> Link:
        return Link.asof(left, right, left_time_column=left_time, right_time_column=right_time)

    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=RemedyAsofConsumer,
            features=[RemedyAsofConsumer.get_class_name()],
            link_factory=asof_link,
            root_a=RemedyAsofRootA,
            root_b=RemedyAsofRootB,
            extra_groups={RemedyAsofRootA1},
            compute_frameworks=[PandasDataFrame, PyArrowTable, SecondCfw, ThirdCfw],
        )

    message = str(exc_info.value)
    assert "members of one link side" in message
    assert "RemedyAsofRootA" in message
    assert "RemedyAsofRootA1" in message
    assert "one member of the link side per consumer" in message
    assert "reads both members directly" not in message


class RemedyRootBSubclass(MultiLinkRootBDistinct):
    """Second member of the B link side on B's framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_bd2"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_bd2": [100, 200, 300], "mlg_idx": ["x", "y", "z"]}


RemedyBothSidesMembersConsumer = _remedy_sum_consumer(
    "RemedyBothSidesMembersConsumer", SecondCfw, ("mlg_a", "mlg_a1", "mlg_bd", "mlg_bd2")
)


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_members_on_both_link_sides_are_rejected_at_plan_time(swap_link_sides: bool) -> None:
    with pytest.raises(ValueError) as exc_info:
        _remedy_second_b_prepare(
            Link.inner,
            swap_link_sides,
            ParallelizationMode.SYNC,
            consumer=RemedyBothSidesMembersConsumer,
            extra_groups={RemedyRootA1, RemedyRootBSubclass},
        )

    assert "members of one link side" in str(exc_info.value)


class RemedyTwoNameConsumer(FeatureGroup):
    """B-framework consumer serving a direct and a mid-reading feature name."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"remedy_two_direct", "remedy_two_mid"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        first = "mlg_a" if str(feature_name) == "remedy_two_direct" else "SidePathArrowP"
        return {Feature(first), Feature("mlg_a1"), Feature("mlg_bd")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SecondCfw}


@pytest.mark.parametrize(
    "names",
    [("remedy_two_direct", "remedy_two_mid"), ("remedy_two_mid", "remedy_two_direct")],
    ids=["direct_first", "mid_first"],
)
def test_one_consumer_block_reading_the_members_both_directly_and_through_a_feature_is_rejected(
    names: tuple[str, str],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            consumer=RemedyTwoNameConsumer,
            features=list(names),
            **{**_REMEDY_KWARGS, "extra_groups": {RemedyRootA1, RemedyTwoNameConsumer}},
        )

    assert "members of one link side" in str(exc_info.value)


def test_a_consumer_on_one_members_framework_with_the_other_side_on_the_other_members_framework_is_infeasible() -> None:
    with pytest.raises(ValueError, match="No compute framework assignment satisfies the hard rules") as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            consumer=RemedyArrowSameBConsumer,
            features=[RemedyArrowSameBConsumer.get_class_name()],
            **{**_REMEDY_KWARGS, "root_b": RemedyRootBThird},
        )

    message = str(exc_info.value)
    assert "join inner" in message
    assert "members of one link side" not in message


class RemedyRootASibling(MultiLinkRootA):
    """Root A variant on A's own framework."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mlg_a2"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mlg_a2": [4, 5, 6], "mlg_idx": ["x", "y", "z"]}


RemedyArrowSiblingConsumer = _remedy_sum_consumer(
    "RemedyArrowSiblingConsumer", PyArrowTable, ("mlg_a", "mlg_a2", "mlg_bd")
)


_ONE_FRAMEWORK_REASONS = (
    "members of one link side",
    "one member of the link side per consumer",
    "declare one link per member",
    "each read by its own consumer",
    "Link.inner(JoinSpec(",
    ") and Link.inner(JoinSpec(",
)

RemedySecondSiblingConsumer = _remedy_sum_consumer(
    "RemedySecondSiblingConsumer", SecondCfw, ("mlg_a", "mlg_a2", "mlg_bd")
)
RemedyArrowSiblingSameBConsumer = _remedy_sum_consumer(
    "RemedyArrowSiblingSameBConsumer", PyArrowTable, ("mlg_a", "mlg_a2", "mlg_b")
)
RemedyArrowBMembersConsumer = _remedy_sum_consumer(
    "RemedyArrowBMembersConsumer", PyArrowTable, ("mlg_a", "mlg_bd", "mlg_bd2")
)
RemedySecondBMembersConsumer = _remedy_sum_consumer(
    "RemedySecondBMembersConsumer", SecondCfw, ("mlg_a", "mlg_bd", "mlg_bd2")
)

_A_MEMBERS = (MultiLinkRootA, RemedyRootASibling)
_B_MEMBERS = (MultiLinkRootBDistinct, RemedyRootBSubclass)

# (consumer, root_b, extra member group, the two members named by the reason)
_SAME_FRAMEWORK_MEMBER_LAYOUTS = [
    pytest.param(
        RemedyArrowSiblingConsumer,
        MultiLinkRootBDistinct,
        RemedyRootASibling,
        _A_MEMBERS,
        id="a_members_arrow_b_second",
    ),
    pytest.param(
        RemedySecondSiblingConsumer,
        MultiLinkRootBDistinct,
        RemedyRootASibling,
        _A_MEMBERS,
        id="a_members_second_consumer_b_second",
    ),
    pytest.param(
        RemedyArrowSiblingSameBConsumer, MultiLinkRootBSame, RemedyRootASibling, _A_MEMBERS, id="a_members_all_arrow"
    ),
    pytest.param(
        RemedyArrowBMembersConsumer, MultiLinkRootBDistinct, RemedyRootBSubclass, _B_MEMBERS, id="b_members_arrow"
    ),
    pytest.param(
        RemedySecondBMembersConsumer, MultiLinkRootBDistinct, RemedyRootBSubclass, _B_MEMBERS, id="b_members_second"
    ),
]


@pytest.mark.parametrize("empty_chooser_rules", [False, True], ids=["chooser", "planner_backstop"])
@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
@pytest.mark.parametrize("consumer, root_b, extra_group, members", _SAME_FRAMEWORK_MEMBER_LAYOUTS)
def test_one_consumer_reading_two_members_of_one_link_side_on_one_framework_over_one_base_link_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    consumer: type[FeatureGroup],
    root_b: type[FeatureGroup],
    extra_group: type[FeatureGroup],
    members: tuple[type[FeatureGroup], type[FeatureGroup]],
    swap_link_sides: bool,
    empty_chooser_rules: bool,
) -> None:
    if empty_chooser_rules:
        # Empty chooser rules reach the planner's backstops.
        monkeypatch.setattr(ChooseComputeFrameworks, "_side_member_rules", lambda self, *args, **kwargs: [])

    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=consumer,
            features=[consumer.get_class_name()],
            **{**_REMEDY_KWARGS, "root_b": root_b, "extra_groups": {extra_group}},
        )

    message = str(exc_info.value)
    assert consumer.get_class_name() in message
    for member in members:
        assert member.get_class_name() in message
    for reason in _ONE_FRAMEWORK_REASONS:
        assert reason in message


RemedySecondFirstMemberConsumer = _remedy_sum_consumer(
    "RemedySecondFirstMemberConsumer", SecondCfw, ("mlg_a", "mlg_bd")
)
RemedySecondSecondMemberConsumer = _remedy_sum_consumer(
    "RemedySecondSecondMemberConsumer", SecondCfw, ("mlg_a2", "mlg_bd")
)


def _two_consumers_one_member_each_prepare(links: set[Link] | None, swap_link_sides: bool = False) -> Any:
    consumers = (RemedySecondFirstMemberConsumer, RemedySecondSecondMemberConsumer)
    if links is None:
        left, right = (
            (MultiLinkRootBDistinct, MultiLinkRootA) if swap_link_sides else (MultiLinkRootA, MultiLinkRootBDistinct)
        )
        links = {Link.inner(JoinSpec(left, MLG_INDEX), JoinSpec(right, MLG_INDEX))}
    return mloda.prepare(
        [consumer.get_class_name() for consumer in consumers],
        links=links,
        compute_frameworks=[PandasDataFrame, PyArrowTable, SecondCfw, ThirdCfw],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {MultiLinkRootA, RemedyRootASibling, MultiLinkRootBDistinct, *consumers}
        ),
    )


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_two_consumers_each_reading_one_member_over_one_base_link_are_told_to_declare_one_link_per_member(
    swap_link_sides: bool,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _two_consumers_one_member_each_prepare(None, swap_link_sides)

    message = str(exc_info.value)
    assert "declare one link per member" in message
    assert "each read by its own consumer" in message
    assert "Link.inner(JoinSpec(MultiLinkRootA," in message
    assert "Link.inner(JoinSpec(RemedyRootASibling," in message
    assert "Link(MultiLinkRootA" not in message
    assert "MultiLinkRootA" in message
    assert "RemedyRootASibling" in message
    assert "one member of the link side per consumer" not in message


def test_two_consumers_each_reading_one_member_get_their_values_over_one_link_per_member() -> None:
    links = {
        Link.inner(JoinSpec(member, MLG_INDEX), JoinSpec(MultiLinkRootBDistinct, MLG_INDEX))
        for member in (MultiLinkRootA, RemedyRootASibling)
    }

    results = _two_consumers_one_member_each_prepare(links).run()

    assert _side_path_values(results, RemedySecondFirstMemberConsumer) == [[11, 22, 33]]
    assert _side_path_values(results, RemedySecondSecondMemberConsumer) == [[14, 25, 36]]


class RemedyRootAFree(RemedyRootASibling):
    """Second member of the A link side that may run on PyArrow or Pandas."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


RemedyArrowFreeMemberConsumer = _remedy_sum_consumer(
    "RemedyArrowFreeMemberConsumer", PyArrowTable, ("mlg_a", "mlg_a2", "mlg_b")
)


@pytest.mark.parametrize("swap_link_sides", [False, True], ids=["a_b", "b_a"])
def test_a_member_allowed_on_two_frameworks_is_pushed_apart_from_its_sibling_and_both_columns_return(
    swap_link_sides: bool,
) -> None:
    session = _side_path_prepare(
        SidePathArrowP,
        False,
        swap_link_sides,
        consumer=RemedyArrowFreeMemberConsumer,
        features=[RemedyArrowFreeMemberConsumer.get_class_name()],
        **{**_REMEDY_KWARGS, "root_b": MultiLinkRootBSame, "extra_groups": {RemedyRootAFree}},
    )

    results = session.run()

    assert _side_path_values(results, RemedyArrowFreeMemberConsumer) == [[15, 27, 39]]


def _star_group(name: str, column: str, rows: dict[str, int], framework: type[ComputeFramework]) -> type[FeatureGroup]:
    def feature_names_supported(cls: Any) -> set[str]:
        return {column}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        return {column: list(rows.values()), "mlg_idx": list(rows)}

    def compute_framework_rule(cls: Any) -> set[type[ComputeFramework]]:
        return {framework}

    namespace = {
        "__module__": __name__,
        "feature_names_supported": classmethod(feature_names_supported),
        "calculate_feature": classmethod(calculate_feature),
        "compute_framework_rule": classmethod(compute_framework_rule),
    }
    created = type(name, (GeneratedFeatureGroup,), namespace)
    globals()[name] = created  # module level, so multiprocessing can pickle the link back
    return created


_XYZ_KEYS = ("x", "y", "z")


def _xyz(values: list[int]) -> dict[str, int]:
    return dict(zip(_XYZ_KEYS, values))


_STAR_C_ROWS = {"y": 7, "z": 8, "w": 9, "q": 10}
_STAR_B_ROWS = {"x": 10, "y": 20, "z": 30, "v": 50, "u": 60}
StarSpokeCArrow = _star_group("StarSpokeCArrow", "star_c", _STAR_C_ROWS, PyArrowTable)
StarSpokeCSecond = _star_group("StarSpokeCSecond", "star_c", _STAR_C_ROWS, SecondCfw)
StarHubArrow = _star_group("StarHubArrow", "star_b", _STAR_B_ROWS, PyArrowTable)
StarConsumerArrow = _remedy_null_safe_consumer("StarConsumerArrow", PyArrowTable, ("mlg_a", "star_c", "star_b"))
_ALL_FRAMEWORKS = [PandasDataFrame, PyArrowTable, SecondCfw, ThirdCfw]


def _star_prepare(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
    hub_first: bool,
    spoke_c: type[FeatureGroup] = StarSpokeCArrow,
    second: tuple[Callable[[JoinSpec, JoinSpec], Link], bool] | None = None,
) -> Any:
    """Both spokes join the hub with link_factory; second, if given, is the spoke C link's (factory, hub_first)."""

    def link(spoke: type[FeatureGroup], factory: Callable[[JoinSpec, JoinSpec], Link], first: bool) -> Link:
        sides = (StarHubArrow, spoke) if first else (spoke, StarHubArrow)
        return factory(JoinSpec(sides[0], MLG_INDEX), JoinSpec(sides[1], MLG_INDEX))

    second_factory, second_hub_first = second or (link_factory, hub_first)
    return mloda.prepare(
        [StarConsumerArrow.get_class_name()],
        links={
            link(MultiLinkRootAPartial, link_factory, hub_first),
            link(spoke_c, second_factory, second_hub_first),
        },
        compute_frameworks=_ALL_FRAMEWORKS,
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {MultiLinkRootAPartial, spoke_c, StarHubArrow, StarConsumerArrow}
        ),
    )


def _assert_star_keeping_two_groups_reason(message: str, consumer: str, kept: tuple[str, str], dropped: str) -> None:
    assert consumer in message
    for name in (*kept, dropped):
        assert name in message
    assert "not defined" in message
    assert "Link.star(" in message
    assert 'index_column=..., jointype="left")' in message
    assert "right or left" not in message
    assert "inner joins for both or outer joins for both" in message
    assert "separate consumers" in message


@pytest.mark.parametrize(
    "spoke_c",
    [pytest.param(StarSpokeCArrow, id="all_on_one_framework"), pytest.param(StarSpokeCSecond, id="spokes_differ")],
)
def test_two_left_joins_keeping_different_spokes_of_a_shared_group_are_rejected_at_plan_time(
    spoke_c: type[FeatureGroup],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _star_prepare(Link.left, hub_first=False, spoke_c=spoke_c)

    _assert_star_keeping_two_groups_reason(
        str(exc_info.value),
        StarConsumerArrow.get_class_name(),
        ("MultiLinkRootAPartial", spoke_c.get_class_name()),
        "StarHubArrow",
    )


def test_two_right_joins_keeping_different_spokes_of_a_shared_group_still_raise() -> None:
    with pytest.raises(ValueError, match="have multiple right joins for the same feature group on the left side"):
        _star_prepare(Link.right, hub_first=True)


_ORDER_DEPENDENT_STAR_ROWS = [
    pytest.param(Link.left, False, (Link.inner, False), id="left_keeping_a_spoke_plus_inner"),
    pytest.param(Link.left, False, (Link.outer, False), id="left_keeping_a_spoke_plus_outer"),
    pytest.param(Link.left, True, (Link.outer, False), id="left_keeping_the_hub_plus_outer"),
    pytest.param(Link.inner, False, (Link.outer, False), id="inner_plus_outer"),
    pytest.param(Link.left, False, (Link.left, True), id="chain_through_the_hub"),
    pytest.param(Link.right, False, (Link.left, False), id="right_keeping_the_hub_plus_left_dropping_it"),
]


@pytest.mark.parametrize("spoke_c", [StarSpokeCArrow, StarSpokeCSecond], ids=["one_framework", "spokes_differ"])
@pytest.mark.parametrize("link_factory, hub_first, second", _ORDER_DEPENDENT_STAR_ROWS)
def test_a_star_whose_rows_depend_on_which_join_runs_first_is_rejected_at_plan_time(
    link_factory: Any, hub_first: bool, second: Any, spoke_c: type[FeatureGroup]
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _star_prepare(link_factory, hub_first, spoke_c, second)

    message = str(exc_info.value)
    for name in (StarConsumerArrow.get_class_name(), "MultiLinkRootAPartial", spoke_c.get_class_name(), "StarHubArrow"):
        assert name in message, f"{name} must be named; got: {message}"
    assert "runs first" in message
    assert "separate consumers" in message


_STAR_ROWS = [
    pytest.param(Link.left, True, [11, 29, 41, 50, 60], id="left_keeps_hub"),
    pytest.param(Link.inner, False, [29, 41], id="inner"),
    pytest.param(Link.outer, False, [10, 11, 13, 29, 41, 50, 60], id="outer"),
]


@pytest.mark.parametrize("spoke_c", [StarSpokeCArrow, StarSpokeCSecond], ids=["one_framework", "spokes_differ"])
@pytest.mark.parametrize("link_factory, hub_first, expected", _STAR_ROWS)
def test_a_star_with_a_defined_result_returns_its_rows_when_the_spokes_hold_different_keys(
    link_factory: Any, hub_first: bool, expected: list[int], spoke_c: type[FeatureGroup]
) -> None:
    results = _star_prepare(link_factory, hub_first, spoke_c).run()

    assert _side_path_values(results, StarConsumerArrow) == [expected]


@pytest.mark.parametrize("spoke_c", [StarSpokeCArrow, StarSpokeCSecond], ids=["one_framework", "spokes_differ"])
def test_a_left_join_keeping_the_hub_beside_an_inner_join_returns_the_inner_rows_with_every_spoke_value(
    spoke_c: type[FeatureGroup],
) -> None:
    results = _star_prepare(Link.left, True, spoke_c, (Link.inner, False)).run()

    assert _side_path_values(results, StarConsumerArrow) == [[29, 41]]


@pytest.mark.parametrize("spoke_c", [StarSpokeCArrow, StarSpokeCSecond], ids=["one_framework", "spokes_differ"])
@pytest.mark.parametrize("link_factory, hub_first", [pytest.param(Link.inner, False), pytest.param(Link.left, True)])
def test_a_join_group_consumer_is_never_left_stranded_in_a_frames_keep_alive_after_a_star_run(
    link_factory: Any, hub_first: bool, spoke_c: type[FeatureGroup]
) -> None:
    session = _star_prepare(link_factory, hub_first, spoke_c)
    session.run()

    assert session.runner is not None
    stranded = {
        type(cfw).__name__: cfw.keep_alive - cfw.already_calculated_children_tracker
        for cfw in session.runner.executor.cfw_collection.values()
        if cfw.data is not None and cfw.keep_alive - cfw.already_calculated_children_tracker
    }
    assert stranded == {}


RemedyRootBFourKeysArrow = type(
    "RemedyRootBFourKeysArrow",
    (RemedyRootBFourKeys,),
    {"__module__": __name__, "compute_framework_rule": classmethod(lambda cls: {PyArrowTable})},
)
RemedyRootBFourKeysThird = type(
    "RemedyRootBFourKeysThird",
    (RemedyRootBFourKeys,),
    {"__module__": __name__, "compute_framework_rule": classmethod(lambda cls: {ThirdCfw})},
)
_POLY_INPUTS = ("mlg_a", "mlg_a1", "mlg_bd")
_POLY_B_LAYOUTS = [
    pytest.param(
        RemedyRootBFourKeysThird,
        _remedy_null_safe_consumer("StarPolyThirdConsumer", ThirdCfw, _POLY_INPUTS),
        id="b_on_a1_framework",
    ),
    pytest.param(
        RemedyRootBFourKeysArrow,
        _remedy_null_safe_consumer("StarPolyArrowConsumer", PyArrowTable, _POLY_INPUTS),
        id="b_on_a_framework",
    ),
    pytest.param(
        RemedyRootBFourKeys,
        _remedy_null_safe_consumer("StarPolySecondConsumer", SecondCfw, _POLY_INPUTS),
        id="b_on_third_framework",
    ),
]


@pytest.mark.parametrize(
    "link_factory, swap_link_sides",
    [pytest.param(Link.left, False, id="left_a_b"), pytest.param(Link.right, True, id="right_b_a")],
)
@pytest.mark.parametrize("root_b, consumer", _POLY_B_LAYOUTS)
def test_a_left_or_right_join_keeping_two_members_of_one_link_side_with_different_keys_is_rejected(
    root_b: type[FeatureGroup], consumer: type[FeatureGroup], link_factory: Any, swap_link_sides: bool
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _side_path_prepare(
            SidePathArrowP,
            False,
            swap_link_sides,
            consumer=consumer,
            features=[consumer.get_class_name()],
            link_factory=link_factory,
            root_a=MultiLinkRootAPartial,
            root_b=root_b,
            extra_groups={RemedyRootA1Partial},
            compute_frameworks=_ALL_FRAMEWORKS,
        )

    _assert_star_keeping_two_groups_reason(
        str(exc_info.value),
        consumer.get_class_name(),
        ("MultiLinkRootAPartial", "RemedyRootA1Partial"),
        root_b.get_class_name(),
    )


def _below_list(column: Any) -> list[Any]:
    return list(column.to_pylist() if hasattr(column, "to_pylist") else column.tolist())


def _below_values(column: Any) -> list[int]:
    return [int(v) for v in _below_list(column)]


def _below_consumer(
    name: str, framework: type[ComputeFramework], inputs: tuple[str, ...], scale: int = 1
) -> type[FeatureGroup]:
    def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(column) for column in inputs}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        columns = [_below_values(data[column]) for column in inputs]
        total = [scale * columns[0][i] + sum(c[i] for c in columns[1:]) for i in range(len(columns[0]))]
        # Keep the key so a later link can join this consumer's frame.
        return {name: total, "mlg_idx": [str(v) for v in _below_list(data["mlg_idx"])]}

    def compute_framework_rule(cls: Any) -> set[type[ComputeFramework]]:
        return {framework}

    namespace = {
        "__module__": __name__,
        "input_features": input_features,
        "calculate_feature": classmethod(calculate_feature),
        "compute_framework_rule": classmethod(compute_framework_rule),
    }
    created = type(name, (FeatureGroup,), namespace)
    globals()[name] = created  # module level, so multiprocessing can pickle the link back
    return created


def _below_layout(
    tag: str,
    source: type[ComputeFramework],
    other: type[ComputeFramework],
    joined_and_d_framework: type[ComputeFramework] | None = None,
) -> dict[str, Any]:
    """Root A on `source`, root B and the consumer's child on `other`, the consumer on A's framework."""
    root_a = _star_group(f"BelowRootA{tag}", f"below_a_{tag}", _xyz([1, 2, 3]), source)
    root_b = _star_group(f"BelowRootB{tag}", f"below_b_{tag}", _xyz([10, 20, 30]), other)
    root_d = _star_group(f"BelowRootD{tag}", f"below_d_{tag}", _xyz([7, 8, 9]), PandasDataFrame)
    joined_inputs = (f"below_a_{tag}", f"below_b_{tag}")
    joined = _below_consumer(f"BelowJoined{tag}", source, joined_inputs)
    layout = {
        "a": root_a,
        "b": root_b,
        "d": root_d,
        "joined": joined,
        "below": _below_consumer(f"BelowChild{tag}", other, (joined.get_class_name(),), scale=2),
        "bd": _below_consumer(f"BelowBD{tag}", PandasDataFrame, (f"below_b_{tag}", f"below_d_{tag}")),
    }
    if joined_and_d_framework is not None:
        inputs = (joined.get_class_name(), f"below_d_{tag}")
        layout["below_jd"] = _below_consumer(f"BelowJD{tag}", joined_and_d_framework, inputs, scale=2)
    return layout


_BELOW_LAYOUTS = {
    "arrow_source": _below_layout("Pa", PyArrowTable, PandasDataFrame),
    "pandas_source": _below_layout("Pd", PandasDataFrame, PyArrowTable),
    "jd_arrow_pandas": _below_layout("PaJd", PyArrowTable, PandasDataFrame, PandasDataFrame),
    "jd_pandas_arrow": _below_layout("PdJd", PandasDataFrame, PyArrowTable, PandasDataFrame),
    "jd_joined_framework": _below_layout("PaJf", PyArrowTable, PandasDataFrame, PyArrowTable),
}
_BELOW_PLAIN_LAYOUTS = ("arrow_source", "pandas_source")
_BELOW_JD_LAYOUTS = ("jd_arrow_pandas", "jd_pandas_arrow", "jd_joined_framework")


def _below_link_set(
    layout: dict[str, Any],
    swap_link_sides: bool,
    links: tuple[tuple[str, str], ...],
    swap_second_link: bool = False,
) -> set[Link]:
    """The first link flips with `swap_link_sides`, the second with `swap_second_link`."""
    flips = (swap_link_sides, swap_second_link)
    link_set = set()
    for position, (left, right) in enumerate(links):
        flip = flips[position] if position < len(flips) else False
        first, second = (right, left) if flip else (left, right)
        link_set.add(Link.inner(JoinSpec(layout[first], MLG_INDEX), JoinSpec(layout[second], MLG_INDEX)))
    return link_set


def _below_run(
    layout_name: str,
    swap_link_sides: bool,
    mode: ParallelizationMode,
    requested: tuple[str, ...],
    links: tuple[tuple[str, str], ...] = (("a", "b"),),
    flight_server: Any = None,
    swap_second_link: bool = False,
) -> dict[str, list[list[int]]]:
    layout = _BELOW_LAYOUTS[layout_name]
    link_set = _below_link_set(layout, swap_link_sides, links, swap_second_link)
    names = [layout[key].get_class_name() for key in requested]
    results = mloda.run_all(
        names,
        links=link_set,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={mode},
        flight_server=flight_server,
        plugin_collector=PluginCollector.enabled_feature_groups(set(layout.values())),
    )
    return {key: _column_values(results, layout[key].get_class_name()) for key in requested}


_BELOW_ALONE = (("below",), (("a", "b"),), {"below": [[22, 44, 66]]})
_BELOW_NEXT_TO_SECOND_LINK = (
    ("below", "bd"),
    (("a", "b"), ("b", "d")),
    {"below": [[22, 44, 66]], "bd": [[17, 28, 39]]},
)
_BELOW_CASES = [
    pytest.param(layout, swap, mode, *variant, id=f"{name}-{layout}-{'b_a' if swap else 'a_b'}-{mode.name.lower()}")
    for name, variant in (("alone", _BELOW_ALONE), ("second_link", _BELOW_NEXT_TO_SECOND_LINK))
    for layout in _BELOW_PLAIN_LAYOUTS
    for swap in (False, True)
    for mode in (ParallelizationMode.SYNC, ParallelizationMode.THREADING)
]
_BELOW_MULTIPROCESSING = pytest.param(
    "arrow_source",
    False,
    ParallelizationMode.MULTIPROCESSING,
    *_BELOW_ALONE,
    # Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
    marks=pytest.mark.timeout(60),
    id="alone-arrow_source-a_b-multiprocessing",
)


@pytest.mark.parametrize(
    "layout, swap_link_sides, mode, requested, links, expected", [*_BELOW_CASES, _BELOW_MULTIPROCESSING]
)
def test_a_feature_below_a_join_consumer_reads_the_joined_frame(
    request: pytest.FixtureRequest,
    layout: str,
    swap_link_sides: bool,
    mode: ParallelizationMode,
    requested: tuple[str, ...],
    links: tuple[tuple[str, str], ...],
    expected: dict[str, list[list[int]]],
) -> None:
    server = request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
    values = _below_run(layout, swap_link_sides, mode, requested, links=links, flight_server=server)

    assert values == expected


_BELOW_JOINED_AND_D = (("a", "b"), ("joined", "d"))


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING], ids=["sync", "threading"])
@pytest.mark.parametrize("swap_second_link", [False, True], ids=["joined_d", "d_joined"])
@pytest.mark.parametrize("layout", _BELOW_JD_LAYOUTS)
def test_a_feature_reading_a_join_consumer_and_a_root_joined_to_it_returns_its_values(
    layout: str, swap_second_link: bool, mode: ParallelizationMode
) -> None:
    values = _below_run(
        layout, False, mode, ("below_jd",), links=_BELOW_JOINED_AND_D, swap_second_link=swap_second_link
    )

    assert values == {"below_jd": [[29, 52, 75]]}


@pytest.mark.parametrize("swap_second_link", [False, True], ids=["joined_d", "d_joined"])
@pytest.mark.parametrize("layout", _BELOW_JD_LAYOUTS)
def test_a_feature_reading_a_join_consumer_and_a_root_anchors_its_frame_on_the_join_destination(
    layout: str, swap_second_link: bool
) -> None:
    fixture = _BELOW_LAYOUTS[layout]
    link_set = _below_link_set(fixture, False, _BELOW_JOINED_AND_D, swap_second_link)
    session = mloda.prepare(
        [fixture["below_jd"].get_class_name()],
        links=link_set,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups(set(fixture.values())),
    )

    assert session.engine is not None
    planner = session.engine.execution_planner
    steps = [s for s in planner if isinstance(s, FeatureGroupStep) and s.feature_group is fixture["below_jd"]]
    assert len(steps) == 1
    records = [r for r in planner.resolved_join_plan.records if r.consumers & steps[0].get_uuids()]
    assert records
    assert steps[0].features.any_uuid in set().union(*(r.destination_uuids for r in records))
