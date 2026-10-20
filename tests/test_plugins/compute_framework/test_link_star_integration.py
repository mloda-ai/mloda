"""End-to-end integration coverage for ``Link.star`` through the real mloda engine.

These tests build a genuine star topology (one hub feature group plus multiple
spoke feature groups, all sharing a single ``"row_id"`` index column) and run the
result through ``mloda.run_all`` on the pandas compute framework. Unlike the unit
tests for ``Link.star`` (which only inspect the returned ``Link`` set), these
exercise the joins the engine actually performs and assert on concrete joined
cell values.
"""

from typing import Any

import pyarrow.compute as pc
import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, BaseInputData, DataCreator
from mloda.user import (
    Feature,
    FeatureName,
    Index,
    Link,
    JoinSpec,
    JoinType,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)

import mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# === Fully-matching star topology (all FGs expose row_id [1, 2, 3]) ===


class StarHubFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "hub_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "hub_value": ["h1", "h2", "h3"]}


class StarSpokeAFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "spoke_a_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "spoke_a_value": [10, 20, 30]}


class StarSpokeBFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "spoke_b_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "spoke_b_value": ["x", "y", "z"]}


class StarConsumerFG(FeatureGroup):
    """Downstream consumer that combines the hub column with both spoke columns.

    Requesting all three features forces the engine to materialise the star join
    before this group runs, so the merged frame is what ``calculate_feature`` sees.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="hub_value"),
            Feature(name="spoke_a_value"),
            Feature(name="spoke_b_value"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = (
            data["hub_value"] + "|" + data["spoke_a_value"].astype(str) + "|" + data["spoke_b_value"]
        )
        return data


# === Partial star topology (spoke B is missing row_id 3) to observe LEFT vs INNER ===


class StarLeftHubFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "left_hub_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "left_hub_value": ["h1", "h2", "h3"]}


class StarLeftSpokeAFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "left_spoke_a_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "left_spoke_a_value": [10, 20, 30]}


class StarLeftSpokeBFG(FeatureGroup):
    """Spoke B is intentionally missing row_id 3 so INNER drops it and LEFT keeps it."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "left_spoke_b_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2], "left_spoke_b_value": ["x", "y"]}


class StarLeftConsumerFG(FeatureGroup):
    """Consumer for the partial topology.

    It requests all three features (so both spoke joins run) but only combines the
    always-present hub and spoke-A columns, avoiding NaN concatenation on the row
    that spoke B does not supply.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="left_hub_value"),
            Feature(name="left_spoke_a_value"),
            Feature(name="left_spoke_b_value"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["left_hub_value"] + "|" + data["left_spoke_a_value"].astype(str)
        return data


# === Outer star topology (spoke has a row_id the hub lacks) to observe OUTER vs INNER ===


class StarOuterHubFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "outer_hub_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "outer_hub_value": ["h1", "h2", "h3"]}


class StarOuterSpokeAFG(FeatureGroup):
    """Spoke carries row_id 4 (which the hub lacks) and lacks row_id 1 (which the hub has)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"row_id", "outer_spoke_a_value"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [2, 3, 4], "outer_spoke_a_value": [20, 30, 40]}


class StarOuterConsumerFG(FeatureGroup):
    """Consumer for the outer topology.

    It requests both features (so the join runs) but only passes the join key
    ``row_id`` through, so no possibly-null column is string-concatenated.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="outer_hub_value"),
            Feature(name="outer_spoke_a_value"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["row_id"]
        return data


class TestLinkStarIntegration:
    def test_default_inner_star_end_to_end(self, flight_server: Any) -> None:
        """Default inner ``Link.star`` produces a 3-row merged frame with exact combined values."""
        links = Link.star(StarHubFG, StarSpokeAFG, StarSpokeBFG, index_column="row_id")

        result = mloda.run_all(
            [Feature(name=StarConsumerFG.get_class_name())],
            links=links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {StarHubFG, StarSpokeAFG, StarSpokeBFG, StarConsumerFG}
            ),
            flight_server=flight_server,
        )

        assert len(result) == 1
        res = result[0]
        assert len(res) == 3
        column = StarConsumerFG.get_class_name()
        assert column in res.columns
        assert set(res[column].values) == {"h1|10|x", "h2|20|y", "h3|30|z"}

    def test_star_matches_manual_inner_links(self, flight_server: Any) -> None:
        """``Link.star`` is a faithful drop-in for a hand-built loop of ``Link.inner`` links."""
        star_links = Link.star(StarHubFG, StarSpokeAFG, StarSpokeBFG, index_column="row_id")
        manual_links = {
            Link.inner(JoinSpec(StarHubFG, Index(("row_id",))), JoinSpec(StarSpokeAFG, Index(("row_id",)))),
            Link.inner(JoinSpec(StarHubFG, Index(("row_id",))), JoinSpec(StarSpokeBFG, Index(("row_id",)))),
        }

        collector = PluginCollector.enabled_feature_groups({StarHubFG, StarSpokeAFG, StarSpokeBFG, StarConsumerFG})
        column = StarConsumerFG.get_class_name()

        star_result = mloda.run_all(
            [Feature(name=column)],
            links=star_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )
        manual_result = mloda.run_all(
            [Feature(name=column)],
            links=manual_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )

        assert len(star_result) == 1
        assert len(manual_result) == 1
        star_values = sorted(star_result[0][column].values)
        manual_values = sorted(manual_result[0][column].values)
        assert star_values == manual_values
        assert star_values == ["h1|10|x", "h2|20|y", "h3|30|z"]

    def test_left_star_preserves_hub_rows(self, flight_server: Any) -> None:
        """A LEFT ``Link.star`` keeps every hub row while an inner star drops the unmatched one."""
        left_links = Link.star(
            StarLeftHubFG,
            StarLeftSpokeAFG,
            StarLeftSpokeBFG,
            index_column="row_id",
            jointype=JoinType.LEFT,
        )
        inner_links = Link.star(
            StarLeftHubFG,
            StarLeftSpokeAFG,
            StarLeftSpokeBFG,
            index_column="row_id",
        )

        collector = PluginCollector.enabled_feature_groups(
            {StarLeftHubFG, StarLeftSpokeAFG, StarLeftSpokeBFG, StarLeftConsumerFG}
        )
        column = StarLeftConsumerFG.get_class_name()

        left_result = mloda.run_all(
            [Feature(name=column)],
            links=left_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )
        inner_result = mloda.run_all(
            [Feature(name=column)],
            links=inner_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )

        assert len(left_result) == 1
        left_res = left_result[0]
        # LEFT star keeps all three hub rows (row_id 3 has no spoke-B match).
        assert len(left_res) == 3
        assert set(left_res[column].values) == {"h1|10", "h2|20", "h3|30"}

        # INNER star on the same data drops the unmatched hub row: observable contrast.
        assert len(inner_result) == 1
        inner_res = inner_result[0]
        assert len(inner_res) == 2
        assert set(inner_res[column].values) == {"h1|10", "h2|20"}

    def test_manual_left_links_keeping_the_hub_match_a_left_star(self, flight_server: Any) -> None:
        """Hand-built hub-kept left links return what the LEFT ``Link.star`` returns."""
        index = Index(("row_id",))
        star_links = Link.star(
            StarLeftHubFG, StarLeftSpokeAFG, StarLeftSpokeBFG, index_column="row_id", jointype=JoinType.LEFT
        )
        manual_links = {
            Link.left(JoinSpec(StarLeftHubFG, index), JoinSpec(StarLeftSpokeAFG, index)),
            Link.left(JoinSpec(StarLeftHubFG, index), JoinSpec(StarLeftSpokeBFG, index)),
        }
        collector = PluginCollector.enabled_feature_groups(
            {StarLeftHubFG, StarLeftSpokeAFG, StarLeftSpokeBFG, StarLeftConsumerFG}
        )
        column = StarLeftConsumerFG.get_class_name()

        results = [
            mloda.run_all(
                [Feature(name=column)],
                links=links,
                compute_frameworks=["PandasDataFrame"],
                plugin_collector=collector,
                flight_server=flight_server,
            )
            for links in (star_links, manual_links)
        ]

        assert [len(result) for result in results] == [1, 1]
        star_values, manual_values = (sorted(result[0][column].values) for result in results)
        assert manual_values == star_values == ["h1|10", "h2|20", "h3|30"]

    def test_outer_star_keeps_unmatched_spoke_row(self, flight_server: Any) -> None:
        """An OUTER ``Link.star`` keeps the spoke row_id the hub lacks; an inner star drops it."""
        outer_links = Link.star(
            StarOuterHubFG,
            StarOuterSpokeAFG,
            index_column="row_id",
            jointype=JoinType.OUTER,
        )
        inner_links = Link.star(
            StarOuterHubFG,
            StarOuterSpokeAFG,
            index_column="row_id",
        )

        collector = PluginCollector.enabled_feature_groups({StarOuterHubFG, StarOuterSpokeAFG, StarOuterConsumerFG})
        column = StarOuterConsumerFG.get_class_name()

        outer_result = mloda.run_all(
            [Feature(name=column)],
            links=outer_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )
        inner_result = mloda.run_all(
            [Feature(name=column)],
            links=inner_links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=collector,
            flight_server=flight_server,
        )

        # OUTER keeps the full union of row_ids: hub-only (1), matched (2, 3) and spoke-only (4).
        assert len(outer_result) == 1
        outer_ids = {int(v) for v in outer_result[0][column].values}
        assert outer_ids == {1, 2, 3, 4}
        assert 4 in outer_ids

        # INNER keeps only the matched row_ids, dropping the spoke-only row 4: observable contrast.
        assert len(inner_result) == 1
        inner_ids = {int(v) for v in inner_result[0][column].values}
        assert inner_ids == {2, 3}
        assert 4 not in inner_ids


# === Hub on the right of both links: two joins share only their source side ===

HUB_RIGHT_INDEX = Index(("idx",))
_HR_A_ROWS = {"x": 1, "y": 2, "z": 3, "w": 4}
_HR_C_ROWS = {"y": 7, "z": 8, "w": 9, "q": 10}
_HR_B_ROWS = {"x": 10, "y": 20, "z": 30, "v": 50, "u": 60}


def _hub_right_group(
    name: str, column: str, rows: dict[str, int], framework: type[ComputeFramework]
) -> type[FeatureGroup]:
    def input_data(cls: Any) -> BaseInputData | None:
        return DataCreator({column})

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        return {"idx": list(rows), column: list(rows.values())}

    def compute_framework_rule(cls: Any) -> set[type[ComputeFramework]]:
        return {framework}

    namespace = {
        "__module__": __name__,
        "input_data": classmethod(input_data),
        "calculate_feature": classmethod(calculate_feature),
        "compute_framework_rule": classmethod(compute_framework_rule),
    }
    return type(name, (FeatureGroup,), namespace)


HubRightA = _hub_right_group("HubRightA", "hr_a", _HR_A_ROWS, PyArrowTable)
HubRightB = _hub_right_group("HubRightB", "hr_b", _HR_B_ROWS, PyArrowTable)
HubRightCArrow = _hub_right_group("HubRightCArrow", "hr_c", _HR_C_ROWS, PyArrowTable)
HubRightCPandas = _hub_right_group("HubRightCPandas", "hr_c", _HR_C_ROWS, PandasDataFrame)


class HubRightConsumer(FeatureGroup):
    """Sums a, c and b per row, treating a missing value as 0."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_a"), Feature("hr_c"), Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.fill_null(data["hr_a"], 0)
        for column in ("hr_c", "hr_b"):
            total = pc.add(total, pc.fill_null(data[column], 0))
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _hub_right_run(
    link_factory: Any, spoke_c: type[FeatureGroup], mode: ParallelizationMode, flight_server: Any = None
) -> list[int]:
    links = {
        link_factory(JoinSpec(HubRightA, HUB_RIGHT_INDEX), JoinSpec(HubRightB, HUB_RIGHT_INDEX)),
        link_factory(JoinSpec(spoke_c, HUB_RIGHT_INDEX), JoinSpec(HubRightB, HUB_RIGHT_INDEX)),
    }
    results = mloda.run_all(
        [Feature(name=HubRightConsumer.get_class_name())],
        links=links,
        compute_frameworks=[PyArrowTable, PandasDataFrame],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups({HubRightA, HubRightB, spoke_c, HubRightConsumer}),
        flight_server=flight_server,
    )
    column = HubRightConsumer.get_class_name()
    return [sorted(r[column].to_pylist()) for r in results if column in r.column_names][0]


_HUB_RIGHT_INNER_EXPECTED = [29, 41]
_HUB_RIGHT_OUTER_EXPECTED = [10, 11, 13, 29, 41, 50, 60]
# The failure is a race, so a single THREADING run can pass by luck.
_THREADING_REPEATS = 4


class TestHubOnTheRightOfBothLinks:
    @pytest.mark.parametrize(
        "link_factory, expected, spoke_c, mode",
        [
            pytest.param(
                Link.inner, _HUB_RIGHT_INNER_EXPECTED, spoke_c, mode, id=f"inner-{spoke_id}-{mode.name.lower()}"
            )
            for spoke_c, spoke_id in ((HubRightCArrow, "one_framework"), (HubRightCPandas, "spokes_differ"))
            for mode in (ParallelizationMode.SYNC, ParallelizationMode.THREADING)
        ]
        + [
            pytest.param(
                Link.outer,
                _HUB_RIGHT_OUTER_EXPECTED,
                HubRightCArrow,
                ParallelizationMode.THREADING,
                id="outer-one_framework-threading",
            )
        ],
    )
    def test_two_links_sharing_the_right_side_return_their_rows(
        self, link_factory: Any, expected: list[int], spoke_c: type[FeatureGroup], mode: ParallelizationMode
    ) -> None:
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            assert _hub_right_run(link_factory, spoke_c, mode) == expected

    @pytest.mark.timeout(60)
    def test_two_inner_links_sharing_the_right_side_return_the_matched_rows_with_multiprocessing(
        self, flight_server: Any
    ) -> None:
        rows = _hub_right_run(Link.inner, HubRightCArrow, ParallelizationMode.MULTIPROCESSING, flight_server)

        assert rows == _HUB_RIGHT_INNER_EXPECTED


class HubRightPairConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_a"), Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["hr_a"], data["hr_b"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class HubRightPairChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(HubRightPairConsumer.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data[HubRightPairConsumer.get_class_name()], 1))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def test_one_join_feeding_a_consumer_and_its_child_runs_with_threading() -> None:
    child = HubRightPairChild.get_class_name()
    for _ in range(_THREADING_REPEATS):
        results = mloda.run_all(
            [Feature(name=child), Feature(name=HubRightPairConsumer.get_class_name())],
            links={Link.inner(JoinSpec(HubRightA, HUB_RIGHT_INDEX), JoinSpec(HubRightB, HUB_RIGHT_INDEX))},
            compute_frameworks=[PyArrowTable],
            parallelization_modes={ParallelizationMode.THREADING},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {HubRightA, HubRightB, HubRightPairConsumer, HubRightPairChild}
            ),
        )

        values = [sorted(r[child].to_pylist()) for r in results if child in r.column_names]
        assert values == [[12, 23, 34]]


# === Two consumers each reading one spoke of the same star ===


class HubRightCPairConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_c"), Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["hr_c"], data["hr_b"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class HubRightB2(FeatureGroup):
    """Feature derived from hr_b on PyArrow."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.multiply(data["hr_b"], 100))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class HubRightCDerivedConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_c"), Feature(HubRightB2.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data["hr_c"], data[HubRightB2.get_class_name()])
        return data.append_column(cls.get_class_name(), total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _rows(results: list[Any], name: str, nullable: bool = False) -> list[Any]:
    """Sorted values of the one result holding `name`; nulls sort last when `nullable`."""
    (values,) = [r[name].to_pylist() for r in results if name in r.column_names]
    return sorted(values, key=lambda v: (v is None, 0 if v is None else v)) if nullable else sorted(values)


def _two_consumers_run(
    links: set[Link],
    spoke_c: type[FeatureGroup],
    y_consumer: type[FeatureGroup],
    mode: ParallelizationMode,
    flight_server: Any = None,
    extra: set[type[FeatureGroup]] | None = None,
    x_consumer: type[FeatureGroup] = HubRightPairConsumer,
    nullable: bool = False,
) -> dict[str, list[Any]]:
    groups: set[type[FeatureGroup]] = {HubRightA, HubRightB, spoke_c, x_consumer, y_consumer}
    groups |= extra or set()
    results = mloda.run_all(
        [Feature(name=x_consumer.get_class_name()), Feature(name=y_consumer.get_class_name())],
        links=links,
        compute_frameworks=[PyArrowTable, PandasDataFrame],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        flight_server=flight_server,
    )
    names = (x_consumer.get_class_name(), y_consumer.get_class_name())
    return {n: _rows(results, n, nullable) for n in names}


def _star_links(
    spoke_c: type[FeatureGroup],
    hub_right: bool,
    hub_c: type[FeatureGroup] = HubRightB,
    c_link: Any = Link.inner,
) -> set[Link]:
    a, b, c = (
        JoinSpec(HubRightA, HUB_RIGHT_INDEX),
        JoinSpec(HubRightB, HUB_RIGHT_INDEX),
        JoinSpec(spoke_c, HUB_RIGHT_INDEX),
    )
    c_hub = JoinSpec(hub_c, HUB_RIGHT_INDEX)
    if hub_right:
        return {Link.inner(a, b), c_link(c, c_hub)}
    return {Link.inner(b, a), c_link(c_hub, c)}


_X_EXPECTED = [11, 22, 33]
_Y_EXPECTED = [27, 38]
_SPOKES = [HubRightCArrow, HubRightCPandas]
_SPOKE_IDS = ["one_framework", "spokes_differ"]
_ORIENTATION_IDS = ["spokes_left", "hub_left"]


class TestTwoConsumersReadingDifferentSpokes:
    @pytest.mark.parametrize(
        "spoke_c, hub_right, mode",
        [
            pytest.param(
                spoke_c,
                hub_right,
                mode,
                id=f"{spoke_id}-{'spokes_left' if hub_right else 'hub_left'}-{mode.name.lower()}",
            )
            for spoke_c, spoke_id in ((HubRightCArrow, "one_framework"), (HubRightCPandas, "spokes_differ"))
            for hub_right in (True, False)
            for mode in (ParallelizationMode.SYNC, ParallelizationMode.THREADING)
        ],
    )
    def test_each_consumer_gets_only_its_own_join_rows(
        self, spoke_c: type[FeatureGroup], hub_right: bool, mode: ParallelizationMode
    ) -> None:
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(_star_links(spoke_c, hub_right), spoke_c, HubRightCPairConsumer, mode)

            assert got[HubRightPairConsumer.get_class_name()] == _X_EXPECTED
            assert got[HubRightCPairConsumer.get_class_name()] == _Y_EXPECTED

    @pytest.mark.timeout(60)
    def test_each_consumer_gets_only_its_own_join_rows_with_multiprocessing(self, flight_server: Any) -> None:
        got = _two_consumers_run(
            _star_links(HubRightCArrow, True),
            HubRightCArrow,
            HubRightCPairConsumer,
            ParallelizationMode.MULTIPROCESSING,
            flight_server,
        )

        assert got[HubRightPairConsumer.get_class_name()] == _X_EXPECTED
        assert got[HubRightCPairConsumer.get_class_name()] == _Y_EXPECTED

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_a_consumer_joining_a_derived_frame_leaves_the_other_consumer_rows_intact(
        self, mode: ParallelizationMode
    ) -> None:
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(
                _star_links(HubRightCArrow, True, HubRightB2),
                HubRightCArrow,
                HubRightCDerivedConsumer,
                mode,
                extra={HubRightB2},
            )

            assert got[HubRightPairConsumer.get_class_name()] == _X_EXPECTED
            assert got[HubRightCDerivedConsumer.get_class_name()] == [2007, 3008]


class TestAConsumerReadingPartOfAStar:
    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    @pytest.mark.parametrize("spoke_c", _SPOKES, ids=_SPOKE_IDS)
    def test_a_consumer_missing_a_join_of_the_star_gets_its_own_rows(
        self, spoke_c: type[FeatureGroup], hub_right: bool, mode: ParallelizationMode
    ) -> None:
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(_star_links(spoke_c, hub_right), spoke_c, HubRightConsumer, mode)

            assert got[HubRightPairConsumer.get_class_name()] == _X_EXPECTED
            assert got[HubRightConsumer.get_class_name()] == _HUB_RIGHT_INNER_EXPECTED

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    @pytest.mark.parametrize("spoke_c", _SPOKES, ids=_SPOKE_IDS)
    def test_a_consumer_missing_a_join_of_the_star_gets_its_own_rows_with_multiprocessing(
        self, spoke_c: type[FeatureGroup], hub_right: bool, flight_server: Any
    ) -> None:
        got = _two_consumers_run(
            _star_links(spoke_c, hub_right),
            spoke_c,
            HubRightConsumer,
            ParallelizationMode.MULTIPROCESSING,
            flight_server,
        )

        assert got[HubRightPairConsumer.get_class_name()] == _X_EXPECTED
        assert got[HubRightConsumer.get_class_name()] == _HUB_RIGHT_INNER_EXPECTED

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    def test_a_consumer_missing_an_outer_join_of_the_star_gets_its_own_rows(
        self, hub_right: bool, mode: ParallelizationMode
    ) -> None:
        a, b, c = (JoinSpec(g, HUB_RIGHT_INDEX) for g in (HubRightA, HubRightB, HubRightCArrow))
        links = {Link.outer(a, b), Link.outer(c, b)} if hub_right else {Link.outer(b, a), Link.outer(b, c)}
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(links, HubRightCArrow, HubRightConsumer, mode, nullable=True)

            pair = got[HubRightPairConsumer.get_class_name()]
            assert [v for v in pair if v is not None] == _X_EXPECTED
            assert pair.count(None) == 3
            assert got[HubRightConsumer.get_class_name()] == _HUB_RIGHT_OUTER_EXPECTED

    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    def test_three_consumers_reading_different_parts_of_a_star_each_get_their_own_rows(self, hub_right: bool) -> None:
        consumers = [HubRightPairConsumer, HubRightConsumer, HubRightCPairConsumer]
        groups: set[type[FeatureGroup]] = {HubRightA, HubRightB, HubRightCArrow, *consumers}

        results = mloda.run_all(
            [Feature(name=consumer.get_class_name()) for consumer in consumers],
            links=_star_links(HubRightCArrow, hub_right),
            compute_frameworks=[PyArrowTable, PandasDataFrame],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
        )

        expected = {
            HubRightPairConsumer: _X_EXPECTED,
            HubRightConsumer: _HUB_RIGHT_INNER_EXPECTED,
            HubRightCPairConsumer: _Y_EXPECTED,
        }
        for consumer, rows in expected.items():
            assert _rows(results, consumer.get_class_name()) == rows

    def test_two_consumers_reading_part_of_a_star_plan_separate_joins_and_a_private_hop(self) -> None:
        links = _star_links(HubRightCArrow, True)
        ab = next(
            link for link in links if link.right_feature_group is HubRightB and link.left_feature_group is HubRightA
        )
        cb = next(link for link in links if link is not ab)
        consumers = [HubRightPairConsumer, HubRightConsumer]
        groups: set[type[FeatureGroup]] = {HubRightA, HubRightB, HubRightCArrow, *consumers}
        session = mloda.prepare(
            [Feature(name=consumer.get_class_name()) for consumer in consumers],
            links=links,
            compute_frameworks=[PyArrowTable],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
        )

        assert session.engine is not None
        plan = session.engine.execution_planner
        joins = [step for step in plan if isinstance(step, JoinStep)]
        ab_tokens = [step.uuid for step in joins if step.link.uuid == ab.uuid]
        cb_tokens = [step.uuid for step in joins if step.link.uuid == cb.uuid]
        consumers_of = {record.token: record.consumers for record in plan.planned_records}

        assert len(ab_tokens) == 2
        assert not consumers_of[ab_tokens[0]] & consumers_of[ab_tokens[1]]
        assert len(cb_tokens) == 1
        hops = [step for step in plan if isinstance(step, TransformFrameworkStep) and step.private_copy]
        assert any(cb_tokens[0] in hop.copy_readers for hop in hops)
        assert not any(cb_tokens[0] in step.children_if_root for step in plan if isinstance(step, FeatureGroupStep))

    @pytest.mark.timeout(60)
    def test_a_consumer_missing_a_join_that_keeps_the_shared_group_gets_its_own_rows_with_multiprocessing(
        self, flight_server: Any
    ) -> None:
        got = _two_consumers_run(
            _star_links(HubRightCArrow, True, c_link=Link.right),
            HubRightCArrow,
            HubRightConsumer,
            ParallelizationMode.MULTIPROCESSING,
            flight_server,
        )

        assert got[HubRightPairConsumer.get_class_name()] == [11, 22, 33]
        assert got[HubRightConsumer.get_class_name()] == [11, 29, 41]

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    @pytest.mark.parametrize("spoke_c", _SPOKES, ids=_SPOKE_IDS)
    def test_a_consumer_missing_a_join_that_keeps_the_shared_group_gets_its_own_rows(
        self, spoke_c: type[FeatureGroup], hub_right: bool, mode: ParallelizationMode
    ) -> None:
        c_link = Link.right if hub_right else Link.left
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(_star_links(spoke_c, hub_right, c_link=c_link), spoke_c, HubRightConsumer, mode)

            assert got[HubRightPairConsumer.get_class_name()] == [11, 22, 33]
            assert got[HubRightConsumer.get_class_name()] == [11, 29, 41]


class OnlyB(FeatureGroup):
    """Reads hr_b only, so it needs no join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), data["hr_b"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TestAStepReadingAJoinSideWithoutALink:
    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    @pytest.mark.parametrize("hub_right", [True, False], ids=["hub_right", "hub_left"])
    @pytest.mark.parametrize("shape", ["one_join", "star"])
    def test_a_step_reading_a_side_frame_keeps_all_its_rows(
        self, shape: str, hub_right: bool, mode: ParallelizationMode
    ) -> None:
        a, b = JoinSpec(HubRightA, HUB_RIGHT_INDEX), JoinSpec(HubRightB, HUB_RIGHT_INDEX)
        consumer: type[FeatureGroup]
        if shape == "one_join":
            links = {Link.inner(a, b) if hub_right else Link.inner(b, a)}
            consumer, expected = HubRightPairConsumer, _X_EXPECTED
        else:
            links = _star_links(HubRightCArrow, hub_right)
            consumer, expected = HubRightConsumer, _HUB_RIGHT_INNER_EXPECTED
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(links, HubRightCArrow, consumer, mode, x_consumer=OnlyB)

            assert got[OnlyB.get_class_name()] == [10, 20, 30, 50, 60]
            assert got[consumer.get_class_name()] == expected


HubRightD = _hub_right_group("HubRightD", "hr_d", {"x": 100, "v": 200, "q": 300}, PyArrowTable)


class HubRightDPairConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hr_d"), Feature("hr_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), pc.add(data["hr_d"], data["hr_b"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TestStarNextToARivalGroup:
    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    @pytest.mark.parametrize("hub_right", [True, False], ids=_ORIENTATION_IDS)
    @pytest.mark.parametrize("spoke_c", _SPOKES, ids=_SPOKE_IDS)
    def test_a_star_and_a_rival_join_of_the_same_group_each_get_their_own_rows(
        self, hub_right: bool, spoke_c: type[FeatureGroup], mode: ParallelizationMode
    ) -> None:
        a, b, c, d = (JoinSpec(g, HUB_RIGHT_INDEX) for g in (HubRightA, HubRightB, spoke_c, HubRightD))
        if hub_right:
            links = {Link.inner(a, b), Link.inner(c, b), Link.inner(d, b)}
        else:
            links = {Link.inner(b, a), Link.inner(b, c), Link.inner(b, d)}
        repeats = _THREADING_REPEATS if mode == ParallelizationMode.THREADING else 1
        for _ in range(repeats):
            got = _two_consumers_run(
                links, spoke_c, HubRightDPairConsumer, mode, x_consumer=HubRightConsumer, extra={HubRightD}
            )

            assert got[HubRightConsumer.get_class_name()] == _HUB_RIGHT_INNER_EXPECTED
            assert got[HubRightDPairConsumer.get_class_name()] == [110, 250]
