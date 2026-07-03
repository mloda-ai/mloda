"""End-to-end integration coverage for ``Link.star`` through the real mloda engine.

These tests build a genuine star topology (one hub feature group plus multiple
spoke feature groups, all sharing a single ``"row_id"`` index column) and run the
result through ``mloda.run_all`` on the pandas compute framework. Unlike the unit
tests for ``Link.star`` (which only inspect the returned ``Link`` set), these
exercise the joins the engine actually performs and assert on concrete joined
cell values.
"""

from typing import Any, Optional

from mloda.provider import FeatureGroup, FeatureSet, BaseInputData, DataCreator
from mloda.user import Feature, FeatureName, Index, Link, JoinSpec, JoinType, Options, PluginCollector, mloda

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401


# === Fully-matching star topology (all FGs expose row_id [1, 2, 3]) ===


class StarHubFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "hub_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "hub_value": ["h1", "h2", "h3"]}


class StarSpokeAFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "spoke_a_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "spoke_a_value": [10, 20, 30]}


class StarSpokeBFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "spoke_b_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "spoke_b_value": ["x", "y", "z"]}


class StarConsumerFG(FeatureGroup):
    """Downstream consumer that combines the hub column with both spoke columns.

    Requesting all three features forces the engine to materialise the star join
    before this group runs, so the merged frame is what ``calculate_feature`` sees.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "left_hub_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "left_hub_value": ["h1", "h2", "h3"]}


class StarLeftSpokeAFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "left_spoke_a_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "left_spoke_a_value": [10, 20, 30]}


class StarLeftSpokeBFG(FeatureGroup):
    """Spoke B is intentionally missing row_id 3 so INNER drops it and LEFT keeps it."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "left_spoke_b_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "outer_hub_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [1, 2, 3], "outer_hub_value": ["h1", "h2", "h3"]}


class StarOuterSpokeAFG(FeatureGroup):
    """Spoke carries row_id 4 (which the hub lacks) and lacks row_id 1 (which the hub has)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"row_id", "outer_spoke_a_value"})

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("row_id",))]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"row_id": [2, 3, 4], "outer_spoke_a_value": [20, 30, 40]}


class StarOuterConsumerFG(FeatureGroup):
    """Consumer for the outer topology.

    It requests both features (so the join runs) but only passes the join key
    ``row_id`` through, so no possibly-null column is string-concatenated.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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
