"""
Integration test: ApiInputDataCollection with Links and Joins.

Three features:
1. ApiDataFeatureGroup (existing) - receives mloda data
2. CreatorDataFeature (custom) - creates own data via DataCreator
3. JoinedFeature (custom) - depends on both with a join
"""

from typing import Any, List, Optional, Set, Union
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import ApiDataFeatureGroup


# ============================================================================
# Feature B: Data Creator Feature
# ============================================================================
class CreatorDataFeature(FeatureGroup):
    """Creates its own data via DataCreator."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"creator_id", "creator_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"creator_id": [1, 2, 3], "creator_value": ["a", "b", "c"]}

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"creator_id", "creator_value"}


# ============================================================================
# Feature C: Joined Feature (depends on mloda and Creator)
# ============================================================================
class LeftJoinedFeature(FeatureGroup):
    """Joins mloda data with Creator data using LEFT join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features with a LEFT join link."""

        # Create the link: LEFT join on api_id = creator_id
        link = Link.left(
            JoinSpec(ApiDataFeatureGroup, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
        )

        # Return features - attach link to one of them
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the joined data - output column must match feature name
        data["LeftJoinedFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


class AppendedFeature(FeatureGroup):
    """Appends mloda data with Creator data (stacks vertically)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features with an APPEND link."""

        # Create the link: APPEND stacks data vertically
        link = Link.append(
            JoinSpec(ApiDataFeatureGroup, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
        )

        # Return features - attach link to one of them
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the appended data - output column must match feature name
        data["AppendedFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Class
# ============================================================================
class TestApiLinkJoin:
    """Integration tests for ApiInputDataCollection with Links."""

    _enabled_left = PluginCollector.enabled_feature_groups(
        {
            ApiDataFeatureGroup,
            CreatorDataFeature,
            LeftJoinedFeature,
        }
    )

    _enabled_append = PluginCollector.enabled_feature_groups(
        {
            ApiDataFeatureGroup,
            CreatorDataFeature,
            AppendedFeature,
        }
    )

    def test_api_creator_left_join(self) -> None:
        """
        Test LEFT join between mloda data and DataCreator data.

        mloda data (4 rows): ids 1, 2, 3, 4
        Creator data (3 rows): ids 1, 2, 3
        Result: 4 rows (LEFT keeps all mloda rows, id 4 has null for creator_value)
        """
        # Request the joined feature
        feature_list: List[Union[Feature, str]] = [Feature(name="LeftJoinedFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_left,
            compute_frameworks={PandasDataFrame},
            api_data={"ApiExample": {"api_id": [1, 2, 3, 4], "api_value": ["w", "x", "y", "z"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "LeftJoinedFeature" in df.columns
        assert len(df) == 4  # LEFT join keeps all mloda rows

    def test_api_creator_append(self) -> None:
        """
        Test APPEND between mloda data and DataCreator data.

        mloda data (2 rows)
        Creator data (3 rows)
        Result: 5 rows (stacked vertically)
        """
        # Request the appended feature
        feature_list: List[Union[Feature, str]] = [Feature(name="AppendedFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_append,
            compute_frameworks={PandasDataFrame},
            api_data={"ApiExample": {"api_id": [1, 2], "api_value": ["x", "y"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "AppendedFeature" in df.columns
        assert len(df) == 5  # APPEND stacks: 2 mloda + 3 Creator rows
