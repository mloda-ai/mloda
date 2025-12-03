"""
Integration test: ApiInputDataCollection with Links and Joins.

Three features:
1. ApiInputDataFeature (existing) - receives API data
2. CreatorDataFeature (custom) - creates own data via DataCreator
3. JoinedFeature (custom) - depends on both with a join
"""

from typing import Any, List, Optional, Set, Union
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import Link, JoinSpec
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature


# ============================================================================
# Feature B: Data Creator Feature
# ============================================================================
class CreatorDataFeature(AbstractFeatureGroup):
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
# Feature C: Joined Feature (depends on API and Creator)
# ============================================================================
class LeftJoinedFeature(AbstractFeatureGroup):
    """Joins API data with Creator data using LEFT join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features with a LEFT join link."""

        # Create the link: LEFT join on api_id = creator_id
        link = Link.left(
            JoinSpec(ApiInputDataFeature, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
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


class AppendedFeature(AbstractFeatureGroup):
    """Appends API data with Creator data (stacks vertically)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features with an APPEND link."""

        # Create the link: APPEND stacks data vertically
        link = Link.append(
            JoinSpec(ApiInputDataFeature, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
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

    _enabled_left = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            CreatorDataFeature,
            LeftJoinedFeature,
        }
    )

    _enabled_append = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            CreatorDataFeature,
            AppendedFeature,
        }
    )

    def test_api_creator_left_join(self) -> None:
        """
        Test LEFT join between API data and DataCreator data.

        API data (4 rows): ids 1, 2, 3, 4
        Creator data (3 rows): ids 1, 2, 3
        Result: 4 rows (LEFT keeps all API rows, id 4 has null for creator_value)
        """
        # Request the joined feature
        feature_list: List[Union[Feature, str]] = [Feature(name="LeftJoinedFeature")]

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled_left,
            compute_frameworks={PandasDataFrame},
            api_data={"ApiExample": {"api_id": [1, 2, 3, 4], "api_value": ["w", "x", "y", "z"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "LeftJoinedFeature" in df.columns
        assert len(df) == 4  # LEFT join keeps all API rows

    def test_api_creator_append(self) -> None:
        """
        Test APPEND between API data and DataCreator data.

        API data (2 rows)
        Creator data (3 rows)
        Result: 5 rows (stacked vertically)
        """
        # Request the appended feature
        feature_list: List[Union[Feature, str]] = [Feature(name="AppendedFeature")]

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled_append,
            compute_frameworks={PandasDataFrame},
            api_data={"ApiExample": {"api_id": [1, 2], "api_value": ["x", "y"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "AppendedFeature" in df.columns
        assert len(df) == 5  # APPEND stacks: 2 API + 3 Creator rows
