"""
Test simplified API data parameter for mlodaAPI.run_all().

These tests verify that run_all() accepts api_data as a simple dict.
The ApiInputDataCollection is created automatically from the dict structure.

Simplified API:
    result = mlodaAPI.run_all(
        features,
        api_data={"KeyName": {"col1": [1, 2], "col2": ["a", "b"]}}
    )
"""

from typing import Any, List, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature


# ============================================================================
# Test Feature: Simple API Consumer
# ============================================================================
class SimpleApiFeature(AbstractFeatureGroup):
    """A simple feature that consumes API data."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features from API data."""
        return {
            Feature(name="api_id", index=Index(("api_id",))),
            Feature(name="api_value", index=Index(("api_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Concatenate API columns."""
        data["SimpleApiFeature"] = data["api_id"].astype(str) + "_" + data["api_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Feature: Multi-key API Consumer
# ============================================================================
class MultiKeyApiFeature(AbstractFeatureGroup):
    """A feature that consumes data from multiple API keys."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features from multiple API data sources."""
        # This will require data from both FirstKey and SecondKey
        return {
            Feature(name="first_id", index=Index(("first_id",))),
            Feature(name="second_value", index=Index(("second_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Combine data from multiple keys."""
        data["MultiKeyApiFeature"] = data["first_id"].astype(str) + "_" + data["second_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Feature: Joined with Creator
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


class SimplifiedApiJoinFeature(AbstractFeatureGroup):
    """Joins simplified API data with Creator data using LEFT join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Define input features with a LEFT join link."""
        # Create the link: LEFT join on api_id = creator_id
        link = Link.left((ApiInputDataFeature, Index(("api_id",))), (CreatorDataFeature, Index(("creator_id",))))

        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the joined data
        data["SimplifiedApiJoinFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Class
# ============================================================================
class TestSimplifiedApiData:
    """Tests for simplified api_data parameter in mlodaAPI.run_all()."""

    _enabled_simple = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            SimpleApiFeature,
        }
    )

    _enabled_multikey = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            MultiKeyApiFeature,
        }
    )

    _enabled_join = PlugInCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            CreatorDataFeature,
            SimplifiedApiJoinFeature,
        }
    )

    def test_simplified_api_data_single_key(self) -> None:
        """
        Test that run_all() accepts simplified api_data dict with a single key.

        This test verifies the NEW simplified API:
            api_data={"KeyName": {"col1": [...], "col2": [...]}}

        Instead of the OLD verbose API:
            api_input_data_collection = ApiInputDataCollection()
            api_data = api_input_data_collection.setup_key_api_data(...)
            run_all(..., api_input_data_collection=..., api_data=...)

        Expected behavior:
        - run_all() should auto-create ApiInputDataCollection internally
        - api_data dict keys become the collection key names
        - api_data dict values contain the actual column data
        """
        # NEW simplified API - just pass a dict!
        api_data = {
            "ApiExample": {
                "api_id": [1, 2, 3, 4],
                "api_value": ["w", "x", "y", "z"],
            }
        }

        feature_list: List[Union[Feature, str]] = [Feature(name="SimpleApiFeature")]

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled_simple,
            compute_frameworks={PandasDataframe},
            api_data=api_data,  # No api_input_data_collection needed!
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "SimpleApiFeature" in df.columns
        assert len(df) == 4
        assert df["SimpleApiFeature"].tolist() == ["1_w", "2_x", "3_y", "4_z"]

    def test_simplified_api_data_multiple_keys(self) -> None:
        """
        Test that run_all() accepts simplified api_data dict with multiple keys.

        This verifies that multiple API data sources can be registered in a single dict.
        The test uses only one key's data - combining data from multiple keys requires Links.
        """
        api_data = {
            "FirstKey": {
                "api_id": [1, 2],
                "api_value": ["a", "b"],
            },
            "SecondKey": {
                "second_id": [3, 4],
                "second_value": ["c", "d"],
            },
        }

        # Use SimpleApiFeature which only needs api_id and api_value (from FirstKey)
        feature_list: List[Union[Feature, str]] = [Feature(name="SimpleApiFeature")]

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled_simple,
            compute_frameworks={PandasDataframe},
            api_data=api_data,
        )

        # Validate result - should work even with extra unused keys
        assert len(result) == 1
        df = result[0]
        assert "SimpleApiFeature" in df.columns
        assert len(df) == 2
        assert df["SimpleApiFeature"].tolist() == ["1_a", "2_b"]

    def test_simplified_api_data_two_keys_first_key_used(self) -> None:
        """
        Test that run_all() can register multiple API keys and use the first key's data.

        This verifies that:
        - Multiple API keys can be registered in a single api_data dict
        - The first key's data is accessible via its column names
        - Extra unused keys don't break the registration
        """
        api_data = {
            "FirstKey": {
                "first_id": [1, 2, 3],
                "first_value": ["a", "b", "c"],
            },
            "SecondKey": {
                "second_id": [10, 20],
                "second_value": ["x", "y"],
            },
        }

        # Request feature from first key
        feature_list: List[Union[Feature, str]] = ["first_value"]

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PandasDataframe},
            api_data=api_data,
        )

        assert len(result) == 1
        df = result[0]
        assert "first_value" in df.columns
        assert len(df) == 3
        assert df["first_value"].tolist() == ["a", "b", "c"]

    def test_simplified_api_data_two_keys_second_key_used(self) -> None:
        """
        Test that run_all() can register multiple API keys and use the second key's data.

        This verifies that:
        - Multiple API keys can be registered in a single api_data dict
        - The second key's data is accessible via its column names
        - Extra unused keys don't break the registration
        """
        api_data = {
            "FirstKey": {
                "first_id": [1, 2, 3],
                "first_value": ["a", "b", "c"],
            },
            "SecondKey": {
                "second_id": [10, 20],
                "second_value": ["x", "y"],
            },
        }

        # Request feature from second key
        feature_list: List[Union[Feature, str]] = ["second_value"]

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PandasDataframe},
            api_data=api_data,
        )

        assert len(result) == 1
        df = result[0]
        assert "second_value" in df.columns
        assert len(df) == 2
        assert df["second_value"].tolist() == ["x", "y"]

    def test_simplified_api_data_with_joins(self) -> None:
        """
        Test that simplified api_data works with Links/Joins.

        This ensures the simplified API is compatible with existing
        Link functionality (LEFT join, APPEND, etc).
        """
        api_data = {
            "ApiExample": {
                "api_id": [1, 2, 3, 4],
                "api_value": ["w", "x", "y", "z"],
            }
        }

        feature_list: List[Union[Feature, str]] = [Feature(name="SimplifiedApiJoinFeature")]

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=self._enabled_join,
            compute_frameworks={PandasDataframe},
            api_data=api_data,
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "SimplifiedApiJoinFeature" in df.columns
        assert len(df) == 4  # LEFT join keeps all API rows

    def test_none_api_data_still_works(self) -> None:
        """
        Test that passing None for api_data still works (no regression).

        This ensures backward compatibility for features that don't use API data.
        """
        feature_list: List[Union[Feature, str]] = [Feature(name="creator_id")]

        _enabled_creator_only = PlugInCollector.enabled_feature_groups({CreatorDataFeature})

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=_enabled_creator_only,
            compute_frameworks={PandasDataframe},
            api_data=None,  # Explicitly None
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "creator_id" in df.columns
        assert len(df) == 3

    def test_empty_api_data_dict(self) -> None:
        """
        Test that passing an empty api_data dict behaves correctly.

        Should either work (no API data) or raise a clear error.
        """
        feature_list: List[Union[Feature, str]] = [Feature(name="creator_id")]

        _enabled_creator_only = PlugInCollector.enabled_feature_groups({CreatorDataFeature})

        result = mlodaAPI.run_all(
            feature_list,
            plugin_collector=_enabled_creator_only,
            compute_frameworks={PandasDataframe},
            api_data={},  # Empty dict
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "creator_id" in df.columns
        assert len(df) == 3
