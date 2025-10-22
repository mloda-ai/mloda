"""
Test helper for mloda integration tests.
"""

from typing import Any, Dict, List, Optional, Union

from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.api.request import mlodaAPI


class TestHelperApiData(BaseApiData):
    """API data class for test helper integration tests."""

    @classmethod
    def column_names(cls) -> List[str]:
        return ["SomeSimpleFeature"]


class MlodaTestHelper:
    """Helper class for setting up mloda test environments."""

    def create_plugin_collector(self) -> PluginLoader:
        """
        Create and return a PluginLoader with compute_framework plugins loaded.

        Returns:
            PluginLoader: A PluginLoader instance with compute_framework plugins loaded.
        """
        plugin_loader = PluginLoader()
        plugin_loader.load_group("compute_framework")
        return plugin_loader

    def run_integration_test(self, config: Dict[str, Any], data: Dict[str, List[Any]]) -> List[Any]:
        """
        Run integration test with mlodaAPI.run_all().

        Args:
            config: Configuration dict containing 'features' key with list of Feature objects
            data: Test data in dict format (column_name -> list of values)

        Returns:
            List of results from mlodaAPI.run_all()
        """
        plugin_loader = PluginLoader()
        plugin_loader.load_group("compute_framework")
        plugin_loader.load_group("feature_group")

        api_input_data_collection = ApiInputDataCollection(registry={"TestData": TestHelperApiData})
        api_data = {"TestData": {"SomeSimpleFeature": data.get("value1", [])}}

        return mlodaAPI.run_all(
            features=config["features"],
            compute_frameworks=["PyarrowTable"],
            api_input_data_collection=api_input_data_collection,
            api_data=api_data
        )

    def find_result_with_column(self, results: List[Any], column_name: str) -> Optional[Any]:
        """
        Find the first result that contains the specified column.

        Args:
            results: List of DataFrames or Tables
            column_name: Name of the column to search for

        Returns:
            First result containing the column, or None if not found
        """
        for result in results:
            if hasattr(result, "columns") and column_name in result.columns:
                return result
            if hasattr(result, "schema") and hasattr(result.schema, "names") and column_name in result.schema.names:
                return result
        return None

    def assert_result_found(self, results: List[Any], column_name: str) -> None:
        """
        Assert that a result with the specified column exists.

        Args:
            results: List of DataFrames or Tables
            column_name: Name of the column to search for

        Raises:
            AssertionError: If no result contains the specified column
        """
        result = self.find_result_with_column(results, column_name)
        if result is None:
            raise AssertionError(f"No result found containing column '{column_name}'")

    def count_results(self, results: List[Any]) -> int:
        """
        Count the number of results.

        Args:
            results: List of results

        Returns:
            Number of results in the list
        """
        return len(results)
