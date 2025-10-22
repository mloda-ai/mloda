"""
Unit tests for MlodaTestHelper class.
"""

from typing import Any, Dict, List

import pandas as pd
import pytest
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.abstract_plugins.components.feature import Feature
from tests.test_plugins.feature_group.test_tooling.integration.mloda_test_helper import MlodaTestHelper
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator


class TestMlodaTestHelper:
    """Test MlodaTestHelper class."""

    def test_create_plugin_collector(self) -> None:
        """Test creating a PluginLoader instance."""
        helper = MlodaTestHelper()
        collector = helper.create_plugin_collector()

        assert collector is not None
        assert isinstance(collector, PluginLoader)

    def test_run_integration_test_basic(self) -> None:
        """Test running a basic integration test with mlodaAPI.run_all()."""
        helper = MlodaTestHelper()

        # Create simple test data
        data: Dict[str, List[Any]] = DataGenerator.generate_data(
            n_rows=10,
            numeric_cols=["value1", "value2"],
            categorical_cols=["category"],
        )

        # Create minimal mloda config with a simple feature group
        config = {
            "features": [Feature("SomeSimpleFeature")],
        }

        # Run integration test
        result = helper.run_integration_test(config, data)

        # Assert result is a list and not empty
        assert isinstance(result, list)
        assert len(result) > 0

    def test_find_result_with_column_found(self) -> None:
        """Test finding a result DataFrame that contains a specific column."""
        helper = MlodaTestHelper()

        # Create mock results list with different DataFrames
        results: List[pd.DataFrame] = [
            pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]}),
            pd.DataFrame({"col_c": [7, 8, 9], "target_column": [10, 11, 12]}),
            pd.DataFrame({"col_d": [13, 14, 15], "col_e": [16, 17, 18]}),
        ]

        # Find result containing target_column
        result = helper.find_result_with_column(results, "target_column")

        # Assert result is not None
        assert result is not None

        # Assert returned DataFrame has the target column
        assert "target_column" in result.columns

    def test_assert_result_found_success(self) -> None:
        """Test assert_result_found does not raise when result exists."""
        helper = MlodaTestHelper()

        # Create mock results list with target column
        results: List[pd.DataFrame] = [
            pd.DataFrame({"col_a": [1, 2, 3]}),
            pd.DataFrame({"target_column": [4, 5, 6]}),
        ]

        # Should not raise any exception
        helper.assert_result_found(results, "target_column")

    def test_count_results(self) -> None:
        """Test counting the number of results."""
        helper = MlodaTestHelper()

        # Create a list of 3 mock DataFrames
        results: List[pd.DataFrame] = [
            pd.DataFrame({"col_a": [1, 2, 3]}),
            pd.DataFrame({"col_b": [4, 5, 6]}),
            pd.DataFrame({"col_c": [7, 8, 9]}),
        ]

        # Count results
        count = helper.count_results(results)

        # Assert count equals 3
        assert count == 3
