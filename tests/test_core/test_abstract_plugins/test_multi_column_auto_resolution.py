"""
Integration test demonstrating multi-column resolution with chaining.

This test shows the complete flow of:
1. Producer creates multi-column output using apply_naming_convention()
2. Consumer auto-discovers columns using resolve_multi_column_feature()
3. Chained processor works with consumer's single-column output
"""

from typing import Any, Dict, List, Optional, Set, Type, Union

import numpy as np
import pandas as pd
import pytest

from mloda import FeatureGroup
from mloda import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda import Options
from mloda import ComputeFramework
from mloda.user import PluginLoader
from mloda.user import PluginCollector
from mloda import API
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class MultiColumnTestDataCreator(FeatureGroup):
    """Test data creator providing source data for multi-column tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        """Return a DataCreator with the supported feature names."""
        return DataCreator({"source_data"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Create test data with 5 rows."""
        return pd.DataFrame(
            {
                "source_data": [10, 20, 30, 40, 50],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        """Return the compute framework for this data creator."""
        return {PandasDataFrame}


class MultiColumnProducer(FeatureGroup):
    """
    Producer feature group that creates multi-column output.

    This demonstrates the producer side of multi-column features.
    It takes a source feature and produces 3 columns using apply_naming_convention():
    - base_feature~0
    - base_feature~1
    - base_feature~2
    """

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        """Explicitly support base_feature."""
        return {"base_feature"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Requires source_data as input."""
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Create multi-column output using apply_naming_convention().

        Produces a 2D array with 3 columns, which will be automatically
        split into base_feature~0, base_feature~1, base_feature~2
        """
        # Get the feature name we're computing
        feature_name = next(iter(features.get_all_names()))

        # Extract source data
        source_values = data["source_data"].values

        # Create 3-column output (simulating a transformation like PCA or OneHot encoding)
        # Column 0: original value
        # Column 1: value * 2
        # Column 2: value * 3
        result = np.column_stack([source_values, source_values * 2, source_values * 3])

        # Apply naming convention to create ~0, ~1, ~2 columns
        named_columns = cls.apply_naming_convention(result, feature_name)

        # Add columns to dataframe
        for col_name, col_data in named_columns.items():
            data[col_name] = col_data

        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        """Support Pandas framework."""
        return {PandasDataFrame}


class MultiColumnConsumer(FeatureGroup):
    """
    Consumer feature group that auto-discovers multi-column features.

    This demonstrates the consumer side of multi-column features.
    It uses resolve_multi_column_feature() to discover all columns
    matching the pattern base_feature~* without manually specifying them.
    """

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        """Explicitly support consumed_feature."""
        return {"consumed_feature"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Requires base_feature as input (without ~N suffix).

        The resolve_multi_column_feature() method will discover the actual
        columns (base_feature~0, ~1, ~2) at runtime.
        """
        return {Feature("base_feature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Use resolve_multi_column_feature() to discover and process all columns.

        This demonstrates automatic discovery of multi-column features
        without needing to know the exact number of columns ahead of time.

        KEY POINT: "base_feature" column doesn't exist in data - only base_feature~0, ~1, ~2 exist.
        resolve_multi_column_feature() discovers all matching columns automatically.
        """
        # Get the feature name we're computing
        feature_name = next(iter(features.get_all_names()))

        # Discover all columns matching "base_feature~*" pattern
        # This is the key integration: resolve_multi_column_feature() finds all ~N columns
        discovered_columns = cls.resolve_multi_column_feature("base_feature", set(data.columns))

        # Verify that discover actually found the multi-column features
        assert len(discovered_columns) == 3, (
            f"Expected 3 columns, found {len(discovered_columns)}: {discovered_columns}"
        )
        assert "base_feature~0" in discovered_columns
        assert "base_feature~1" in discovered_columns
        assert "base_feature~2" in discovered_columns

        # Process all discovered columns by summing them
        # This simulates a consumer that aggregates multi-column input
        summed_values = None
        for col_name in discovered_columns:
            if summed_values is None:
                summed_values = data[col_name].values
            else:
                summed_values += data[col_name].values

        # Add the result as a single column
        data[feature_name] = summed_values

        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        """Support Pandas framework."""
        return {PandasDataFrame}


class ChainedProcessor(FeatureGroup):
    """
    Chained feature group that processes consumer's single-column output.

    This demonstrates that chaining continues to work normally after
    multi-column resolution. The consumer produces a single column,
    which this processor can use as input.
    """

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        """Explicitly support chained_feature."""
        return {"chained_feature"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Requires consumed_feature as input."""
        return {Feature("consumed_feature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Process the consumer's output (simple doubling operation).

        This shows that once multi-column features are resolved,
        normal chaining continues to work.
        """
        # Get the feature name we're computing
        feature_name = next(iter(features.get_all_names()))

        # Double the consumed feature value
        data[feature_name] = data["consumed_feature"].values * 2

        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        """Support Pandas framework."""
        return {PandasDataFrame}


def test_multi_column_auto_resolution_with_chaining() -> None:
    """
    Integration test for multi-column resolution with chaining.

    This test validates the complete flow:
    1. MultiColumnProducer creates base_feature~0, ~1, ~2
    2. MultiColumnConsumer discovers all 3 columns automatically
    3. MultiColumnConsumer produces single output column (consumed_feature)
    4. ChainedProcessor uses consumed_feature as input

    Expected behavior:
    - Producer creates 3 columns using apply_naming_convention()
    - Consumer discovers all 3 using resolve_multi_column_feature()
    - Consumer sums them: (value + value*2 + value*3) = value*6
    - Chained processor doubles: value*6*2 = value*12

    For source_data = [10, 20, 30, 40, 50]:
    - base_feature~0 = [10, 20, 30, 40, 50]
    - base_feature~1 = [20, 40, 60, 80, 100]
    - base_feature~2 = [30, 60, 90, 120, 150]
    - consumed_feature = [60, 120, 180, 240, 300] (sum of above)
    - chained_feature = [120, 240, 360, 480, 600] (doubled)
    """
    # Load plugins
    PluginLoader().all()

    # Enable the necessary feature groups
    plugin_collector = PluginCollector.enabled_feature_groups(
        {
            MultiColumnTestDataCreator,
            MultiColumnProducer,
            MultiColumnConsumer,
            ChainedProcessor,
        }
    )

    # Request all features we want to verify in the output
    # This triggers the entire chain:
    # source_data -> base_feature -> consumed_feature -> chained_feature
    # We request all intermediate features to verify the complete flow
    features_to_request: List[Union[Feature, str]] = [
        Feature("base_feature"),  # Multi-column producer output
        Feature("consumed_feature"),  # Consumer output that discovered multi-columns
        Feature("chained_feature"),  # Final chained output
    ]

    # Run the computation
    api = API(
        features_to_request,
        {PandasDataFrame},
        plugin_collector=plugin_collector,
    )
    api._batch_run()
    results = api.get_result()

    # Verify results
    # The API returns separate DataFrames per feature group
    assert len(results) == 3, "Should return 3 DataFrames (one per requested feature)"

    # Combine results into a single DataFrame for easier verification
    # (In real usage, each DataFrame would be used separately)
    df = pd.concat(results, axis=1)

    # Step 1: Verify producer created multi-column output
    assert "base_feature~0" in df.columns, "Producer should create base_feature~0"
    assert "base_feature~1" in df.columns, "Producer should create base_feature~1"
    assert "base_feature~2" in df.columns, "Producer should create base_feature~2"

    # Verify producer column values
    np.testing.assert_array_equal(
        df["base_feature~0"].values,
        np.array([10, 20, 30, 40, 50]),
        err_msg="Producer column ~0 should contain original values",
    )
    np.testing.assert_array_equal(
        df["base_feature~1"].values,
        np.array([20, 40, 60, 80, 100]),
        err_msg="Producer column ~1 should contain values*2",
    )
    np.testing.assert_array_equal(
        df["base_feature~2"].values,
        np.array([30, 60, 90, 120, 150]),
        err_msg="Producer column ~2 should contain values*3",
    )

    # Step 2: Verify consumer discovered all columns and produced single output
    assert "consumed_feature" in df.columns, "Consumer should create consumed_feature"

    # Consumer should sum all 3 columns: value*1 + value*2 + value*3 = value*6
    expected_consumed = np.array([60, 120, 180, 240, 300])
    np.testing.assert_array_equal(
        df["consumed_feature"].values, expected_consumed, err_msg="Consumer should sum all discovered columns"
    )

    # Step 3: Verify chained processor worked with consumer's output
    assert "chained_feature" in df.columns, "Chained processor should create chained_feature"

    # Chained processor should double the consumed value: value*6*2 = value*12
    expected_chained = np.array([120, 240, 360, 480, 600])
    np.testing.assert_array_equal(
        df["chained_feature"].values, expected_chained, err_msg="Chained processor should double consumed values"
    )
