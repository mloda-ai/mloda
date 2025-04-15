"""
Utility functions and data creators for combined feature group tests.
"""

from typing import Any, Dict, List

import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


# List of features to test in the combined feature chain
COMBINED_FEATURES: List[Feature | str] = [
    "mean_imputed__price",  # Step 1: Mean imputation
    "sum_7_day_window__mean_imputed__price",  # Step 2: 7-day window sum
    "max_aggr__sum_7_day_window__mean_imputed__price",  # Step 3: Max aggregation
]


class CombinedFeatureTestDataCreator(ATestDataCreator):
    """Base class for combined feature test data creators."""

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary with missing values and time series."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        return {
            "price": [100.0, None, 90.0, 95.0, None, 105.0, 110.0, None, 100.0, 95.0],
            "quantity": [10, 15, 12, 8, 14, 9, 11, 13, 10, 12],
            "category": ["A", "B", None, "A", "B", None, "A", "B", "A", None],
            DefaultOptionKeys.reference_time.value: dates,
        }


class PandasCombinedFeatureTestDataCreator(CombinedFeatureTestDataCreator):
    compute_framework = PandasDataframe


class PyArrowCombinedFeatureTestDataCreator(CombinedFeatureTestDataCreator):
    compute_framework = PyarrowTable


def validate_combined_features(result: List[Any]) -> None:
    """
    Validate the results of the combined feature test.

    Args:
        result: List of DataFrames or Tables from the mlodaAPI.run_all call

    Raises:
        AssertionError: If validation fails
    """
    # Verify we have the expected number of results (at least one)
    assert len(result) >= 1, "Expected at least one result"

    # Convert all results to pandas DataFrames for consistent validation
    dfs = []
    for res in result:
        if hasattr(res, "to_pandas"):
            dfs.append(res.to_pandas())
        else:
            dfs.append(res)

    # Find the DataFrame with the final feature
    final_feature = "max_aggr__sum_7_day_window__mean_imputed__price"
    final_df = None

    for df in dfs:
        if final_feature in df.columns:
            final_df = df
            break

    assert final_df is not None, f"DataFrame with {final_feature} not found"

    # Verify all features in the chain exist
    for feature in COMBINED_FEATURES:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name if isinstance(feature, Feature) else feature
        # Find the DataFrame containing this feature
        feature_df = None
        for df in dfs:
            if feature_name in df.columns:
                feature_df = df
                break
        assert feature_df is not None, f"Feature '{feature_name}' not found in any result"

    # Verify the intermediate features have expected properties

    # Find the DataFrame with mean_imputed__price
    imputed_df = None
    for df in dfs:
        if "mean_imputed__price" in df.columns:
            imputed_df = df
            break

    assert imputed_df is not None, "DataFrame with mean_imputed__price not found"

    # Verify mean imputation worked (no missing values)
    assert not imputed_df["mean_imputed__price"].isna().any(), "mean_imputed__price should not have missing values"

    # Calculate the expected mean value for imputation
    # Original price data: [100.0, None, 90.0, 95.0, None, 105.0, 110.0, None, 100.0, 95.0]
    # Mean of non-missing values: (100.0 + 90.0 + 95.0 + 105.0 + 110.0 + 100.0 + 95.0) / 7 = 695.0 / 7 ≈ 99.29
    expected_mean = 99.29

    # Verify the imputed values are correct
    # The original missing values at indices 1, 4, and 7 should be replaced with the mean
    assert abs(imputed_df["mean_imputed__price"].iloc[1] - expected_mean) < 0.1, (
        f"Expected mean value at index 1 to be {expected_mean}"
    )
    assert abs(imputed_df["mean_imputed__price"].iloc[4] - expected_mean) < 0.1, (
        f"Expected mean value at index 4 to be {expected_mean}"
    )
    assert abs(imputed_df["mean_imputed__price"].iloc[7] - expected_mean) < 0.1, (
        f"Expected mean value at index 7 to be {expected_mean}"
    )

    # Verify the non-missing values are preserved
    assert abs(imputed_df["mean_imputed__price"].iloc[0] - 100.0) < 0.1, "Original value at index 0 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[2] - 90.0) < 0.1, "Original value at index 2 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[3] - 95.0) < 0.1, "Original value at index 3 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[5] - 105.0) < 0.1, "Original value at index 5 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[6] - 110.0) < 0.1, "Original value at index 6 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[8] - 100.0) < 0.1, "Original value at index 8 should be preserved"
    assert abs(imputed_df["mean_imputed__price"].iloc[9] - 95.0) < 0.1, "Original value at index 9 should be preserved"

    # Find the DataFrame with the time window feature
    window_df = None
    for df in dfs:
        if "sum_7_day_window__mean_imputed__price" in df.columns:
            window_df = df
            break

    assert window_df is not None, "DataFrame with sum_7_day_window__mean_imputed__price not found"

    # Calculate expected 7-day window sums
    # After imputation: [100.0, 99.29, 90.0, 95.0, 99.29, 105.0, 110.0, 99.29, 100.0, 95.0]
    expected_window_sums = [
        100.0,  # Day 1: 100.0
        100.0 + 99.29,  # Day 2: 199.29
        100.0 + 99.29 + 90.0,  # Day 3: 289.29
        100.0 + 99.29 + 90.0 + 95.0,  # Day 4: 384.29
        100.0 + 99.29 + 90.0 + 95.0 + 99.29,  # Day 5: 483.58
        100.0 + 99.29 + 90.0 + 95.0 + 99.29 + 105.0,  # Day 6: 588.58
        100.0 + 99.29 + 90.0 + 95.0 + 99.29 + 105.0 + 110.0,  # Day 7: 698.58
        99.29 + 90.0 + 95.0 + 99.29 + 105.0 + 110.0 + 99.29,  # Day 8: 697.87
        90.0 + 95.0 + 99.29 + 105.0 + 110.0 + 99.29 + 100.0,  # Day 9: 698.58
        95.0 + 99.29 + 105.0 + 110.0 + 99.29 + 100.0 + 95.0,  # Day 10: 703.58
    ]

    # Verify the window sums are correct (with a small tolerance for floating point differences)
    for i, expected_sum in enumerate(expected_window_sums):
        assert abs(window_df["sum_7_day_window__mean_imputed__price"].iloc[i] - expected_sum) < 0.5, (
            f"Expected window sum at index {i} to be {expected_sum}"
        )

    # Verify the final aggregated feature has the expected value
    # Maximum value from the window sums: 703.58 (from Day 10)
    expected_max = 703.58
    assert abs(final_df[final_feature].iloc[0] - expected_max) < 0.5, f"Expected max aggregation to be {expected_max}"
