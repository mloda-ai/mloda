"""
Utility functions and data creators for missing value tests.
"""

from typing import Any, Dict, List

import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


# List of missing value features to test
MISSING_VALUE_FEATURES: List[Feature | str] = [
    "income__mean_imputed",  # Mean imputation
    "age__median_imputed",  # Median imputation
    "category__mode_imputed",  # Mode imputation
    "category__constant_imputed",  # Constant imputation
    "temperature__ffill_imputed",  # Forward fill imputation
]


class MissingValueTestDataCreator(ATestDataCreator):
    """Base class for missing value test data creators."""

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "income": [50000, None, 75000, None, 60000],
            "age": [30, 25, None, 45, None],
            "category": ["A", None, "B", "A", None],
            "temperature": [72.5, 68.3, None, None, 70.1],
            "group": ["X", "Y", "X", "Y", "X"],
        }


class PandasMissingValueTestDataCreator(MissingValueTestDataCreator):
    compute_framework = PandasDataFrame


class PyArrowMissingValueTestDataCreator(MissingValueTestDataCreator):
    compute_framework = PyArrowTable


def validate_missing_value_features(result: List[pd.DataFrame]) -> None:
    # Verify the results
    assert len(result) == 2, "Expected two results: one for source data, one for imputed features"

    # Find the DataFrame with the imputed features
    imputed_df = None
    first_feature = MISSING_VALUE_FEATURES[0]
    first_feature_name = first_feature.name if isinstance(first_feature, Feature) else first_feature

    for df in result:
        if first_feature_name in df.columns:
            imputed_df = df
            break

    assert imputed_df is not None, "DataFrame with imputed features not found"

    # Verify all expected features exist
    for feature in MISSING_VALUE_FEATURES:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name if isinstance(feature, Feature) else feature
        assert feature_name in imputed_df.columns, f"Expected feature '{feature_name}' not found"

    # Verify that missing values are imputed
    assert imputed_df["category__constant_imputed"].iloc[1] == "Unknown"
    assert imputed_df["category__constant_imputed"].iloc[4] == "Unknown"

    assert not pd.isna(imputed_df["temperature__ffill_imputed"].iloc[2])
    assert not pd.isna(imputed_df["temperature__ffill_imputed"].iloc[3])

    assert "income__mean_imputed" in imputed_df.columns
    assert abs(imputed_df["income__mean_imputed"].iloc[1] - 61666.67) < 1.0  # Increased tolerance for PyArrow
    assert abs(imputed_df["income__mean_imputed"].iloc[3] - 61666.67) < 1.0  # Increased tolerance for PyArrow

    assert "age__median_imputed" in imputed_df.columns
    assert imputed_df["age__median_imputed"].iloc[2] == 30
    assert imputed_df["age__median_imputed"].iloc[4] == 30

    assert "category__mode_imputed" in imputed_df.columns
    assert imputed_df["category__mode_imputed"].iloc[1] == "A"
    assert imputed_df["category__mode_imputed"].iloc[4] == "A"
