"""
Utility functions and data creators for aggregation tests.
"""

from typing import Any, Dict, List

import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


AGGREGATION_FEATURES: List[Feature | str] = [
    "sum_aggr__sales",  # Sum of sales
    "avg_aggr__price",  # Average price
    "min_aggr__discount",  # Minimum discount
    "max_aggr__customer_rating",  # Maximum customer rating
]


class AggregatedTestDataCreator(ATestDataCreator):
    """Base class for aggregation test data creators."""

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "sales": [100, 200, 300, 400, 500],
            "quantity": [10, 20, 30, 40, 50],
            "price": [10.0, 9.5, 9.0, 8.5, 8.0],
            "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
            "customer_rating": [4, 5, 3, 4, 5],
        }


class PandasAggregatedTestDataCreator(AggregatedTestDataCreator):
    compute_framework = PandasDataframe


class PyArrowAggregatedTestDataCreator(AggregatedTestDataCreator):
    compute_framework = PyarrowTable


def validate_aggregated_features(result: List[pd.DataFrame]) -> None:
    # Verify the results
    assert len(result) == 2, "Expected two results: one for source data, one for aggregated features"

    # Find the DataFrame with the aggregated features
    agg_df = None
    first_feature = AGGREGATION_FEATURES[0]
    first_feature_name = first_feature.name if isinstance(first_feature, Feature) else first_feature

    for df in result:
        if first_feature_name in df.columns:
            agg_df = df
            break

    assert agg_df is not None, "DataFrame with aggregated features not found"

    # Verify all expected features exist
    for feature in AGGREGATION_FEATURES:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name if isinstance(feature, Feature) else feature
        assert feature_name in agg_df.columns, f"Expected feature '{feature_name}' not found"

    # Verify specific aggregation values
    assert agg_df["sum_aggr__sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]
    assert agg_df["avg_aggr__price"].iloc[0] == 9.0  # Average of [10.0, 9.5, 9.0, 8.5, 8.0]
    assert agg_df["min_aggr__discount"].iloc[0] == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]
    assert agg_df["max_aggr__customer_rating"].iloc[0] == 5  # Max of [4, 5, 3, 4, 5]
