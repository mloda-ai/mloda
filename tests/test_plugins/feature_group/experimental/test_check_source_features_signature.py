"""
Tests that all experimental feature group plugins use the unified
_check_source_features_exist(cls, data, feature_names: List[str]) signature.

This validates issue #296: source feature check methods should have a single,
consistent signature across all plugins.
"""

import inspect
from typing import Any

import pytest

from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import (
    PandasAggregatedFeatureGroup,
)
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import (
    PandasMissingValueFeatureGroup,
)
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import (
    PandasSklearnPipelineFeatureGroup,
)
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaningFeatureGroup

try:
    from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup

    HAS_TIME_WINDOW_PANDAS = True
except ImportError:
    HAS_TIME_WINDOW_PANDAS = False

try:
    from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup

    HAS_FORECASTING_PANDAS = True
except ImportError:
    HAS_FORECASTING_PANDAS = False


import pandas as pd


ALL_PANDAS_CLASSES = [
    PandasAggregatedFeatureGroup,
    PandasClusteringFeatureGroup,
    PandasMissingValueFeatureGroup,
    PandasDimensionalityReductionFeatureGroup,
    PandasGeoDistanceFeatureGroup,
    PandasNodeCentralityFeatureGroup,
    PandasEncodingFeatureGroup,
    PandasSklearnPipelineFeatureGroup,
    PandasScalingFeatureGroup,
    PandasTextCleaningFeatureGroup,
]

if HAS_TIME_WINDOW_PANDAS:
    ALL_PANDAS_CLASSES.append(PandasTimeWindowFeatureGroup)

if HAS_FORECASTING_PANDAS:
    ALL_PANDAS_CLASSES.append(PandasForecastingFeatureGroup)


@pytest.mark.parametrize("cls", ALL_PANDAS_CLASSES, ids=lambda c: c.__name__)
class TestUnifiedCheckSourceFeaturesSignature:
    """Verify every concrete plugin exposes _check_source_features_exist(cls, data, feature_names: List[str])."""

    def test_method_exists(self, cls: Any) -> None:
        """Every plugin must define _check_source_features_exist."""
        assert hasattr(cls, "_check_source_features_exist"), f"{cls.__name__} is missing _check_source_features_exist"

    def test_method_signature_has_three_params(self, cls: Any) -> None:
        """Signature must be (cls, data, feature_names)."""
        sig = inspect.signature(cls._check_source_features_exist)
        params = list(sig.parameters.keys())
        assert len(params) == 2, (
            f"{cls.__name__}._check_source_features_exist has params {params}, expected (data, feature_names)"
        )
        assert params[1] == "feature_names", (
            f"{cls.__name__}._check_source_features_exist second param is '{params[1]}', expected 'feature_names'"
        )

    def test_accepts_list_of_strings(self, cls: Any) -> None:
        """Method must accept a List[str] as feature_names."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        # Should not raise for existing features
        cls._check_source_features_exist(df, ["col_a"])

    def test_raises_on_missing_feature(self, cls: Any) -> None:
        """Method must raise ValueError when a feature is missing."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError):
            cls._check_source_features_exist(df, ["nonexistent_column"])

    def test_old_singular_method_absent(self, cls: Any) -> None:
        """Old singular _check_source_feature_exists must not be present."""
        assert not hasattr(cls, "_check_source_feature_exists"), (
            f"{cls.__name__} still has the old singular _check_source_feature_exists method"
        )

    def test_old_point_method_absent(self, cls: Any) -> None:
        """Old _check_point_features_exist must not be present."""
        assert not hasattr(cls, "_check_point_features_exist"), (
            f"{cls.__name__} still has the old _check_point_features_exist method"
        )
