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


TOLERANT_PANDAS_CLASSES = [
    PandasAggregatedFeatureGroup,
    PandasClusteringFeatureGroup,
]

if HAS_TIME_WINDOW_PANDAS:
    TOLERANT_PANDAS_CLASSES.append(PandasTimeWindowFeatureGroup)

if HAS_FORECASTING_PANDAS:
    TOLERANT_PANDAS_CLASSES.append(PandasForecastingFeatureGroup)

STRICT_PANDAS_CLASSES = [
    PandasMissingValueFeatureGroup,
    PandasEncodingFeatureGroup,
    PandasScalingFeatureGroup,
    PandasSklearnPipelineFeatureGroup,
    PandasDimensionalityReductionFeatureGroup,
    PandasTextCleaningFeatureGroup,
    PandasNodeCentralityFeatureGroup,
    PandasGeoDistanceFeatureGroup,
]


class TestTolerantPluginsAllowPartialPresence:
    """Tolerant plugins raise only when ALL features are missing, not when some are present."""

    @pytest.mark.parametrize("cls", TOLERANT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_no_raise_when_some_features_present(self, cls: Any) -> None:
        """Tolerant plugins must NOT raise when at least one feature exists in the data."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        # col_a exists, nonexistent does not -- tolerant plugins should accept this
        cls._check_source_features_exist(df, ["col_a", "nonexistent"])


class TestStrictPluginsRaiseOnPartialPresence:
    """Strict plugins raise when ANY feature is missing, even if some are present."""

    @pytest.mark.parametrize("cls", STRICT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_raises_when_some_features_missing(self, cls: Any) -> None:
        """Strict plugins must raise ValueError when any feature is missing from data."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError):
            cls._check_source_features_exist(df, ["col_a", "nonexistent"])


class TestStrictPluginsReportAllMissingFeatures:
    """Strict plugins must collect and report ALL missing features, not just the first one."""

    @pytest.mark.parametrize("cls", STRICT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_error_contains_all_missing_features(self, cls: Any) -> None:
        """When multiple features are missing, ALL must appear in the error message."""
        df = pd.DataFrame({"col_a": [1, 2]})
        with pytest.raises(ValueError, match="missing1") as exc_info:
            cls._check_source_features_exist(df, ["missing1", "missing2"])
        error_msg = str(exc_info.value)
        assert "missing1" in error_msg, f"{cls.__name__} error message does not contain 'missing1': {error_msg}"
        assert "missing2" in error_msg, f"{cls.__name__} error message does not contain 'missing2': {error_msg}"


class TestErrorMessagesIncludeAvailableColumns:
    """All plugin error messages must include the available columns for debuggability."""

    @pytest.mark.parametrize("cls", STRICT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_strict_error_includes_available_columns(self, cls: Any) -> None:
        """Strict plugin errors must list available columns so users can diagnose the issue."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError) as exc_info:
            cls._check_source_features_exist(df, ["nonexistent"])
        error_msg = str(exc_info.value)
        assert "col_a" in error_msg, f"{cls.__name__} error does not mention available column 'col_a': {error_msg}"
        assert "col_b" in error_msg, f"{cls.__name__} error does not mention available column 'col_b': {error_msg}"

    @pytest.mark.parametrize("cls", TOLERANT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_tolerant_error_includes_available_columns(self, cls: Any) -> None:
        """Tolerant plugin errors must list available columns so users can diagnose the issue."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError) as exc_info:
            cls._check_source_features_exist(df, ["nonexistent"])
        error_msg = str(exc_info.value)
        assert "col_a" in error_msg, f"{cls.__name__} error does not mention available column 'col_a': {error_msg}"
        assert "col_b" in error_msg, f"{cls.__name__} error does not mention available column 'col_b': {error_msg}"


class TestStrictPluginsUseNormalizedMessageFormat:
    """Strict plugins must use the standardized error message prefix."""

    @pytest.mark.parametrize("cls", STRICT_PANDAS_CLASSES, ids=lambda c: c.__name__)
    def test_error_message_has_normalized_prefix(self, cls: Any) -> None:
        """Error message must start with 'Source features not found in data:' for consistency."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError) as exc_info:
            cls._check_source_features_exist(df, ["nonexistent"])
        error_msg = str(exc_info.value)
        assert "Source features not found in data:" in error_msg, (
            f"{cls.__name__} error does not use normalized prefix. Got: {error_msg}"
        )
