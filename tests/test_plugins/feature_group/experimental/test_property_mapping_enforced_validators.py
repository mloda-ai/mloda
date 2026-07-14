"""Enforcement tests for the six PROPERTY_MAPPING keys that declare a value constraint (issue #724).

Each key must reject bad values on both match paths:
- config-based path: the element_validator raises, match_feature_group_criteria returns False,
  and _strict_validation_rejection_reason surfaces the discarded message.
- string-named path: the PREFIX_PATTERN match short-circuits property-mapping validation, so only
  the match_guard runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.user import Options
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup


@dataclass(frozen=True)
class KeyCase:
    """One PROPERTY_MAPPING key under test, with both match paths pre-wired."""

    feature_group: type[FeatureChainParserMixin]
    key: str
    string_feature_name: str
    base_context: dict[str, Any]
    valid_value: Any
    invalid_values: tuple[Any, ...]
    list_value: list[Any]
    digit_string: str | None = field(default=None)

    def options(self, value: Any) -> Options:
        """Options for the string-named path: only the key under test."""
        return Options(context={self.key: value})

    def config_options(self, value: Any) -> Options:
        """Options for the config-based path: the required option set plus the key under test."""
        return Options(context={**self.base_context, self.key: value})


CLUSTERING_CONTEXT: dict[str, Any] = {
    ClusteringFeatureGroup.ALGORITHM: "kmeans",
    ClusteringFeatureGroup.K_VALUE: 5,
    DefaultOptionKeys.in_features: "customer_behavior",
}

FORECASTING_CONTEXT: dict[str, Any] = {
    ForecastingFeatureGroup.ALGORITHM: "linear",
    ForecastingFeatureGroup.HORIZON: 7,
    ForecastingFeatureGroup.TIME_UNIT: "day",
    DefaultOptionKeys.in_features: "sales",
}

DIMENSION_CONTEXT: dict[str, Any] = {
    DimensionalityReductionFeatureGroup.ALGORITHM: "tsne",
    DimensionalityReductionFeatureGroup.DIMENSION: 2,
    DefaultOptionKeys.in_features: "feature0,feature1,feature2",
}

BOOL_INVALID: tuple[Any, ...] = ("maybe", 1)
NUMERIC_INVALID: tuple[Any, ...] = (0, -5, "abc", 2.5)

CASES: list[KeyCase] = [
    KeyCase(
        feature_group=ClusteringFeatureGroup,
        key=ClusteringFeatureGroup.OUTPUT_PROBABILITIES,
        string_feature_name="customer_behavior__cluster_kmeans_5",
        base_context=CLUSTERING_CONTEXT,
        valid_value=True,
        invalid_values=BOOL_INVALID,
        list_value=[True],
    ),
    KeyCase(
        feature_group=ForecastingFeatureGroup,
        key=ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS,
        string_feature_name="sales__linear_forecast_7day",
        base_context=FORECASTING_CONTEXT,
        valid_value=True,
        invalid_values=BOOL_INVALID,
        list_value=[True],
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.TSNE_MAX_ITER,
        string_feature_name="feature0,feature1,feature2__tsne_2d",
        base_context=DIMENSION_CONTEXT,
        valid_value=250,
        invalid_values=NUMERIC_INVALID,
        list_value=[250, 300],
        digit_string="250",
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.TSNE_N_ITER_WITHOUT_PROGRESS,
        string_feature_name="feature0,feature1,feature2__tsne_2d",
        base_context=DIMENSION_CONTEXT,
        valid_value=30,
        invalid_values=NUMERIC_INVALID,
        list_value=[30, 50],
        digit_string="250",
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.ICA_MAX_ITER,
        string_feature_name="feature0,feature1,feature2__ica_2d",
        base_context={**DIMENSION_CONTEXT, DimensionalityReductionFeatureGroup.ALGORITHM: "ica"},
        valid_value=300,
        invalid_values=NUMERIC_INVALID,
        list_value=[200, 300],
        digit_string="250",
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS,
        string_feature_name="feature0,feature1,feature2__isomap_2d",
        base_context={**DIMENSION_CONTEXT, DimensionalityReductionFeatureGroup.ALGORITHM: "isomap"},
        valid_value=3,
        invalid_values=NUMERIC_INVALID,
        list_value=[3, 5],
        digit_string="250",
    ),
]

NUMERIC_CASES: list[KeyCase] = [case for case in CASES if case.digit_string is not None]


def _case_params() -> list[Any]:
    """One param per key."""
    return [pytest.param(case, id=case.key) for case in CASES]


def _numeric_case_params() -> list[Any]:
    """One param per numeric key."""
    return [pytest.param(case, id=case.key) for case in NUMERIC_CASES]


def _invalid_value_params() -> list[Any]:
    """One param per (key, invalid value) pair."""
    return [
        pytest.param(case, invalid_value, id=f"{case.key}-{invalid_value!r}")
        for case in CASES
        for invalid_value in case.invalid_values
    ]


@pytest.mark.parametrize("case, invalid_value", _invalid_value_params())
def test_string_named_path_rejects_invalid_value(case: KeyCase, invalid_value: Any) -> None:
    """The match_guard must reject a bad value even when the feature name matches PREFIX_PATTERN."""
    assert (
        case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(invalid_value)) is False
    )


@pytest.mark.parametrize("case", _case_params())
def test_string_named_path_accepts_valid_value(case: KeyCase) -> None:
    """Regression guard: existing valid string-named usages must keep matching."""
    assert case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(case.valid_value))


@pytest.mark.parametrize("case, invalid_value", _invalid_value_params())
def test_config_based_path_rejects_invalid_value(case: KeyCase, invalid_value: Any) -> None:
    """The element_validator must reject a bad value on the config-based path."""
    options = case.config_options(invalid_value)
    assert case.feature_group.match_feature_group_criteria("placeholder", options) is False


@pytest.mark.parametrize("case, invalid_value", _invalid_value_params())
def test_config_based_path_reports_rejection_reason(case: KeyCase, invalid_value: Any) -> None:
    """The discarded ValueError must name the property key and the rejected value."""
    options = case.config_options(invalid_value)
    reason = case.feature_group._strict_validation_rejection_reason("placeholder", options)
    assert reason is not None
    assert case.key in reason
    assert str(invalid_value) in reason


@pytest.mark.parametrize("case", _case_params())
def test_config_based_path_accepts_valid_value(case: KeyCase) -> None:
    """Regression guard: the valid config-based option set must keep matching."""
    assert case.feature_group.match_feature_group_criteria("placeholder", case.config_options(case.valid_value))


@pytest.mark.parametrize("case", _numeric_case_params())
def test_digit_string_is_accepted_on_string_named_path(case: KeyCase) -> None:
    """The numeric keys accept a digit string, like the sibling DIMENSION key."""
    assert case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(case.digit_string))


@pytest.mark.parametrize("case", _numeric_case_params())
def test_digit_string_is_accepted_on_config_based_path(case: KeyCase) -> None:
    """The numeric keys accept a digit string on the config-based path too."""
    assert case.feature_group.match_feature_group_criteria("placeholder", case.config_options(case.digit_string))


@pytest.mark.parametrize("case", _case_params())
def test_string_named_path_rejects_list_value(case: KeyCase) -> None:
    """The match_guard checks the whole raw value, so a list is rejected even if its elements are valid."""
    assert (
        case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(case.list_value))
        is False
    )
