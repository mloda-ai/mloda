"""Enforcement tests for the six PROPERTY_MAPPING keys that declare a value constraint (issue #724).

Each key must reject bad values on both match paths. Value validation does not depend on how the
feature was created: the element_validator runs on the config-based path AND on the string-named
one, match_feature_group_criteria returns False, and _strict_validation_rejection_reason surfaces
the discarded message. The match_guard additionally judges the raw, un-unpacked value.

A rejection on either path must be REPORTABLE: _strict_validation_rejection_reason is what
identify_feature_group turns into the user-facing hint, so a match_guard rejection may not stay a
silent logger.debug. The reporting must not fire for a feature group that does not match the
feature at all, and a rejected VALUE must not be misdiagnosed as an input-feature forwarding
problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
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
    reported_invalid_value: Any
    list_value: list[Any]
    digit_string: str | None = None

    def options(self, value: Any) -> Options:
        """Options for the string-named path: only the key under test."""
        return Options(context={self.key: value})

    def config_options(self, value: Any) -> Options:
        """Options for the config-based path: the required option set plus the key under test."""
        return Options(context={**self.base_context, self.key: value})

    def as_feature_group(self) -> type[FeatureGroup]:
        """The same class seen through the FeatureGroup API (every case subclasses both)."""
        return cast(type[FeatureGroup], self.feature_group)


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

# A value the user-facing error must echo back. Picked so it cannot occur by accident inside a
# feature name ("0" would also match the "feature0" in the dimensionality-reduction names).
BOOL_REPORTED: Any = "maybe"
NUMERIC_REPORTED: Any = -5

# str.isdigit() is True for the unicode superscript two, but int("²") raises ValueError.
SUPERSCRIPT_TWO = "²"

# Matched by none of the feature groups under test: no PREFIX_PATTERN hit, and the required
# config-based keys (algorithm, ...) are absent.
UNRELATED_FEATURE_NAME = "an_unrelated_feature_of_no_group"

CASES: list[KeyCase] = [
    KeyCase(
        feature_group=ClusteringFeatureGroup,
        key=ClusteringFeatureGroup.OUTPUT_PROBABILITIES,
        string_feature_name="customer_behavior__cluster_kmeans_5",
        base_context=CLUSTERING_CONTEXT,
        valid_value=True,
        invalid_values=BOOL_INVALID,
        reported_invalid_value=BOOL_REPORTED,
        list_value=[True],
    ),
    KeyCase(
        feature_group=ForecastingFeatureGroup,
        key=ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS,
        string_feature_name="sales__linear_forecast_7day",
        base_context=FORECASTING_CONTEXT,
        valid_value=True,
        invalid_values=BOOL_INVALID,
        reported_invalid_value=BOOL_REPORTED,
        list_value=[True],
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.TSNE_MAX_ITER,
        string_feature_name="feature0,feature1,feature2__tsne_2d",
        base_context=DIMENSION_CONTEXT,
        valid_value=250,
        invalid_values=NUMERIC_INVALID,
        reported_invalid_value=NUMERIC_REPORTED,
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
        reported_invalid_value=NUMERIC_REPORTED,
        list_value=[30, 50],
        digit_string="30",
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.ICA_MAX_ITER,
        string_feature_name="feature0,feature1,feature2__ica_2d",
        base_context={**DIMENSION_CONTEXT, DimensionalityReductionFeatureGroup.ALGORITHM: "ica"},
        valid_value=300,
        invalid_values=NUMERIC_INVALID,
        reported_invalid_value=NUMERIC_REPORTED,
        list_value=[200, 300],
        digit_string="300",
    ),
    KeyCase(
        feature_group=DimensionalityReductionFeatureGroup,
        key=DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS,
        string_feature_name="feature0,feature1,feature2__isomap_2d",
        base_context={**DIMENSION_CONTEXT, DimensionalityReductionFeatureGroup.ALGORITHM: "isomap"},
        valid_value=3,
        invalid_values=NUMERIC_INVALID,
        reported_invalid_value=NUMERIC_REPORTED,
        list_value=[3, 5],
        digit_string="3",
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


def _resolution_error(feature: Feature, feature_group: type[FeatureGroup]) -> str:
    """Resolve the feature against a single accessible feature group and return the raised error."""
    accessible_plugins: FeatureGroupEnvironmentMapping = {feature_group: {PandasDataFrame}}
    with pytest.raises(ValueError) as exc_info:
        IdentifyFeatureGroupClass(feature=feature, accessible_plugins=accessible_plugins, links=None)
    return str(exc_info.value)


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


class TestMatchGuardRejectionIsReportable:
    """A match_guard rejection must be recoverable as a message, not only a logger.debug line."""

    @pytest.mark.parametrize("case, invalid_value", _invalid_value_params())
    def test_string_named_path_reports_guard_rejection(self, case: KeyCase, invalid_value: Any) -> None:
        """On the string-named path the guard is the only enforcement, so it must report itself."""
        reason = case.feature_group._strict_validation_rejection_reason(
            case.string_feature_name, case.options(invalid_value)
        )
        assert reason is not None
        assert case.key in reason
        assert str(invalid_value) in reason

    @pytest.mark.parametrize("case", _case_params())
    def test_string_named_path_reports_nothing_for_valid_value(self, case: KeyCase) -> None:
        """No false positives: a value the guard accepts has nothing to report."""
        reason = case.feature_group._strict_validation_rejection_reason(
            case.string_feature_name, case.options(case.valid_value)
        )
        assert reason is None

    @pytest.mark.parametrize("case", _case_params())
    def test_string_named_path_reports_nothing_when_key_is_absent(self, case: KeyCase) -> None:
        """No false positives: an absent key is not a rejection."""
        assert case.feature_group._strict_validation_rejection_reason(case.string_feature_name, Options()) is None

    @pytest.mark.parametrize("case", _case_params())
    def test_no_guard_rejection_reported_for_unrelated_feature_name(self, case: KeyCase) -> None:
        """A feature group that does not match the name at all must not report its guards.

        The list value passes every element_validator and is rejected only by the match_guard, so a
        guard check that ignores whether the feature group matches would report a phantom rejection.
        """
        options = case.options(case.list_value)
        assert case.feature_group.match_feature_group_criteria(UNRELATED_FEATURE_NAME, options) is False
        assert case.feature_group._strict_validation_rejection_reason(UNRELATED_FEATURE_NAME, options) is None

    def test_no_guard_rejection_reported_for_option_outside_the_mapping(self) -> None:
        """An aggregated feature carrying another group's option has nothing to report."""
        options = Options(context={ClusteringFeatureGroup.OUTPUT_PROBABILITIES: "maybe"})
        assert AggregatedFeatureGroup._strict_validation_rejection_reason("sales__sum_aggr", options) is None


class TestRejectedValueNamedInResolutionError:
    """The user-facing no-feature-group error must name the culprit option, not stay generic."""

    @pytest.mark.parametrize("case", _case_params())
    def test_string_named_rejection_names_key_and_value(self, case: KeyCase) -> None:
        """A string-named feature rejected by a guard must not fail with a bare 'No feature groups found'."""
        feature = Feature(case.string_feature_name, case.options(case.reported_invalid_value))

        message = _resolution_error(feature, case.as_feature_group())

        assert case.key in message
        assert str(case.reported_invalid_value) in message
        assert case.feature_group.__name__ in message

    @pytest.mark.parametrize("case", _case_params())
    def test_config_based_rejection_names_key_and_value(self, case: KeyCase) -> None:
        """Regression guard: the config-based path already names the culprit option."""
        feature = Feature("placeholder", case.config_options(case.reported_invalid_value))

        message = _resolution_error(feature, case.as_feature_group())

        assert case.key in message
        assert str(case.reported_invalid_value) in message
        assert case.feature_group.__name__ in message


class TestRejectedValueIsNotAForwardingProblem:
    """A rejected VALUE in group options must not be reported as an extra-group-option problem.

    Since #791/#782 the extra-group-option hint does not exist at all: the pure renderer never
    speculatively re-matches. These assertions therefore pin the message's shape, not a branch choice.
    """

    def test_group_placed_bad_value_names_the_value_not_forwarding(self) -> None:
        """ica_max_iter=-1 in group options is a bad value; forward_group_exclude would not fix it."""
        feature = Feature(
            "feature0,feature1,feature2__ica_2d",
            Options({DimensionalityReductionFeatureGroup.ICA_MAX_ITER: -1}),
        )

        message = _resolution_error(feature, DimensionalityReductionFeatureGroup)

        assert DimensionalityReductionFeatureGroup.ICA_MAX_ITER in message
        assert "-1" in message
        assert "extra group option" not in message
        assert "forward_group_exclude" not in message
        assert "Group options flow onto input features" not in message

    def test_group_placed_valid_value_still_matches(self) -> None:
        """Regression guard: a valid value in group options is not a rejection at all."""
        options = Options({DimensionalityReductionFeatureGroup.ICA_MAX_ITER: 300})
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "feature0,feature1,feature2__ica_2d", options
        )


class TestNumericPredicateHardening:
    """The positive-int predicate behind the four numeric keys must be type-honest."""

    @pytest.mark.parametrize("case", _numeric_case_params())
    def test_numpy_integer_is_accepted_on_both_paths(self, case: KeyCase) -> None:
        """A numpy integer is a positive integer; rejecting it on isinstance(int) is a false negative."""
        np = pytest.importorskip("numpy")
        value = np.int64(int(case.valid_value))

        assert case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(value))
        assert case.feature_group.match_feature_group_criteria("placeholder", case.config_options(value))

    def test_numpy_integer_is_accepted_for_the_dimension_key(self) -> None:
        """DIMENSION shares the predicate with the numeric keys, so it must accept a numpy integer too."""
        np = pytest.importorskip("numpy")
        options = Options(
            context={**DIMENSION_CONTEXT, DimensionalityReductionFeatureGroup.DIMENSION: np.int64(2)},
        )

        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("placeholder", options)

    @pytest.mark.parametrize("case", _numeric_case_params())
    def test_bool_is_rejected_on_both_paths(self, case: KeyCase) -> None:
        """True is not an iteration count: bool must not be read as 1."""
        assert case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(True)) is False
        assert case.feature_group.match_feature_group_criteria("placeholder", case.config_options(True)) is False

    @pytest.mark.parametrize("case", _numeric_case_params())
    def test_superscript_digit_is_rejected_cleanly_on_string_named_path(self, case: KeyCase) -> None:
        """The superscript passes str.isdigit() but breaks int(): the rejection must stay clean."""
        options = case.options(SUPERSCRIPT_TWO)

        assert case.feature_group.match_feature_group_criteria(case.string_feature_name, options) is False

        reason = case.feature_group._strict_validation_rejection_reason(case.string_feature_name, options)
        assert reason is not None
        assert case.key in reason
        assert "invalid literal for int()" not in reason

    @pytest.mark.parametrize("case", _numeric_case_params())
    def test_superscript_digit_is_rejected_cleanly_on_config_based_path(self, case: KeyCase) -> None:
        """The config-based rejection reason must name the key, not leak int()'s internal error."""
        options = case.config_options(SUPERSCRIPT_TWO)

        assert case.feature_group.match_feature_group_criteria("placeholder", options) is False

        reason = case.feature_group._strict_validation_rejection_reason("placeholder", options)
        assert reason is not None
        assert case.key in reason
        assert "invalid literal for int()" not in reason


class TestStrictValidationDoesNotMakeKeysRequired:
    """strict_validation=True enforces the VALUE SPACE, never presence."""

    @pytest.mark.parametrize("case", _case_params())
    def test_absent_key_still_matches_on_string_named_path(self, case: KeyCase) -> None:
        """Regression guard: the key stays optional on the string-named path."""
        assert case.feature_group.match_feature_group_criteria(case.string_feature_name, Options())

    @pytest.mark.parametrize("case", _case_params())
    def test_absent_key_still_matches_on_config_based_path(self, case: KeyCase) -> None:
        """Regression guard: the key stays optional on the config-based path."""
        options = Options(context=dict(case.base_context))
        assert case.feature_group.match_feature_group_criteria("placeholder", options)

    @pytest.mark.parametrize("case", _case_params())
    def test_explicit_none_is_treated_as_absent_on_string_named_path(self, case: KeyCase) -> None:
        """options.get cannot tell an explicit None from an absent key, so None still matches."""
        assert case.feature_group.match_feature_group_criteria(case.string_feature_name, case.options(None))

    @pytest.mark.parametrize("case", _case_params())
    def test_explicit_none_is_treated_as_absent_on_config_based_path(self, case: KeyCase) -> None:
        """Same Options semantics on the config-based path."""
        assert case.feature_group.match_feature_group_criteria("placeholder", case.config_options(None))


class TestConfigBasedListRejection:
    """A list value survives the element_validator (it sees the elements) and dies on the match_guard."""

    @pytest.mark.parametrize("case", _case_params())
    def test_config_based_path_rejects_list_value(self, case: KeyCase) -> None:
        """Regression guard: the raw list is not a valid value even when every element is."""
        options = case.config_options(case.list_value)
        assert case.feature_group.match_feature_group_criteria("placeholder", options) is False

    @pytest.mark.parametrize("case", _case_params())
    def test_config_based_path_reports_list_rejection(self, case: KeyCase) -> None:
        """The guard-only rejection must be reportable, like the element_validator one already is."""
        options = case.config_options(case.list_value)

        reason = case.feature_group._strict_validation_rejection_reason("placeholder", options)
        assert reason is not None
        assert case.key in reason
        assert str(case.list_value) in reason
