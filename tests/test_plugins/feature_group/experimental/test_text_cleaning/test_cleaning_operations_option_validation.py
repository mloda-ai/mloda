"""A value the element_validator cannot accept is a non-match, whether the validator returns falsy or raises.

``TextCleaningFeatureGroup`` ships ``element_validator: lambda op: op in SUPPORTED_OPERATIONS``, a membership
test against a dict, so an unhashable element makes the validator itself raise TypeError.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaningFeatureGroup


CLEANED_FEATURE = "text__cleaned_text"

# Unhashable: `{"a": 1} in SUPPORTED_OPERATIONS` raises TypeError instead of returning False.
UNHASHABLE_OPERATION: dict[str, int] = {"a": 1}


def _name_path_options(operations: object) -> Options:
    return Options(context={TextCleaningFeatureGroup.CLEANING_OPERATIONS: operations})


def _config_path_options(operations: object) -> Options:
    return Options(
        context={
            TextCleaningFeatureGroup.CLEANING_OPERATIONS: operations,
            DefaultOptionKeys.in_features: "text",
        }
    )


class TestUnhashableOperationIsANonMatch:
    def test_name_path_unhashable_operation_returns_false(self) -> None:
        """The validator raises TypeError on the name path; the verdict is a non-match."""
        result = TextCleaningFeatureGroup.match_feature_group_criteria(
            CLEANED_FEATURE, _name_path_options([UNHASHABLE_OPERATION])
        )

        assert result is False

    def test_config_path_unhashable_operation_returns_false(self) -> None:
        """The config-based path reaches the same verdict for the same value."""
        result = TextCleaningFeatureGroup.match_feature_group_criteria(
            "placeholder", _config_path_options([UNHASHABLE_OPERATION])
        )

        assert result is False

    def test_rejection_reason_names_key_and_value_on_name_path(self) -> None:
        """The reason builder reports the rejection instead of raising the validator's TypeError."""
        reason = TextCleaningFeatureGroup._strict_validation_rejection_reason(
            CLEANED_FEATURE, _name_path_options([UNHASHABLE_OPERATION])
        )

        assert reason is not None
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS in reason
        assert str(UNHASHABLE_OPERATION) in reason

    def test_rejection_reason_names_key_and_value_on_config_path(self) -> None:
        reason = TextCleaningFeatureGroup._strict_validation_rejection_reason(
            "placeholder", _config_path_options([UNHASHABLE_OPERATION])
        )

        assert reason is not None
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS in reason
        assert str(UNHASHABLE_OPERATION) in reason


class TestOrdinaryOperationVerdictsUnchanged:
    """The validator keeps accepting what it accepted and rejecting what it could already judge."""

    def test_valid_operations_on_name_path_match(self) -> None:
        result = TextCleaningFeatureGroup.match_feature_group_criteria(
            CLEANED_FEATURE, _name_path_options(["normalize", "remove_urls"])
        )

        assert result is True

    def test_valid_operations_on_config_path_match(self) -> None:
        result = TextCleaningFeatureGroup.match_feature_group_criteria(
            "placeholder", _config_path_options(["normalize"])
        )

        assert result is True

    def test_name_path_without_options_is_a_non_match(self) -> None:
        """cleaning_operations is option-required on the name path, so its absence is a non-match."""
        assert TextCleaningFeatureGroup.match_feature_group_criteria(CLEANED_FEATURE, Options()) is False

    def test_hashable_unsupported_operation_is_a_non_match(self) -> None:
        """A value the validator can judge (and rejects) keeps its verdict."""
        result = TextCleaningFeatureGroup.match_feature_group_criteria(CLEANED_FEATURE, _name_path_options(["bogus"]))

        assert result is False


class TestUnhashableOperationAtEngineLevel:
    """Nothing escapes into the engine."""

    def test_engine_reports_no_feature_groups_found_with_the_reason(self) -> None:
        feature = Feature(CLEANED_FEATURE, _name_path_options([UNHASHABLE_OPERATION]))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PandasTextCleaningFeatureGroup: {PandasDataFrame},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        message = str(exc_info.value)
        assert "No feature groups found" in message, (
            f"A validator that raises must be a non-match with mloda's standard error, "
            f"not a TypeError out of the engine, but got: {message}"
        )
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS in message
        assert str(UNHASHABLE_OPERATION) in message
