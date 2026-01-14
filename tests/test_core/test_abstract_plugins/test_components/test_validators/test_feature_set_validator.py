"""
Tests for FeatureSetValidator class.

This validator extracts validation logic from FeatureSet class.
Tests verify:
- Options initialization validation
- Equal options across features validation
- Feature added validation
- Filters not set validation
- Filters type validation
"""

from typing import Set
import pytest

from mloda.user import Feature
from mloda.user import Options
from mloda.user import SingleFilter
from mloda.provider import FeatureSetValidator


class TestValidateOptionsInitialized:
    """Tests for validate_options_initialized method."""

    def test_valid_options_passes(self) -> None:
        """Valid Options object should not raise an error."""
        options = Options(group={"key": "value"})

        # Should not raise
        FeatureSetValidator.validate_options_initialized(options)

    def test_none_options_raises_value_error(self) -> None:
        """None options should raise ValueError."""
        with pytest.raises(ValueError):
            FeatureSetValidator.validate_options_initialized(None)

    def test_error_message_includes_context(self) -> None:
        """Error message should include the provided context string."""
        with pytest.raises(ValueError, match="custom_context"):
            FeatureSetValidator.validate_options_initialized(None, context="custom_context")


class TestValidateEqualOptions:
    """Tests for validate_equal_options method."""

    def test_all_same_options_passes(self) -> None:
        """All features with identical options should not raise."""
        options = Options(group={"key": "value"})
        features: Set[Feature] = {
            Feature("feature1", options),
            Feature("feature2", options),
            Feature("feature3", options),
        }

        # Should not raise
        FeatureSetValidator.validate_equal_options(features)

    def test_different_options_raises_value_error(self) -> None:
        """Features with different options should raise ValueError."""
        options1 = Options(group={"key": "value1"})
        options2 = Options(group={"key": "value2"})
        features: Set[Feature] = {
            Feature("feature1", options1),
            Feature("feature2", options2),
        }

        with pytest.raises(ValueError):
            FeatureSetValidator.validate_equal_options(features)

    def test_empty_features_passes(self) -> None:
        """Empty feature set should not raise."""
        features: Set[Feature] = set()

        # Should not raise
        FeatureSetValidator.validate_equal_options(features)

    def test_single_feature_passes(self) -> None:
        """Single feature should not raise."""
        options = Options(group={"key": "value"})
        features: Set[Feature] = {Feature("feature1", options)}

        # Should not raise
        FeatureSetValidator.validate_equal_options(features)


class TestValidateFeatureAdded:
    """Tests for validate_feature_added method."""

    def test_feature_name_present_passes(self) -> None:
        """Non-None feature name should not raise."""
        feature_name = "some_feature"

        # Should not raise
        FeatureSetValidator.validate_feature_added(feature_name)

    def test_none_raises_value_error(self) -> None:
        """None feature name should raise ValueError."""
        with pytest.raises(ValueError):
            FeatureSetValidator.validate_feature_added(None)

    def test_error_message_includes_context(self) -> None:
        """Error message should include the provided context string."""
        with pytest.raises(ValueError, match="custom_feature"):
            FeatureSetValidator.validate_feature_added(None, context="custom_feature")


class TestValidateFiltersNotSet:
    """Tests for validate_filters_not_set method."""

    def test_none_filters_passes(self) -> None:
        """None filters should not raise (filters not yet set)."""
        # Should not raise
        FeatureSetValidator.validate_filters_not_set(None)

    def test_filters_already_set_raises(self) -> None:
        """Non-None filters should raise ValueError."""
        filter1 = SingleFilter("feature1", "range", {"min": 0, "max": 100})
        filters: Set[SingleFilter] = {filter1}

        with pytest.raises(ValueError):
            FeatureSetValidator.validate_filters_not_set(filters)


class TestValidateFiltersIsSetType:
    """Tests for validate_filters_is_set_type method."""

    def test_set_type_passes(self) -> None:
        """Set type should not raise."""
        filter1 = SingleFilter("feature1", "range", {"min": 0, "max": 100})
        filters: Set[SingleFilter] = {filter1}

        # Should not raise
        FeatureSetValidator.validate_filters_is_set_type(filters)

    def test_list_raises_value_error(self) -> None:
        """List type should raise ValueError."""
        filter1 = SingleFilter("feature1", "range", {"min": 0, "max": 100})
        filters = [filter1]  # List instead of Set

        with pytest.raises(ValueError):
            FeatureSetValidator.validate_filters_is_set_type(filters)

    def test_dict_raises_value_error(self) -> None:
        """Dict type should raise ValueError."""
        filters = {"filter1": "value"}  # Dict instead of Set

        with pytest.raises(ValueError):
            FeatureSetValidator.validate_filters_is_set_type(filters)
