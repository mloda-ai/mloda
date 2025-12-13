import pytest
from typing import Any, List, Optional, Set

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


class ConcreteMissingValueFeatureGroup(MissingValueFeatureGroup):
    """Minimal concrete implementation for testing base class methods."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> Set[str]:
        return set()

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        pass

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        return data

    @classmethod
    def _perform_imputation(
        cls,
        data: Any,
        imputation_method: str,
        in_features: List[str],
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> Any:
        return data


class TestMissingValueFeatureGroup:
    """Tests for the MissingValueFeatureGroup class."""

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Test valid feature names
        feature_name = "income__mean_imputed"

        # Test that the PREFIX_PATTERN is correctly defined
        assert hasattr(MissingValueFeatureGroup, "PREFIX_PATTERN")

        # Test that FeatureChainParser methods work with the PREFIX_PATTERN
        assert FeatureChainParser.extract_in_feature(feature_name, MissingValueFeatureGroup.PREFIX_PATTERN) == "income"

    def test_get_imputation_method(self) -> None:
        """Test extraction of imputation method from feature name."""
        assert MissingValueFeatureGroup.get_imputation_method("income__mean_imputed") == "mean"
        assert MissingValueFeatureGroup.get_imputation_method("age__median_imputed") == "median"
        assert MissingValueFeatureGroup.get_imputation_method("category__mode_imputed") == "mode"
        assert MissingValueFeatureGroup.get_imputation_method("status__constant_imputed") == "constant"
        assert MissingValueFeatureGroup.get_imputation_method("temperature__ffill_imputed") == "ffill"
        assert MissingValueFeatureGroup.get_imputation_method("humidity__bfill_imputed") == "bfill"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            MissingValueFeatureGroup.get_imputation_method("invalid_feature_name")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert MissingValueFeatureGroup.match_feature_group_criteria("income__mean_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("age__median_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("category__mode_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("status__constant_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("temperature__ffill_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("humidity__bfill_imputed", options)

        # Test with FeatureName objects
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("income__mean_imputed"), options)
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("age__median_imputed"), options)

        # Test with invalid feature names
        assert not MissingValueFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("mean_filled_income", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("unknown_imputed_income", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = ConcreteMissingValueFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("income__mean_imputed"))
        assert input_features == {Feature("income")}

        input_features = feature_group.input_features(options, FeatureName("age__median_imputed"))
        assert input_features == {Feature("age")}

        input_features = feature_group.input_features(options, FeatureName("category__mode_imputed"))
        assert input_features == {Feature("category")}

        input_features = feature_group.input_features(options, FeatureName("status__constant_imputed"))
        assert input_features == {Feature("status")}

        input_features = feature_group.input_features(options, FeatureName("temperature__ffill_imputed"))
        assert input_features == {Feature("temperature")}

        input_features = feature_group.input_features(options, FeatureName("humidity__bfill_imputed"))
        assert input_features == {Feature("humidity")}
