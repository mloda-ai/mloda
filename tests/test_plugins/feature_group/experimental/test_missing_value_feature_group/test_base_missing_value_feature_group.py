import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


class TestMissingValueFeatureGroup:
    """Tests for the MissingValueFeatureGroup class."""

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Test valid feature names
        feature_name = "mean_imputed__income"

        # Test that the PREFIX_PATTERN is correctly defined
        assert hasattr(MissingValueFeatureGroup, "PREFIX_PATTERN")

        # Test that FeatureChainParser methods work with the PREFIX_PATTERN
        assert FeatureChainParser.validate_feature_name(feature_name, MissingValueFeatureGroup.PREFIX_PATTERN)
        assert FeatureChainParser.get_prefix_part(feature_name, MissingValueFeatureGroup.PREFIX_PATTERN) == "mean"
        assert (
            FeatureChainParser.extract_source_feature(feature_name, MissingValueFeatureGroup.PREFIX_PATTERN) == "income"
        )

        # Test invalid feature names
        assert not FeatureChainParser.validate_feature_name(
            "invalid_feature_name", MissingValueFeatureGroup.PREFIX_PATTERN
        )
        assert not FeatureChainParser.validate_feature_name(
            "mean_filled_income", MissingValueFeatureGroup.PREFIX_PATTERN
        )

    def test_get_imputation_method(self) -> None:
        """Test extraction of imputation method from feature name."""
        assert MissingValueFeatureGroup.get_imputation_method("mean_imputed__income") == "mean"
        assert MissingValueFeatureGroup.get_imputation_method("median_imputed__age") == "median"
        assert MissingValueFeatureGroup.get_imputation_method("mode_imputed__category") == "mode"
        assert MissingValueFeatureGroup.get_imputation_method("constant_imputed__status") == "constant"
        assert MissingValueFeatureGroup.get_imputation_method("ffill_imputed__temperature") == "ffill"
        assert MissingValueFeatureGroup.get_imputation_method("bfill_imputed__humidity") == "bfill"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            MissingValueFeatureGroup.get_imputation_method("invalid_feature_name")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert MissingValueFeatureGroup.match_feature_group_criteria("mean_imputed__income", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("median_imputed__age", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("mode_imputed__category", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("constant_imputed__status", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("ffill_imputed__temperature", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("bfill_imputed__humidity", options)

        # Test with FeatureName objects
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("mean_imputed__income"), options)
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("median_imputed__age"), options)

        # Test with invalid feature names
        assert not MissingValueFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("mean_filled_income", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("unknown_imputed_income", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = MissingValueFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("mean_imputed__income"))
        assert input_features == {Feature("income")}

        input_features = feature_group.input_features(options, FeatureName("median_imputed__age"))
        assert input_features == {Feature("age")}

        input_features = feature_group.input_features(options, FeatureName("mode_imputed__category"))
        assert input_features == {Feature("category")}

        input_features = feature_group.input_features(options, FeatureName("constant_imputed__status"))
        assert input_features == {Feature("status")}

        input_features = feature_group.input_features(options, FeatureName("ffill_imputed__temperature"))
        assert input_features == {Feature("temperature")}

        input_features = feature_group.input_features(options, FeatureName("bfill_imputed__humidity"))
        assert input_features == {Feature("humidity")}
