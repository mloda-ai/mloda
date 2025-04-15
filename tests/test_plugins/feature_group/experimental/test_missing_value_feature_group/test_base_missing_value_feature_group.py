import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import BaseMissingValueFeatureGroup


class TestBaseMissingValueFeatureGroup:
    """Tests for the BaseMissingValueFeatureGroup class."""

    def test_parse_feature_name(self) -> None:
        """Test parsing of feature name into components."""
        # Test valid feature names
        assert BaseMissingValueFeatureGroup.parse_feature_name("mean_imputed_income") == ("mean", "income")
        assert BaseMissingValueFeatureGroup.parse_feature_name("median_imputed_age") == ("median", "age")
        assert BaseMissingValueFeatureGroup.parse_feature_name("mode_imputed_category") == ("mode", "category")
        assert BaseMissingValueFeatureGroup.parse_feature_name("constant_imputed_status") == ("constant", "status")
        assert BaseMissingValueFeatureGroup.parse_feature_name("ffill_imputed_temperature") == ("ffill", "temperature")
        assert BaseMissingValueFeatureGroup.parse_feature_name("bfill_imputed_humidity") == ("bfill", "humidity")

        # Test invalid feature names
        with pytest.raises(ValueError):
            BaseMissingValueFeatureGroup.parse_feature_name("invalid_feature_name")

        with pytest.raises(ValueError):
            BaseMissingValueFeatureGroup.parse_feature_name("mean_filled_income")

        with pytest.raises(ValueError):
            BaseMissingValueFeatureGroup.parse_feature_name("unknown_imputed_income")

    def test_get_imputation_method(self) -> None:
        """Test extraction of imputation method from feature name."""
        assert BaseMissingValueFeatureGroup.get_imputation_method("mean_imputed_income") == "mean"
        assert BaseMissingValueFeatureGroup.get_imputation_method("median_imputed_age") == "median"
        assert BaseMissingValueFeatureGroup.get_imputation_method("mode_imputed_category") == "mode"
        assert BaseMissingValueFeatureGroup.get_imputation_method("constant_imputed_status") == "constant"
        assert BaseMissingValueFeatureGroup.get_imputation_method("ffill_imputed_temperature") == "ffill"
        assert BaseMissingValueFeatureGroup.get_imputation_method("bfill_imputed_humidity") == "bfill"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            BaseMissingValueFeatureGroup.get_imputation_method("invalid_feature_name")

    def test_mloda_source_feature(self) -> None:
        """Test extraction of source feature from feature name."""
        assert BaseMissingValueFeatureGroup.mloda_source_feature("mean_imputed_income") == "income"
        assert BaseMissingValueFeatureGroup.mloda_source_feature("median_imputed_age") == "age"
        assert BaseMissingValueFeatureGroup.mloda_source_feature("mode_imputed_category") == "category"
        assert BaseMissingValueFeatureGroup.mloda_source_feature("constant_imputed_status") == "status"
        assert BaseMissingValueFeatureGroup.mloda_source_feature("ffill_imputed_temperature") == "temperature"
        assert BaseMissingValueFeatureGroup.mloda_source_feature("bfill_imputed_humidity") == "humidity"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            BaseMissingValueFeatureGroup.mloda_source_feature("invalid_feature_name")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("mean_imputed_income", options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("median_imputed_age", options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("mode_imputed_category", options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("constant_imputed_status", options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("ffill_imputed_temperature", options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria("bfill_imputed_humidity", options)

        # Test with FeatureName objects
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria(FeatureName("mean_imputed_income"), options)
        assert BaseMissingValueFeatureGroup.match_feature_group_criteria(FeatureName("median_imputed_age"), options)

        # Test with invalid feature names
        assert not BaseMissingValueFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not BaseMissingValueFeatureGroup.match_feature_group_criteria("mean_filled_income", options)
        assert not BaseMissingValueFeatureGroup.match_feature_group_criteria("unknown_imputed_income", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = BaseMissingValueFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("mean_imputed_income"))
        assert input_features == {Feature("income")}

        input_features = feature_group.input_features(options, FeatureName("median_imputed_age"))
        assert input_features == {Feature("age")}

        input_features = feature_group.input_features(options, FeatureName("mode_imputed_category"))
        assert input_features == {Feature("category")}

        input_features = feature_group.input_features(options, FeatureName("constant_imputed_status"))
        assert input_features == {Feature("status")}

        input_features = feature_group.input_features(options, FeatureName("ffill_imputed_temperature"))
        assert input_features == {Feature("temperature")}

        input_features = feature_group.input_features(options, FeatureName("bfill_imputed_humidity"))
        assert input_features == {Feature("humidity")}
