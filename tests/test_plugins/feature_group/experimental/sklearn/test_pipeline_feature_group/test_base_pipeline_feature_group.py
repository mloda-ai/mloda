"""
Tests for base SklearnPipelineFeatureGroup.
"""

import pytest
from unittest.mock import Mock, patch
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options


class TestSklearnPipelineFeatureGroup:
    """Test cases for SklearnPipelineFeatureGroup."""

    def test_match_feature_group_criteria_valid_names(self) -> None:
        """Test that valid feature names match the criteria."""
        valid_names = [
            "sklearn_pipeline_preprocessing__raw_features",
            "sklearn_pipeline_scaling__income",
            "sklearn_pipeline_feature_engineering__customer_data",
            "sklearn_pipeline_imputation__missing_data",
        ]

        for name in valid_names:
            assert SklearnPipelineFeatureGroup.match_feature_group_criteria(name, Options({})), (
                f"Feature name '{name}' should match criteria"
            )

    def test_match_feature_group_criteria_invalid_names(self) -> None:
        """Test that invalid feature names don't match the criteria."""
        invalid_names = [
            "sklearn_preprocessing__raw_features",  # Missing 'pipeline'
            "pipeline_preprocessing__raw_features",  # Missing 'sklearn'
            "sklearn_pipeline__raw_features",  # Missing pipeline name
            "sklearn_pipeline_preprocessing",  # Missing source features
            "other_feature_name",  # Completely different pattern
            "",  # Empty string
        ]

        for name in invalid_names:
            assert not SklearnPipelineFeatureGroup.match_feature_group_criteria(name, Options({})), (
                f"Feature name '{name}' should not match criteria"
            )

    def test_get_pipeline_name(self) -> None:
        """Test extraction of pipeline name from feature name."""
        test_cases = [
            ("sklearn_pipeline_preprocessing__raw_features", "preprocessing"),
            ("sklearn_pipeline_scaling__income", "scaling"),
            ("sklearn_pipeline_feature_engineering__customer_data", "feature_engineering"),
            ("sklearn_pipeline_imputation__missing_data", "imputation"),
        ]

        for feature_name, expected_pipeline_name in test_cases:
            result = SklearnPipelineFeatureGroup.get_pipeline_name(feature_name)
            assert result == expected_pipeline_name

    def test_get_pipeline_name_invalid(self) -> None:
        """Test that invalid feature names raise ValueError."""
        invalid_names = [
            "sklearn_preprocessing__raw_features",
            "invalid_feature_name",
            "",
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid sklearn pipeline feature name format"):
                SklearnPipelineFeatureGroup.get_pipeline_name(name)

    def test_input_features_single_source(self) -> None:
        """Test input_features method with single source feature."""
        feature_name = FeatureName("sklearn_pipeline_preprocessing__raw_features")
        options = Options({})

        result = SklearnPipelineFeatureGroup().input_features(options, feature_name)

        assert result is not None
        assert len(result) == 1
        feature_names = [f.name for f in result]
        assert "raw_features" in feature_names

    def test_input_features_multiple_sources(self) -> None:
        """Test input_features method with multiple source features."""
        feature_name = FeatureName("sklearn_pipeline_scaling__income,age,salary")
        options = Options({})

        result = SklearnPipelineFeatureGroup().input_features(options, feature_name)

        assert result is not None
        assert len(result) == 3
        feature_names = [f.name for f in result]
        assert "income" in feature_names
        assert "age" in feature_names
        assert "salary" in feature_names

    def test_create_default_pipeline_config_preprocessing(self) -> None:
        """Test default pipeline configuration for preprocessing."""
        # Skip test if sklearn not available
        try:
            SklearnPipelineFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        config = SklearnPipelineFeatureGroup._create_default_pipeline_config("preprocessing")

        assert "steps" in config
        assert "params" in config
        assert len(config["steps"]) == 2

        # Check step names
        step_names = [step[0] for step in config["steps"]]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_create_default_pipeline_config_scaling(self) -> None:
        """Test default pipeline configuration for scaling."""
        # Skip test if sklearn not available
        try:
            SklearnPipelineFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        config = SklearnPipelineFeatureGroup._create_default_pipeline_config("scaling")

        assert "steps" in config
        assert "params" in config
        assert len(config["steps"]) == 1

        # Check step name
        step_names = [step[0] for step in config["steps"]]
        assert "scaler" in step_names

    def test_create_default_pipeline_config_imputation(self) -> None:
        """Test default pipeline configuration for imputation."""
        # Skip test if sklearn not available
        try:
            SklearnPipelineFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        config = SklearnPipelineFeatureGroup._create_default_pipeline_config("imputation")

        assert "steps" in config
        assert "params" in config
        assert len(config["steps"]) == 1

        # Check step name
        step_names = [step[0] for step in config["steps"]]
        assert "imputer" in step_names

    def test_create_default_pipeline_config_unknown(self) -> None:
        """Test default pipeline configuration for unknown pipeline name."""
        # Skip test if sklearn not available
        try:
            SklearnPipelineFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        config = SklearnPipelineFeatureGroup._create_default_pipeline_config("unknown_pipeline")

        assert "steps" in config
        assert "params" in config
        assert len(config["steps"]) == 1

        # Should default to scaling
        step_names = [step[0] for step in config["steps"]]
        assert "scaler" in step_names

    def test_create_default_pipeline_config_missing_sklearn(self) -> None:
        """Test default pipeline configuration when sklearn is not available."""
        with patch.dict("sys.modules", {"sklearn.preprocessing": None, "sklearn.pipeline": None}):
            with pytest.raises(ImportError, match="scikit-learn is required"):
                SklearnPipelineFeatureGroup._create_default_pipeline_config("preprocessing")

    def test_pipeline_matches_config(self) -> None:
        """Test pipeline configuration matching."""
        # Skip test if sklearn not available
        try:
            sklearn_components = SklearnPipelineFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        StandardScaler = sklearn_components["StandardScaler"]
        SimpleImputer = sklearn_components["SimpleImputer"]
        Pipeline = sklearn_components["Pipeline"]

        # Create a pipeline
        pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])

        # Create matching config
        config = {"steps": [("imputer", SimpleImputer()), ("scaler", StandardScaler())], "params": {}}

        assert SklearnPipelineFeatureGroup._pipeline_matches_config(pipeline, config)

        # Create non-matching config (different number of steps)
        config_different = {"steps": [("scaler", StandardScaler())], "params": {}}

        assert not SklearnPipelineFeatureGroup._pipeline_matches_config(pipeline, config_different)

    def test_pipeline_matches_config_no_steps(self) -> None:
        """Test pipeline configuration matching with object without steps."""
        mock_pipeline = Mock()
        # Mock pipeline without steps attribute
        del mock_pipeline.steps

        config = {"steps": [("scaler", Mock())], "params": {}}

        assert not SklearnPipelineFeatureGroup._pipeline_matches_config(mock_pipeline, config)

    def test_abstract_methods_not_implemented(self) -> None:
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SklearnPipelineFeatureGroup._extract_training_data(None, [])

        with pytest.raises(NotImplementedError):
            SklearnPipelineFeatureGroup._apply_pipeline(None, [], None)

        with pytest.raises(NotImplementedError):
            SklearnPipelineFeatureGroup._check_source_feature_exists(None, "test")

        with pytest.raises(NotImplementedError):
            SklearnPipelineFeatureGroup._add_result_to_data(None, "test", None)
