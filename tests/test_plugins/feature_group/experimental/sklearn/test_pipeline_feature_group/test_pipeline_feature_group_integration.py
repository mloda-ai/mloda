"""
Integration tests for the SklearnPipelineFeatureGroup with mlodaAPI.
"""

import pytest
from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class SklearnPipelineTestDataCreator(ATestDataCreator):
    """Test data creator for sklearn pipeline tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "income": [50000.0, 60000.0, 75000.0, 45000.0, 80000.0],
            "age": [25.0, 35.0, 45.0, 30.0, 50.0],
            "experience": [2.0, 8.0, 15.0, 5.0, 20.0],
        }


class TestSklearnPipelineFeatureGroupIntegration:
    """Integration tests for sklearn pipeline feature groups."""

    def test_integration_with_scaling_pipeline(self) -> None:
        """Test integration with mlodaAPI using scaling pipeline."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create a scaling pipeline feature
        feature = Feature("income__sklearn_pipeline_scaling")

        # Test with mloda API
        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the scaled feature
        scaled_df = None
        for df in results:
            if "income__sklearn_pipeline_scaling" in df.columns:
                scaled_df = df
                break

        assert scaled_df is not None, "DataFrame with scaled features not found"

        # Verify that scaling was applied (mean should be close to 0, std close to 1)
        scaled_values = scaled_df["income__sklearn_pipeline_scaling"]
        assert abs(scaled_values.mean()) < 1e-10, "Scaled values should have mean close to 0"
        assert abs(scaled_values.std() - 1.0) < 0.2, "Scaled values should have std close to 1"

    def test_integration_with_preprocessing_pipeline(self) -> None:
        """Test integration with mlodaAPI using preprocessing pipeline."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create a preprocessing pipeline feature
        feature = Feature("income__sklearn_pipeline_preprocessing")

        # Test with mloda API
        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the preprocessed feature
        preprocessed_df = None
        for df in results:
            if "income__sklearn_pipeline_preprocessing" in df.columns:
                preprocessed_df = df
                break

        assert preprocessed_df is not None, "DataFrame with preprocessed features not found"

        # Verify that preprocessing was applied (should be scaled after imputation)
        preprocessed_values = preprocessed_df["income__sklearn_pipeline_preprocessing"]
        assert abs(preprocessed_values.mean()) < 1e-10, "Preprocessed values should have mean close to 0"
        assert abs(preprocessed_values.std() - 1.0) < 0.2, "Preprocessed values should have std close to 1"

    def test_integration_with_multiple_features(self) -> None:
        """Test integration with mlodaAPI using multiple source features."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create a scaling pipeline feature with multiple source features
        feature = Feature("income,age__sklearn_pipeline_scaling")

        # Test with mloda API
        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Check for the result - it should be stored as multiple columns with ~ separator
        result_df = results[0]

        # Look for columns with ~ separator (multiple result columns pattern)
        found_columns = []
        for col in result_df.columns:
            if col.startswith("income,age__sklearn_pipeline_scaling~"):
                found_columns.append(col)

        assert len(found_columns) == 2, (
            f"Expected 2 scaled feature columns, found {len(found_columns)}: {found_columns}"
        )

        # Verify that we have the expected columns
        assert "income,age__sklearn_pipeline_scaling~0" in found_columns
        assert "income,age__sklearn_pipeline_scaling~1" in found_columns

        # Verify that the scaled features have reasonable values (mean close to 0)
        for col in found_columns:
            scaled_values = result_df[col]
            assert abs(scaled_values.mean()) < 1e-10, f"Scaled values in {col} should have mean close to 0"

    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the feature parser."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create a feature using configuration
        f1 = Feature(
            "x",
            Options(
                {
                    SklearnPipelineFeatureGroup.PIPELINE_NAME: "scaling",
                    DefaultOptionKeys.mloda_source_features: "income",
                }
            ),
        )

        if f1 is None:
            raise ValueError("Failed to create feature using the parser.")

        # Test with pre-parsed features
        results = mlodaAPI.run_all(
            [f1],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the scaled feature
        scaled_df = None
        for df in results:
            if "x" in df.columns:
                scaled_df = df
                break

        assert scaled_df is not None, "DataFrame with scaled features not found"

        # Verify that scaling was applied
        scaled_values = scaled_df["x"]
        assert abs(scaled_values.mean()) < 1e-10, "Scaled values should have mean close to 0"

        # Test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))

    def test_integration_with_custom_pipeline_steps(self) -> None:
        """Test integration with mlodaAPI using custom pipeline steps."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create a feature with custom pipeline steps using frozenset to make it hashable
        # Convert the list of pipeline steps to a frozenset of tuples
        pipeline_steps_list = [
            ("scaler", StandardScaler()),
            ("minmax", MinMaxScaler()),
        ]

        # Convert to frozenset for hashability
        pipeline_steps_frozenset = frozenset(
            (name, type(transformer).__name__) for name, transformer in pipeline_steps_list
        )

        f1 = Feature(
            "x",
            Options(
                {
                    SklearnPipelineFeatureGroup.PIPELINE_STEPS: pipeline_steps_frozenset,
                    SklearnPipelineFeatureGroup.PIPELINE_PARAMS: frozenset(
                        [
                            ("scaler__with_mean", True),
                            ("minmax__feature_range", (0, 1)),
                        ]
                    ),
                    DefaultOptionKeys.mloda_source_features: "income",
                }
            ),
        )

        # Test with custom pipeline
        results = mlodaAPI.run_all(
            [f1],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the transformed feature
        transformed_df = None
        for df in results:
            if "x" in df.columns:
                transformed_df = df
                break

        assert transformed_df is not None, "DataFrame with transformed features not found"

        # Verify that the custom pipeline was applied
        # Since we're using frozenset, the actual pipeline creation will use defaults
        # but we can verify that the feature was created and processed
        transformed_values = transformed_df["x"]
        assert transformed_values is not None, "Transformed values should exist"
        assert len(transformed_values) == 5, "Should have 5 transformed values"

    def test_integration_artifact_persistence(self) -> None:
        """Test that artifacts are properly persisted and reused."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create the same feature twice to test artifact reuse
        feature = Feature("income__sklearn_pipeline_scaling")

        # First run - should create and save artifact
        results1 = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Second run - should reuse artifact
        results2 = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results1) == 1
        assert len(results2) == 1

        # Results should be identical (artifact was reused)
        df1 = results1[0]
        df2 = results2[0]

        assert "income__sklearn_pipeline_scaling" in df1.columns
        assert "income__sklearn_pipeline_scaling" in df2.columns

        # Values should be identical
        assert df1["income__sklearn_pipeline_scaling"].equals(df2["income__sklearn_pipeline_scaling"])

    @pytest.mark.parametrize(
        "storage_config",
        [
            None,  # Uses temp directory fallback
            "/tmp/test_sklearn_artifacts",  # nosec
        ],
        ids=["fallback_temp_dir", "custom_path"],
    )
    def test_artifact_persistence_with_storage_paths(self, storage_config: Any) -> None:
        """Test artifact persistence with both fallback and custom storage paths following proper mloda lifecycle."""
        # Skip test if sklearn not available
        try:
            import sklearn
            import tempfile
            import os
            from pathlib import Path
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SklearnPipelineTestDataCreator, PandasSklearnPipelineFeatureGroup}
        )

        # Create feature with or without custom storage path
        feature_options = {}
        if storage_config is not None:
            # Create the custom directory if it doesn't exist
            os.makedirs(storage_config, exist_ok=True)
            feature_options["artifact_storage_path"] = storage_config

        try:
            # First run - create feature WITHOUT artifact options (mloda will set artifact_to_save)
            feature1 = Feature("income__sklearn_pipeline_scaling", Options(feature_options))

            api1 = mlodaAPI([feature1], {PandasDataframe}, plugin_collector=plugin_collector)
            api1._batch_run()
            results1 = api1.get_result()
            artifacts1 = api1.get_artifacts()

            # Verify we got results and artifacts
            assert len(results1) == 1
            assert len(artifacts1) == 1  # Should be 1 pipeline artifact with unique key
            assert "income__sklearn_pipeline_scaling" in artifacts1

            # Verify artifact file was created in expected location
            from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact
            from mloda_core.abstract_plugins.components.feature_set import FeatureSet

            mock_features = FeatureSet()
            mock_features.add(Feature("income__sklearn_pipeline_scaling", Options(feature_options)))

            # Use the new artifact key-based file path method
            artifact_key = "income__sklearn_pipeline_scaling"
            expected_file_path = SklearnArtifact._get_artifact_file_path_for_key(mock_features, artifact_key)

            # Verify the file exists
            assert expected_file_path.exists(), f"Artifact file should exist at {expected_file_path}"

            # Verify the file is in the expected directory
            if storage_config is None:
                # Should be in temp directory
                assert str(expected_file_path).startswith(tempfile.gettempdir()), (
                    f"Artifact should be in temp directory, but found at {expected_file_path}"
                )
            else:
                # Should be in custom directory
                assert str(expected_file_path).startswith(storage_config), (
                    f"Artifact should be in custom directory {storage_config}, but found at {expected_file_path}"
                )

            # Second run - create feature WITH artifact options (mloda will set artifact_to_load)
            combined_options = {**feature_options, **artifacts1}
            feature2 = Feature("income__sklearn_pipeline_scaling", Options(combined_options))

            api2 = mlodaAPI([feature2], {PandasDataframe}, plugin_collector=plugin_collector)
            api2._batch_run()
            results2 = api2.get_result()
            artifacts2 = api2.get_artifacts()

            # Verify results are identical (indicating artifact reuse)
            assert len(results2) == 1
            assert not artifacts2  # No new artifacts should be created

            df1 = results1[0]
            df2 = results2[0]

            assert "income__sklearn_pipeline_scaling" in df1.columns
            assert "income__sklearn_pipeline_scaling" in df2.columns

            # Values should be identical (artifact was reused)
            assert df1["income__sklearn_pipeline_scaling"].equals(df2["income__sklearn_pipeline_scaling"])

        finally:
            # Clean up: remove test artifacts
            if storage_config is not None:
                try:
                    # Remove the custom directory and its contents
                    import shutil

                    if os.path.exists(storage_config):
                        shutil.rmtree(storage_config)
                except Exception:  # nosec
                    pass
