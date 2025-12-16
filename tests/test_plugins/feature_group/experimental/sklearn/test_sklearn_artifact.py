"""
Tests for SklearnArtifact.
"""

from mloda import Feature
import pytest
from unittest.mock import Mock, patch
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact
from mloda.provider import FeatureSet
from mloda import Options


class TestSklearnArtifact:
    """Test cases for SklearnArtifact."""

    def test_serialize_deserialize_artifact(self) -> None:
        """Test serialization and deserialization of sklearn artifacts."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        # Create a simple sklearn transformer
        scaler = StandardScaler()
        # Fit with some dummy data
        import numpy as np

        dummy_data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(dummy_data)

        # Create artifact
        artifact = {
            "fitted_transformer": scaler,
            "feature_names": ["feature1", "feature2"],
            "training_timestamp": "2023-01-01T00:00:00",
        }

        # Serialize
        serialized = SklearnArtifact._serialize_artifact(artifact)
        assert isinstance(serialized, str)

        # Deserialize
        deserialized = SklearnArtifact._deserialize_artifact(serialized)

        # Verify contents
        assert "fitted_transformer" in deserialized
        assert "feature_names" in deserialized
        assert "training_timestamp" in deserialized

        assert deserialized["feature_names"] == ["feature1", "feature2"]
        assert deserialized["training_timestamp"] == "2023-01-01T00:00:00"

        # Verify the transformer works
        result = deserialized["fitted_transformer"].transform(dummy_data)
        assert result.shape == (3, 2)

    def test_serialize_artifact_missing_joblib(self) -> None:
        """Test serialization when joblib is not available."""
        with patch.dict("sys.modules", {"joblib": None}):
            with pytest.raises(ImportError, match="joblib is required"):
                SklearnArtifact._serialize_artifact({"fitted_transformer": Mock()})

    def test_deserialize_artifact_missing_joblib(self) -> None:
        """Test deserialization when joblib is not available."""
        with patch.dict("sys.modules", {"joblib": None}):
            with pytest.raises(ImportError, match="joblib is required"):
                SklearnArtifact._deserialize_artifact('{"fitted_transformer": "dummy"}')

    def test_custom_saver(self) -> None:
        """Test custom_saver method."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            import tempfile
            import os
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        features = FeatureSet()
        features.add(Feature("test_custom_saver_feature", Options()))

        # Use the new multiple artifact format
        artifact = {
            "test_artifact_key": {
                "fitted_transformer": StandardScaler(),
                "feature_names": ["test_custom_saver_feature"],
            }
        }

        try:
            result = SklearnArtifact.custom_saver(features, artifact)
            assert isinstance(result, dict)
            assert "test_artifact_key" in result
            # Verify file was created
            assert os.path.exists(result["test_artifact_key"])
        finally:
            # Clean up
            try:
                import glob

                artifact_files = glob.glob("/tmp/sklearn_artifact_*.joblib")  # nosec
                for file_path in artifact_files:
                    os.remove(file_path)
            except Exception:  # nosec
                pass

    def test_custom_loader_no_options(self) -> None:
        """Test custom_loader when no options are available."""
        import tempfile
        import os

        # Use a unique temporary directory to ensure isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            features = FeatureSet()
            features.add(Feature("test_no_options_feature", Options({"artifact_storage_path": temp_dir})))

            result = SklearnArtifact.custom_loader(features)
            assert result is None

    def test_custom_loader_no_artifact(self) -> None:
        """Test custom_loader when no artifact is stored."""
        import tempfile

        # Use a unique temporary directory to ensure isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            features = FeatureSet()
            features.add(Feature("test_no_artifact_feature_unique", Options({"artifact_storage_path": temp_dir})))

            result = SklearnArtifact.custom_loader(features)
            assert result is None

    def test_custom_loader_with_artifact(self) -> None:
        """Test custom_loader with stored artifact."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            import os
            import glob
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        # Create and serialize an artifact using the new multiple artifact format
        scaler = StandardScaler()
        import numpy as np

        dummy_data = np.array([[1, 2], [3, 4]])
        scaler.fit(dummy_data)

        # Use the new multiple artifact format
        artifact = {"test_with_artifact_key": {"fitted_transformer": scaler, "feature_names": ["feature1", "feature2"]}}

        # Mock features with unique name
        features = FeatureSet()
        features.add(Feature("test_with_artifact_feature_unique", Options({})))

        try:
            # First save the artifact
            saved_paths = SklearnArtifact.custom_saver(features, artifact)
            assert isinstance(saved_paths, dict)
            assert "test_with_artifact_key" in saved_paths
            assert os.path.exists(saved_paths["test_with_artifact_key"])

            # Then load it
            result = SklearnArtifact.custom_loader(features)

            assert result is not None
            assert isinstance(result, dict)
            assert "test_with_artifact_key" in result
            loaded_artifact = result["test_with_artifact_key"]
            assert "fitted_transformer" in loaded_artifact
            assert "feature_names" in loaded_artifact
            assert loaded_artifact["feature_names"] == ["feature1", "feature2"]
        finally:
            # Clean up
            try:
                artifact_files = glob.glob("/tmp/sklearn_artifact_*.joblib")  # nosec
                for file_path in artifact_files:
                    os.remove(file_path)
            except Exception:  # nosec
                pass
