"""
Tests for SklearnArtifact.
"""

from pathlib import Path

from mloda.user import Feature
import pytest
from unittest.mock import Mock, patch
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact
from mloda.provider import FeatureSet
from mloda.user import Options


class TestSklearnArtifact:
    """Test cases for SklearnArtifact."""

    def test_empty_string_artifact_to_save_proceeds(self) -> None:
        """artifact_to_save="" must not be silently treated as unset."""
        features = FeatureSet()
        features.artifact_to_save = ""
        SklearnArtifact.save_sklearn_artifact(features, "my_key", {"data": 1})
        assert features.save_artifact == {"my_key": {"data": 1}}

    def test_none_artifact_to_save_skips(self) -> None:
        """artifact_to_save=None still skips (no change in behaviour)."""
        features = FeatureSet()
        assert features.artifact_to_save is None
        SklearnArtifact.save_sklearn_artifact(features, "my_key", {"data": 1})
        assert features.save_artifact is None

    def test_empty_string_artifact_to_load_attempts_load(self) -> None:
        """artifact_to_load="" must enter the load branch, not silently return None."""
        features = FeatureSet()
        features.artifact_to_load = ""
        with patch.object(SklearnArtifact, "custom_loader", return_value=None):
            with pytest.raises(ValueError, match="Artifact not found"):
                SklearnArtifact.load_sklearn_artifact(features, "my_key")

    def test_none_artifact_to_load_returns_none(self) -> None:
        """artifact_to_load=None still returns None (no change in behaviour)."""
        features = FeatureSet()
        assert features.artifact_to_load is None
        result = SklearnArtifact.load_sklearn_artifact(features, "my_key")
        assert result is None

    def test_serialize_deserialize_artifact(self) -> None:
        """Test serialization and deserialization of sklearn artifacts."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib  # noqa: F401
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

    def test_custom_saver(self, tmp_path: Path) -> None:
        """Test custom_saver method."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib  # noqa: F401
            from sklearn.preprocessing import StandardScaler
            import os
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        features = FeatureSet()
        features.add(Feature("test_custom_saver_feature", Options({"artifact_storage_path": str(tmp_path)})))

        # Use the new multiple artifact format
        artifact = {
            "test_artifact_key": {
                "fitted_transformer": StandardScaler(),
                "feature_names": ["test_custom_saver_feature"],
            }
        }

        result = SklearnArtifact.custom_saver(features, artifact)
        assert isinstance(result, dict)
        assert "test_artifact_key" in result
        # Verify file was created
        assert os.path.exists(result["test_artifact_key"])

    def test_custom_loader_no_options(self) -> None:
        """Test custom_loader when no options are available."""
        import tempfile

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

    def test_custom_loader_with_artifact(self, tmp_path: Path) -> None:
        """Test custom_loader with stored artifact."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib  # noqa: F401
            from sklearn.preprocessing import StandardScaler
            import os
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
        features.add(Feature("test_with_artifact_feature_unique", Options({"artifact_storage_path": str(tmp_path)})))

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


class TestSklearnArtifactDefaultStorage:
    """The default storage dir is private per user, never the shared temp dir; saves are atomic."""

    @staticmethod
    def _features_without_storage_path() -> FeatureSet:
        features = FeatureSet()
        features.add(Feature("default_storage_feature", Options()))
        return features

    def test_loader_ignores_files_planted_in_shared_temp_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("joblib")
        import joblib
        import tempfile

        shared_tmp = tmp_path / "shared_tmp"
        shared_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(shared_tmp))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        joblib.dump({"planted": True}, shared_tmp / "sklearn_artifact_planted.joblib")

        assert SklearnArtifact.custom_loader(self._features_without_storage_path()) is None

    def test_default_dir_is_private_under_xdg_cache_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("joblib")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        features = self._features_without_storage_path()

        saved = SklearnArtifact.custom_saver(features, {"key": {"data": 1}})

        expected_dir = tmp_path / "cache" / "mloda" / "sklearn_artifacts"
        assert saved == {"key": str(expected_dir / "sklearn_artifact_key.joblib")}
        assert expected_dir.stat().st_mode & 0o077 == 0
        assert SklearnArtifact.custom_loader(features) == {"key": {"data": 1}}

    def test_default_dir_writable_by_others_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        storage_dir = tmp_path / "mloda" / "sklearn_artifacts"
        storage_dir.mkdir(parents=True)
        storage_dir.chmod(0o777)

        with pytest.raises(PermissionError, match="writable by other users"):
            SklearnArtifact.custom_loader(self._features_without_storage_path())

    def test_default_dir_not_owned_by_user_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        if not hasattr(os, "getuid"):
            pytest.skip("POSIX ownership check")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)

        with pytest.raises(PermissionError, match="not a directory owned by this user"):
            SklearnArtifact.custom_loader(self._features_without_storage_path())

    def test_save_goes_through_temp_file_and_replace(self, tmp_path: Path) -> None:
        pytest.importorskip("joblib")
        import os

        features = FeatureSet()
        features.add(Feature("atomic_feature", Options({"artifact_storage_path": str(tmp_path)})))
        real_replace = os.replace
        calls: list[tuple[str, str]] = []

        def recording_replace(src: str, dst: Path) -> None:
            calls.append((str(src), str(dst)))
            real_replace(src, dst)

        with patch("mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact.os.replace", recording_replace):
            saved = SklearnArtifact.custom_saver(features, {"key": {"data": 1}})

        assert isinstance(saved, dict)
        assert len(calls) == 1
        assert Path(calls[0][0]).parent == tmp_path
        assert calls[0][1] == saved["key"]
        assert [p.name for p in tmp_path.iterdir()] == ["sklearn_artifact_key.joblib"]

    def test_failed_save_leaves_no_partial_file(self, tmp_path: Path) -> None:
        pytest.importorskip("joblib")
        features = FeatureSet()
        features.add(Feature("atomic_fail_feature", Options({"artifact_storage_path": str(tmp_path)})))

        with patch("joblib.dump", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                SklearnArtifact.custom_saver(features, {"key": {"data": 1}})

        assert list(tmp_path.iterdir()) == []
