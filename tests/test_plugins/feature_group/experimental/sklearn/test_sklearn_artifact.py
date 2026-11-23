"""
Tests for SklearnArtifact.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from mloda.user import Feature
import pytest
from unittest.mock import Mock, patch
from mloda_plugins.feature_group.experimental.sklearn import sklearn_artifact
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact
from mloda.provider import FeatureSet
from mloda.user import Options

REAL_DEFAULT_STORAGE_DIR = sklearn_artifact.default_storage_dir


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

    @pytest.mark.skipif(os.name != "posix", reason="POSIX owner and mode check")
    def test_default_storage_is_private_user_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default storage is a private per-user dir and never reads the shared temp dir."""
        import joblib

        monkeypatch.setattr(sklearn_artifact, "default_storage_dir", REAL_DEFAULT_STORAGE_DIR)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        shared_tmp = tmp_path / "shared"
        shared_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(shared_tmp))
        joblib.dump({"evil": 1}, shared_tmp / "sklearn_artifact_planted.joblib")

        default_dir = tmp_path / "cache" / "mloda" / "sklearn_artifacts"
        features = FeatureSet()
        features.add(Feature("test_default_private_feature", Options({})))

        assert SklearnArtifact.custom_loader(features) is None
        assert not default_dir.exists()

        SklearnArtifact.custom_saver(features, {"private_key": {"value": 1}})
        assert default_dir.is_dir()
        assert (default_dir.stat().st_mode & 0o777) == 0o700

        loaded = SklearnArtifact.custom_loader(features)
        assert loaded == {"private_key": {"value": 1}}

    @pytest.mark.skipif(os.name != "posix", reason="POSIX owner and mode check")
    def test_default_storage_refuses_shared_or_foreign_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sklearn_artifact, "default_storage_dir", REAL_DEFAULT_STORAGE_DIR)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        default_dir = tmp_path / "cache" / "mloda" / "sklearn_artifacts"
        features = FeatureSet()
        features.add(Feature("test_default_refuse_feature", Options({})))

        SklearnArtifact.custom_saver(features, {"refuse_key": {"value": 1}})
        default_dir.chmod(0o755)
        with pytest.raises(PermissionError, match="not shared") as exc_info:
            SklearnArtifact.custom_loader(features)
        assert "artifact_storage_path" in str(exc_info.value)
        with pytest.raises(PermissionError, match="not shared"):
            SklearnArtifact.custom_saver(features, {"refuse_key": {"value": 2}})

        default_dir.chmod(0o700)
        real_uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: real_uid + 1)
        with pytest.raises(PermissionError, match="not shared"):
            SklearnArtifact.custom_loader(features)

    def test_relative_xdg_cache_home_falls_back_to_home_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert REAL_DEFAULT_STORAGE_DIR() == tmp_path / "home" / ".cache" / "mloda" / "sklearn_artifacts"

    def test_explicit_shared_storage_path_is_not_permission_checked(self, tmp_path: Path) -> None:
        storage = tmp_path / "explicit"
        storage.mkdir()
        storage.chmod(0o755)
        features = FeatureSet()
        features.add(Feature("test_explicit_shared_feature", Options({"artifact_storage_path": str(storage)})))

        SklearnArtifact.custom_saver(features, {"explicit_key": {"value": 1}})
        assert SklearnArtifact.custom_loader(features) == {"explicit_key": {"value": 1}}

    def test_failed_save_keeps_previous_files_and_no_temp(self, tmp_path: Path) -> None:
        import joblib

        features = FeatureSet()
        features.add(Feature("test_failed_save_feature", Options({"artifact_storage_path": str(tmp_path)})))
        SklearnArtifact.custom_saver(features, {"key_a": {"value": "old_a"}, "key_b": {"value": "old_b"}})

        real_dump = joblib.dump
        calls: list[int] = []

        def flaky_dump(value: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                return real_dump(value, target, *args, **kwargs)
            Path(target).write_bytes(b"partial")
            raise OSError("disk full")

        with patch("joblib.dump", side_effect=flaky_dump):
            with pytest.raises(OSError, match="disk full"):
                SklearnArtifact.custom_saver(features, {"key_a": {"value": "new_a"}, "key_b": {"value": "new_b"}})

        assert SklearnArtifact.custom_loader(features) == {"key_a": {"value": "old_a"}, "key_b": {"value": "old_b"}}
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "sklearn_artifact_key_a.joblib",
            "sklearn_artifact_key_b.joblib",
        ]
