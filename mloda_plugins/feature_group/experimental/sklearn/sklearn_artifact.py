"""
Artifact for storing fitted scikit-learn transformers and estimators.
"""

import json
import base64
import os
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class SklearnArtifact(BaseArtifact):
    """
    Artifact for storing fitted scikit-learn transformers and estimators.

    This artifact stores fitted scikit-learn objects using joblib serialization,
    allowing for efficient persistence and reuse of trained models and transformers.

    The artifact contains:
    - fitted_transformer: The fitted scikit-learn object
    - feature_names: Names of input features used during fitting
    - training_metadata: Information about the training process

    This class also provides helper methods for managing multiple artifacts
    in sklearn feature groups (encoding, scaling, pipeline).
    """

    @classmethod
    def _serialize_artifact(cls, artifact: Dict[str, Any]) -> str:
        """
        Serialize the artifact to a JSON string.

        Args:
            artifact: The artifact to serialize

        Returns:
            A JSON string representation of the artifact
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        # Create a copy of the artifact
        serializable_artifact = {}

        # Serialize each component of the artifact
        for key, value in artifact.items():
            if key == "fitted_transformer":
                # Use joblib to serialize the fitted transformer
                import io

                buffer = io.BytesIO()
                joblib.dump(value, buffer)
                serializable_artifact[key] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif key == "feature_names":
                # Convert list to JSON
                serializable_artifact[key] = json.dumps(value)
            elif key == "training_timestamp":
                # Convert timestamp to string
                serializable_artifact[key] = str(value)
            else:
                # Keep other values as is
                serializable_artifact[key] = value

        # Convert the entire artifact to a JSON string
        return json.dumps(serializable_artifact)

    @classmethod
    def _deserialize_artifact(cls, serialized_artifact: str) -> Dict[str, Any]:
        """
        Deserialize the artifact from a JSON string.

        Args:
            serialized_artifact: The JSON string to deserialize

        Returns:
            The deserialized artifact
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        # Parse the JSON string
        serializable_artifact = json.loads(serialized_artifact)

        # Create a new artifact
        artifact = {}

        # Deserialize each component of the artifact
        for key, value in serializable_artifact.items():
            if key == "fitted_transformer":
                # Use joblib to deserialize the fitted transformer
                import io

                buffer = io.BytesIO(base64.b64decode(value))
                artifact[key] = joblib.load(buffer)
            elif key == "feature_names":
                # Parse JSON list
                artifact[key] = json.loads(value)
            elif key == "training_timestamp":
                # Keep timestamp as string for now
                artifact[key] = value
            else:
                # Keep other values as is
                artifact[key] = value

        return artifact

    @classmethod
    def _get_artifact_file_path(cls, features: FeatureSet) -> Path:
        """
        Generate a file path for storing the artifact.

        Args:
            features: The feature set

        Returns:
            Path object for the artifact file
        """
        if features.name_of_one_feature is None:
            raise ValueError("Feature name is required for artifact storage")

        # Get storage path from options or use default temp directory
        storage_path = None
        if features.options:
            storage_path = features.options.data.get("artifact_storage_path")

        if storage_path is None:
            storage_path = tempfile.gettempdir()

        # Create a unique filename based on feature name and configuration
        feature_name = features.name_of_one_feature.name

        # Create a hash of the feature configuration for uniqueness
        # Exclude artifact-related keys to ensure consistent hashing
        config_data = {}
        if features.options:
            config_data = {
                k: v
                for k, v in features.options.data.items()
                if not k.startswith(feature_name) and k != "artifact_storage_path"
            }

        # Convert non-serializable objects for hashing
        serializable_config = {}
        for k, v in config_data.items():
            if isinstance(v, frozenset):
                # Convert frozenset to sorted list for consistent hashing
                serializable_config[k] = sorted(list(v))
            else:
                serializable_config[k] = v

        config_str = json.dumps(serializable_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]

        filename = f"sklearn_artifact_{feature_name}_{config_hash}.joblib"

        # Ensure the directory exists
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        return storage_dir / filename

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Save sklearn artifacts to file(s).

        Args:
            features: The feature set
            artifact: Dictionary of {artifact_key: artifact_data} where each artifact_data
                     contains fitted_transformer, feature_name, encoder_type, etc.

        Returns:
            Dictionary of {artifact_key: file_path} where artifacts were saved
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        if not isinstance(artifact, dict):
            raise ValueError(f"Expected artifact to be a dictionary, got {type(artifact)}")

        saved_paths = {}

        for artifact_key, artifact_data in artifact.items():
            # Generate unique file path for this artifact
            file_path = cls._get_artifact_file_path_for_key(features, artifact_key)

            # Save this specific artifact
            joblib.dump(artifact_data, file_path)
            saved_paths[artifact_key] = str(file_path)

        return saved_paths

    @classmethod
    def _get_artifact_file_path_for_key(cls, features: FeatureSet, artifact_key: str) -> Path:
        """
        Generate a file path for storing a specific artifact by key.

        Args:
            features: The feature set
            artifact_key: The specific artifact key (e.g., "onehot_encoded__category")

        Returns:
            Path object for the artifact file
        """
        # Get storage path from options or use default temp directory
        storage_path = None
        if features.options:
            storage_path = features.options.data.get("artifact_storage_path")

        if storage_path is None:
            storage_path = tempfile.gettempdir()

        # Simple filename based on artifact key
        filename = f"sklearn_artifact_{artifact_key}.joblib"

        # Ensure the directory exists
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        return storage_dir / filename

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        Load sklearn artifacts from file(s).

        Args:
            features: The feature set

        Returns:
            Dictionary of {artifact_key: artifact_data} containing all available artifacts
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        print(f"DEBUG: SklearnArtifact.custom_loader called")

        # Get storage path
        storage_path = None
        if features.options:
            storage_path = features.options.data.get("artifact_storage_path")
        if storage_path is None:
            storage_path = tempfile.gettempdir()

        storage_dir = Path(storage_path)
        if not storage_dir.exists():
            print("DEBUG: SklearnArtifact.custom_loader - storage directory does not exist")
            return None

        # Find all sklearn artifact files
        pattern = "sklearn_artifact_*.joblib"
        loaded_artifacts = {}

        for file_path in storage_dir.glob(pattern):
            try:
                # Extract artifact key from filename
                filename = file_path.stem  # Remove .joblib extension
                # Format: sklearn_artifact_{artifact_key}
                if filename.startswith("sklearn_artifact_"):
                    artifact_key = filename[len("sklearn_artifact_") :]

                    print(f"DEBUG: SklearnArtifact.custom_loader - loading {artifact_key} from {file_path}")

                    # Load the artifact
                    artifact_data = joblib.load(file_path)
                    loaded_artifacts[artifact_key] = artifact_data

                    print(f"DEBUG: SklearnArtifact.custom_loader - successfully loaded {artifact_key}")

            except Exception as e:
                print(f"Warning: Failed to load artifact from {file_path}: {e}")
                continue

        if loaded_artifacts:
            print(
                f"DEBUG: SklearnArtifact.custom_loader - loaded {len(loaded_artifacts)} artifacts: {list(loaded_artifacts.keys())}"
            )
            return loaded_artifacts
        else:
            print("DEBUG: SklearnArtifact.custom_loader - no artifacts found")
            return None

    @classmethod
    def load_sklearn_artifact(cls, features: FeatureSet, artifact_key: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to load a specific sklearn artifact by key.

        Args:
            features: The feature set
            artifact_key: The specific artifact key to load

        Returns:
            The artifact data if found, None otherwise
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            if artifacts and artifact_key in artifacts:
                return artifacts[artifact_key]  # type: ignore
            # If artifact_to_load is true but we can't find the specific key, that's an error
            available_keys = list(artifacts.keys()) if artifacts else []
            raise ValueError(f"Artifact not found for key '{artifact_key}'. Available artifacts: {available_keys}")
        return None

    @classmethod
    def save_sklearn_artifact(cls, features: FeatureSet, artifact_key: str, artifact_data: Dict[str, Any]) -> None:
        """
        Helper method to save a sklearn artifact with the proper multiple artifact format.

        Args:
            features: The feature set
            artifact_key: The unique key for this artifact
            artifact_data: The artifact data to save
        """
        if features.artifact_to_save:
            # Support multiple artifacts by using a dictionary
            if not isinstance(features.save_artifact, dict):
                features.save_artifact = {}
            features.save_artifact[artifact_key] = artifact_data

    @classmethod
    def check_artifact_exists(cls, features: FeatureSet, artifact_key: str) -> bool:
        """
        Helper method to check if a specific artifact exists.

        Args:
            features: The feature set
            artifact_key: The artifact key to check

        Returns:
            True if the artifact exists, False otherwise
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            return artifacts is not None and artifact_key in artifacts
        return False

    @classmethod
    def validate_artifact_key_exists(cls, features: FeatureSet, artifact_key: str) -> None:
        """
        Helper method to validate that an artifact key exists, raising an error if not.

        Args:
            features: The feature set
            artifact_key: The artifact key to validate

        Raises:
            ValueError: If the artifact key is not found
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            if artifacts is None or artifact_key not in artifacts:
                available_keys = list(artifacts.keys()) if artifacts else []
                raise ValueError(f"Artifact not found for key '{artifact_key}'. Available artifacts: {available_keys}")
