"""
Artifact for storing trained forecasting models.
"""

import json
import pickle  # nosec
import base64
from typing import Any, Dict, Optional

from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class ForecastingArtifact(BaseArtifact):
    """
    Artifact for storing trained forecasting models.

    This artifact stores the trained model, scaler, and other metadata
    needed to generate forecasts without retraining.

    The artifact contains:
    - model: The trained scikit-learn model
    - scaler: StandardScaler used to normalize features
    - feature_engineering_params: Parameters for feature engineering
    - training_metadata: Information about the training data
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
        # Create a copy of the artifact
        serializable_artifact = {}

        # Serialize each component of the artifact
        for key, value in artifact.items():
            if key == "model" or key == "scaler":
                # Pickle and base64 encode the model and scaler
                pickled = pickle.dumps(value)
                serializable_artifact[key] = base64.b64encode(pickled).decode("utf-8")
            elif key == "feature_names":
                # Convert list to JSON
                serializable_artifact[key] = json.dumps(value)
            elif key == "last_trained_timestamp":
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
        # Parse the JSON string
        serializable_artifact = json.loads(serialized_artifact)

        # Create a new artifact
        artifact = {}

        # Deserialize each component of the artifact
        for key, value in serializable_artifact.items():
            if key == "model" or key == "scaler":
                # Base64 decode and unpickle the model and scaler
                pickled = base64.b64decode(value)
                artifact[key] = pickle.loads(pickled)  # nosec
            elif key == "feature_names":
                # Parse JSON list
                artifact[key] = json.loads(value)
            elif key == "last_trained_timestamp":
                # Keep timestamp as string for now
                artifact[key] = value
            else:
                # Keep other values as is
                artifact[key] = value

        return artifact

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Save the forecasting model artifact.

        Args:
            features: The feature set
            artifact: The model artifact to save (dict containing model, scaler, etc.)

        Returns:
            The saved artifact as a JSON string
        """
        return cls._serialize_artifact(artifact)

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        Load the forecasting model artifact.

        Args:
            features: The feature set

        Returns:
            The loaded artifact (dict containing model, scaler, etc.)
        """
        if features.options is None or features.name_of_one_feature is None:
            return None

        serialized_artifact = features.options.get(features.name_of_one_feature.name)
        if serialized_artifact is None:
            return None

        return cls._deserialize_artifact(serialized_artifact)
