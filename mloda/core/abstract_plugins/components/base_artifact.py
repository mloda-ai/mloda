from abc import ABC
from typing import Any, Optional, final

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options


class BaseArtifact(ABC):
    """
    Abstract base class for handling artifacts within a FeatureSet.

    Artifacts are persistent, reusable outputs produced by a FeatureGroup (e.g. fitted
    sklearn transformers, trained models, computed embeddings). They allow a feature
    group to train/fit once, save the result, and reload it for subsequent runs.

    Lifecycle
    ---------
    1. A FeatureGroup declares its artifact class by overriding ``artifact()``
       to return a BaseArtifact subclass.

    2. When the framework builds a FeatureSet, ``FeatureSet.add_artifact_name()``
       checks whether a previously saved artifact exists in Options:
       - Found:  sets ``artifact_to_load`` (feature name key).
       - Not found: sets ``artifact_to_save`` (feature name key).

    3. During ``calculate_feature``, the producer (plugin author) uses the
       artifact class helpers to interact with artifacts:
       - Load: call the artifact's load helper (e.g. ``SklearnArtifact.load_sklearn_artifact``)
         and apply the loaded object instead of re-fitting.
       - Save: call the artifact's save helper (e.g. ``SklearnArtifact.save_sklearn_artifact``)
         to store the fitted object on ``features.save_artifact``.

    4. After ``calculate_feature`` returns, the framework calls
       ``BaseArtifact.save(features, features.save_artifact)`` if
       ``artifact_to_save`` is set. The ``save`` method delegates to
       ``custom_saver``, and the returned value (the artifact data or
       metadata such as a file path) is registered in the CfwManager for
       subsequent runs.

    Customization
    -------------
    Subclasses override ``custom_loader`` and ``custom_saver`` to control
    serialization. The default ``custom_loader`` reads from Options; the
    default ``custom_saver`` returns the artifact as-is for framework storage.
    For large or non-picklable data, override these to persist to disk (see
    ``SklearnArtifact`` for an example using joblib files).
    """

    @final
    @classmethod
    def load(cls, features: FeatureSet) -> Optional[Any]:
        """
        Loads an artifact from the given config of the feature set, when the custom_loader is not overwritten.
        If the custom_loader is overwritten, this method will call the custom_loader and return the result.

        This method is crucial for data science processes where the reuse of previously computed data
        (artifacts) is necessary. For example, in machine learning pipelines,
        precomputed embeddings can be loaded and reused across different stages of the pipeline.
        """

        if features.artifact_to_load is None:
            return None

        cls._validate(features)

        loaded_artifact = cls.custom_loader(features)

        if loaded_artifact is None:
            raise ValueError("No artifact to load although it was requested.")

        return loaded_artifact

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        In the default case, it loads an artifact from the given config of the features.

        However, you can overwrite this method to load the artifact by any means necessary.
        """

        options = cls.get_singular_option_from_options(features)
        if options is None or features.name_of_one_feature is None:
            return None
        return options[str(features.name_of_one_feature)]

    @classmethod
    def get_singular_option_from_options(cls, features: FeatureSet) -> Options | None:
        """
        Retrieve a single shared Options object from the FeatureSet.

        Artifacts require all features in the set to share the same Options.
        If features carry different Options (e.g. due to context-based
        partitioning), saving and loading a single artifact per feature set
        is ambiguous. Raises ValueError when Options differ across features.
        """

        _options = None
        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError(
                        "All features in the set must share the same Options for artifact "
                        "save/load. Found differing Options, which can happen when context-based "
                        "partitioning produces multiple option variants in one FeatureSet."
                    )

            _options = feature.options

        if _options is None:
            return None

        return _options

    @final
    @classmethod
    def save(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        The default implementation is to return the artifact, as then the framework will handle it.

        In case that the data is larger or cannot be pickled, you can overwrite the custom_saver function to save the artifact by any means necessary.
        In that case, the return value would not be the artifact, but any metadata to identify this artifact.

        Returns:
            Optional[Any]: The artifact or metadata identifying the artifact, depending on implementation.
                           Default behavior is to return the artifact, as then the framework will handle it.

        """
        if features.artifact_to_save is None:
            return None

        cls._validate(features)

        artifact = cls.custom_saver(features, artifact)

        if artifact is None:
            raise ValueError("No artifact to save although it was requested.")

        return artifact

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Subclasses can override this method to implement custom saving logic.

        The default implementation is to return the artifact, as then the framework will handle it.
        """
        return artifact

    @staticmethod
    def _validate(features: FeatureSet) -> None:
        """
        Validates that the FeatureSet has the necessary attributes set.
        """

        options = BaseArtifact.get_singular_option_from_options(features)
        if options is None:
            raise ValueError("No options set. This should only be called after adding a feature.")

        if features.name_of_one_feature is None:
            raise ValueError("Feature name missing in feature set.")

    @classmethod
    @final
    def get_class_name(cls) -> str:
        return cls.__name__
