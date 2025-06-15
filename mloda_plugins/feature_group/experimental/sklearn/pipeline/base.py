"""
Base implementation for scikit-learn pipeline feature groups.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact


class SklearnPipelineFeatureGroup(AbstractFeatureGroup):
    """
    Base class for scikit-learn pipeline feature groups.

    The SklearnPipelineFeatureGroup wraps entire scikit-learn pipelines as single features,
    demonstrating mloda's pipeline management capabilities compared to traditional scikit-learn usage.

    ## Feature Naming Convention

    Pipeline features follow this naming pattern:
    `sklearn_pipeline_{pipeline_name}__{mloda_source_features}`

    The source features (mloda_source_features) are extracted from the feature name and used
    as input for the pipeline. Note the double underscore before the source features.

    Examples:
    - `sklearn_pipeline_preprocessing__raw_features`: Apply preprocessing pipeline to raw_features
    - `sklearn_pipeline_feature_engineering__customer_data`: Apply feature engineering to customer_data
    - `sklearn_pipeline_scaling__income,age`: Apply scaling pipeline to income and age features

    ## Configuration-Based Creation

    SklearnPipelineFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a pipeline feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            SklearnPipelineFeatureGroup.PIPELINE_NAME: "preprocessing",
            SklearnPipelineFeatureGroup.PIPELINE_STEPS: [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer())
            ],
            DefaultOptionKeys.mloda_source_feature: "raw_features"
        })
    )

    # The Engine will automatically parse this into a feature with name
    # "sklearn_pipeline_preprocessing__raw_features"
    ```

    ## Key Advantages over Traditional Scikit-learn

    1. **Dependency Management**: Automatic resolution of feature dependencies
    2. **Reusability**: Pipeline definitions can be reused across projects
    3. **Versioning**: Automatic versioning of pipeline transformations
    4. **Framework Flexibility**: Same pipeline works across pandas, pyarrow, etc.
    5. **Artifact Management**: Automatic persistence and reuse of fitted pipelines
    """

    # Option keys for pipeline configuration
    PIPELINE_NAME = "pipeline_name"
    PIPELINE_STEPS = "pipeline_steps"
    PIPELINE_PARAMS = "pipeline_params"

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        """Return the artifact class for sklearn pipeline persistence."""
        return SklearnArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from the pipeline feature name."""
        mloda_source_features = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)

        # Handle multiple source features separated by commas
        if "," in mloda_source_features:
            source_features = [f.strip() for f in mloda_source_features.split(",")]
        else:
            source_features = [mloda_source_features]

        return {Feature(feature_name) for feature_name in source_features}

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^sklearn_pipeline_([\w]+)__"

    @classmethod
    def get_pipeline_name(cls, feature_name: str) -> str:
        """Extract the pipeline name from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid sklearn pipeline feature name format: {feature_name}")
        return prefix_part

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # Validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Extract pipeline name to ensure it's valid
            pipeline_name = cls.get_pipeline_name(feature_name)
            return len(pipeline_name) > 0
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Apply scikit-learn pipelines to features.

        Processes all requested features, determining the pipeline configuration
        and source features from each feature name.

        Adds the pipeline results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            pipeline_name = cls.get_pipeline_name(feature_name)
            source_features_str = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            # Handle multiple source features
            if "," in source_features_str:
                source_features = [f.strip() for f in source_features_str.split(",")]
            else:
                source_features = [source_features_str]

            # Check that all source features exist
            for source_feature in source_features:
                cls._check_source_feature_exists(data, source_feature)

            # Get pipeline configuration from options or create default
            pipeline_config = cls._get_pipeline_config(features, feature_name, pipeline_name)

            # Create unique artifact key for this pipeline
            artifact_key = f"sklearn_pipeline_{pipeline_name}__{','.join(source_features)}"

            # Try to load existing fitted pipeline from artifact using helper method
            fitted_pipeline = None
            artifact = SklearnArtifact.load_sklearn_artifact(features, artifact_key)
            if artifact:
                fitted_pipeline = artifact["fitted_transformer"]
                if not cls._pipeline_matches_config(fitted_pipeline, pipeline_config):
                    raise ValueError(
                        f"Pipeline configuration mismatch for artifact '{artifact_key}'. Expected configuration does not match loaded pipeline."
                    )

            # If no fitted pipeline available, create and fit new one
            if fitted_pipeline is None:
                fitted_pipeline = cls._create_and_fit_pipeline(data, source_features, pipeline_config)

                # Save the fitted pipeline as artifact using helper method
                artifact_data = {
                    "fitted_transformer": fitted_pipeline,
                    "feature_names": source_features,
                    "pipeline_name": pipeline_name,
                    "training_timestamp": datetime.datetime.now().isoformat(),
                }
                SklearnArtifact.save_sklearn_artifact(features, artifact_key, artifact_data)

            # Apply the fitted pipeline to get results
            result = cls._apply_pipeline(data, source_features, fitted_pipeline)

            # Add result to data
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _get_pipeline_config(cls, features: FeatureSet, feature_name: str, pipeline_name: str) -> Dict[str, Any]:
        """
        Get pipeline configuration from options or create default configuration.

        Args:
            features: The feature set
            feature_name: The name of the feature
            pipeline_name: The name of the pipeline

        Returns:
            Pipeline configuration dictionary
        """
        # Try to get configuration from options
        if features.options and hasattr(features.options, "data"):
            pipeline_steps = features.options.data.get(cls.PIPELINE_STEPS)
            pipeline_params = features.options.data.get(cls.PIPELINE_PARAMS, {})

            if pipeline_steps:
                # Handle frozenset case due to options - convert back to list for sklearn
                if isinstance(pipeline_steps, frozenset):
                    pipeline_steps = cls._reconstruct_pipeline_steps_from_frozenset(pipeline_steps)
                return {"steps": pipeline_steps, "params": pipeline_params}

        # Create default configuration based on pipeline name
        return cls._create_default_pipeline_config(pipeline_name)

    @classmethod
    def _reconstruct_pipeline_steps_from_frozenset(cls, pipeline_steps_frozenset: FrozenSet[Any]) -> List[Any]:
        """
        Reconstruct pipeline steps from frozenset back to list of (name, transformer) tuples.

        Args:
            pipeline_steps_frozenset: Frozenset containing (name, transformer_class_name) tuples

        Returns:
            List of (name, transformer_instance) tuples for sklearn Pipeline
        """
        # Create a mapping of transformer class names to actual instances
        transformer_map = cls._get_transformer_map()

        steps_list = []
        for name, transformer_class_name in pipeline_steps_frozenset:
            if transformer_class_name in transformer_map:
                transformer_instance = transformer_map[transformer_class_name]
                steps_list.append((name, transformer_instance))
            else:
                # Fallback to StandardScaler if unknown transformer
                try:
                    from sklearn.preprocessing import StandardScaler

                    steps_list.append((name, StandardScaler()))
                except ImportError:
                    pass

        return steps_list

    @classmethod
    def _get_transformer_map(cls) -> Dict[str, Any]:
        """
        Get a mapping of transformer class names to transformer instances.

        Returns:
            Dictionary mapping class names to transformer instances
        """
        transformer_map = {}

        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

            transformer_map.update(
                {
                    "StandardScaler": StandardScaler(),
                    "MinMaxScaler": MinMaxScaler(),
                    "RobustScaler": RobustScaler(),
                    "MaxAbsScaler": MaxAbsScaler(),
                }
            )
        except ImportError:
            pass

        try:
            # Try to import SimpleImputer from different locations depending on sklearn version
            try:
                from sklearn.impute import SimpleImputer
            except ImportError:
                from sklearn.preprocessing import Imputer as SimpleImputer
            transformer_map["SimpleImputer"] = SimpleImputer(strategy="mean")
        except ImportError:
            pass

        try:
            from sklearn.preprocessing import LabelEncoder, OneHotEncoder

            transformer_map.update(
                {
                    "LabelEncoder": LabelEncoder(),
                    "OneHotEncoder": OneHotEncoder(),
                }
            )
        except ImportError:
            pass

        return transformer_map

    @classmethod
    def _import_sklearn_components(cls) -> Dict[str, Any]:
        """
        Import sklearn components with fallback logic for different versions.

        Returns:
            Dictionary containing imported sklearn components

        Raises:
            ImportError: If sklearn is not available
        """
        components = {}

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            components["StandardScaler"] = StandardScaler
            components["Pipeline"] = Pipeline

            # Try to import SimpleImputer from different locations depending on sklearn version
            try:
                from sklearn.impute import SimpleImputer

                components["SimpleImputer"] = SimpleImputer
            except ImportError:
                from sklearn.preprocessing import Imputer as SimpleImputer

                components["SimpleImputer"] = SimpleImputer

        except ImportError:
            raise ImportError(
                "scikit-learn is required for SklearnPipelineFeatureGroup. Install with: pip install scikit-learn"
            )

        return components

    @classmethod
    def _create_default_pipeline_config(cls, pipeline_name: str) -> Dict[str, Any]:
        """
        Create default pipeline configuration based on pipeline name.

        Args:
            pipeline_name: The name of the pipeline

        Returns:
            Default pipeline configuration
        """
        sklearn_components = cls._import_sklearn_components()
        StandardScaler = sklearn_components["StandardScaler"]
        SimpleImputer = sklearn_components["SimpleImputer"]

        # Define common pipeline configurations
        if pipeline_name == "preprocessing":
            return {"steps": [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())], "params": {}}
        elif pipeline_name == "scaling":
            return {"steps": [("scaler", StandardScaler())], "params": {}}
        elif pipeline_name == "imputation":
            return {"steps": [("imputer", SimpleImputer(strategy="mean"))], "params": {}}
        else:
            # Default to simple scaling
            return {"steps": [("scaler", StandardScaler())], "params": {}}

    @classmethod
    def _pipeline_matches_config(cls, fitted_pipeline: Any, config: Dict[str, Any]) -> bool:
        """
        Check if a fitted pipeline matches the expected configuration.

        Args:
            fitted_pipeline: The fitted pipeline
            config: The expected configuration

        Returns:
            True if the pipeline matches the configuration
        """
        try:
            # Basic check: compare number of steps
            if hasattr(fitted_pipeline, "steps"):
                return len(fitted_pipeline.steps) == len(config["steps"])
            return False
        except Exception:
            return False

    @classmethod
    def _create_and_fit_pipeline(cls, data: Any, source_features: List[Any], config: Dict[str, Any]) -> Any:
        """
        Create and fit a new pipeline.

        Args:
            data: The input data
            source_features: List of source feature names
            config: Pipeline configuration

        Returns:
            Fitted pipeline
        """
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            raise ImportError(
                "scikit-learn is required for SklearnPipelineFeatureGroup. Install with: pip install scikit-learn"
            )

        # Create pipeline from configuration
        pipeline = Pipeline(config["steps"])

        # Set parameters if provided
        if config.get("params"):
            pipeline.set_params(**config["params"])

        # Extract training data
        X_train = cls._extract_training_data(data, source_features)

        # Fit the pipeline
        pipeline.fit(X_train)

        return pipeline

    @classmethod
    def _extract_training_data(cls, data: Any, source_features: List[Any]) -> Any:
        """
        Extract training data for the specified features.

        Args:
            data: The input data
            source_features: List of source feature names

        Returns:
            Training data for the features
        """
        raise NotImplementedError(f"_extract_training_data not implemented in {cls.__name__}")

    @classmethod
    def _apply_pipeline(cls, data: Any, source_features: List[Any], fitted_pipeline: Any) -> Any:
        """
        Apply the fitted pipeline to the data.

        Args:
            data: The input data
            source_features: List of source feature names
            fitted_pipeline: The fitted pipeline

        Returns:
            Transformed data
        """
        raise NotImplementedError(f"_apply_pipeline not implemented in {cls.__name__}")

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the data
        """
        raise NotImplementedError(f"_check_source_feature_exists not implemented in {cls.__name__}")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        raise NotImplementedError(f"_add_result_to_data not implemented in {cls.__name__}")

    @classmethod
    def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
        """
        Returns the FeatureChainParserConfiguration class for this feature group.

        This method allows the Engine to automatically create features with the correct
        naming convention based on configuration options, rather than requiring explicit
        feature names.

        Returns:
            A configured FeatureChainParserConfiguration class
        """
        return create_configurable_parser(
            parse_keys=[
                cls.PIPELINE_NAME,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="sklearn_pipeline_{pipeline_name}__{mloda_source_feature}",
            validation_rules={
                cls.PIPELINE_NAME: lambda x: isinstance(x, str) and len(x) > 0,
            },
        )
