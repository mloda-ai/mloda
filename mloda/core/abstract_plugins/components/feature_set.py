from typing import Any, Optional, Set, Type
from uuid import UUID

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.validators.feature_set_validator import FeatureSetValidator
from mloda.core.filter.filter_engine import BaseFilterEngine
from mloda.core.filter.single_filter import SingleFilter


class FeatureSet:
    def __init__(self) -> None:
        self.features: Set[Feature] = set()
        self.options: Optional[Options] = None
        # This is just one uuid for easier access
        self.any_uuid: Optional[UUID] = None
        self.filters: Optional[Set[SingleFilter]] = None
        self.name_of_one_feature: Optional[FeatureName] = None
        self.artifact_to_save: Optional[str] = None
        self.artifact_to_load: Optional[str] = None
        self.save_artifact: Optional[Any] = None
        self.filter_engine: Type[BaseFilterEngine] = BaseFilterEngine

    def add_artifact_name(self) -> None:
        FeatureSetValidator.validate_options_initialized(self.options, "add_artifact_name")
        assert self.options is not None  # Type narrowing for mypy

        for feature_name in self.get_all_names():
            if feature_name in self.options.keys():
                self.artifact_to_load = feature_name
                return

        self.artifact_to_save = self.get_name_of_one_feature().name

    def add(self, feature: Feature) -> None:
        self.features.add(feature)
        self.name_of_one_feature = feature.name
        if self.options is None:
            self.options = feature.options
        if self.any_uuid is None:
            self.any_uuid = feature.uuid

    def remove(self, feature: Feature) -> None:
        self.features.discard(feature)

    def get_all_feature_ids(self) -> Set[UUID]:
        return {feature.uuid for feature in self.features}

    def get_all_names(self) -> Set[str]:
        return {feature.name.name for feature in self.features}

    def __str__(self) -> str:
        return f"{self.features}"

    def get_options_key(self, key: str) -> Any:
        """
        Get a value from the shared options across all features in this FeatureSet.

        This method validates that all features in the set have identical options before
        returning the requested value. If features have different options, it raises ValueError.

        Args:
            key: The option key to retrieve

        Returns:
            The value associated with the key, or None if not found

        Raises:
            ValueError: If options are not initialized or if features have different options

        Note:
            Prefer accessing options directly from individual features when possible.
            Only use this when you need to ensure all features share the same option value.
        """
        FeatureSetValidator.validate_options_initialized(self.options, "get_options_key")
        assert self.options is not None  # Type narrowing for mypy
        FeatureSetValidator.validate_equal_options(self.features)
        return self.options.get(key)

    def validate_equal_options(self) -> None:
        """Checks if all features have the same options."""
        FeatureSetValidator.validate_equal_options(self.features)

    def get_initial_requested_features(self) -> Set[FeatureName]:
        return {feature.name for feature in self.features if feature.initial_requested_data}

    def get_name_of_one_feature(self) -> FeatureName:
        FeatureSetValidator.validate_feature_added(
            self.name_of_one_feature.name if self.name_of_one_feature else None, "get_name_of_one_feature"
        )
        assert self.name_of_one_feature is not None  # Type narrowing for mypy
        return self.name_of_one_feature

    def add_filters(self, single_filters: Set[SingleFilter]) -> None:
        FeatureSetValidator.validate_filters_not_set(self.filters)
        FeatureSetValidator.validate_filters_is_set_type(single_filters)
        self.filters = single_filters

    def get_artifact(self, config: Options) -> Any:
        return None
