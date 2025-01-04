from typing import Any, Optional, Set, Type
from uuid import UUID

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter


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
        if self.options is None:
            raise ValueError(
                "No options set. This should only be called after adding a feature and thus at least empty Options."
            )

        for feature_name in self.get_all_names():
            if feature_name in self.options.data.keys():
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

    def get_options(self) -> Options:
        if self.options is None:
            raise ValueError("No options set. This should only be called after adding a feature.")
        return self.options

    def get_options_key(self, key: str) -> Any:
        return self.get_options().data.get(key, None)

    def get_initial_requested_features(self) -> Set[FeatureName]:
        return {feature.name for feature in self.features if feature.initial_requested_data}

    def get_name_of_one_feature(self) -> FeatureName:
        if self.name_of_one_feature is None:
            raise ValueError("Name of one feature not set. This should only be called after adding a feature.")
        return self.name_of_one_feature

    def add_filters(self, single_filters: Set[SingleFilter]) -> None:
        if self.filters is not None:
            raise ValueError("Filters already set. This should only be called once during setup of the feature set.")

        if not isinstance(single_filters, Set):
            raise ValueError("Filters should be a set.")

        self.filters = single_filters

    def get_artifact(self, config: Options) -> Any:
        return None
