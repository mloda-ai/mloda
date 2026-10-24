from abc import abstractmethod
from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class GeneratedFeatureGroup(FeatureGroup):
    """Root group that generates the data for its feature_names_supported(); its dict result becomes its own frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    @abstractmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise NotImplementedError
