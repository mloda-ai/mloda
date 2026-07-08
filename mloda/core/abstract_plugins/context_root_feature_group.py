from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class ContextRootFeatureGroup(FeatureGroup):
    """Abstract root feature group that generates its declared FEATURES in place via DataCreator.

    Overrides five members (feature_names_supported, input_data, match_feature_group_criteria,
    compute_framework_rule, input_features) and requires subclasses to implement calculate_feature.
    FEATURES is required: a subclass with empty FEATURES matches nothing.
    Option- or data-access-sensitive matching must override match_feature_group_criteria;
    overriding input_data() alone does not change matching, which is a pure membership test.
    """

    FEATURES: ClassVar[set[str]] = set()
    COMPUTE_FRAMEWORKS: ClassVar[set[type[ComputeFramework]] | None] = None

    @classmethod
    @abstractmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any: ...

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.FEATURES)

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(cls.feature_names_supported())

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return cls.get_column_base_feature(str(feature_name)) in cls.feature_names_supported()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return set(cls.COMPUTE_FRAMEWORKS) if cls.COMPUTE_FRAMEWORKS is not None else None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None
