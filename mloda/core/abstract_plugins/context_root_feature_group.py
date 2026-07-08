from __future__ import annotations

from typing import ClassVar, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class ContextRootFeatureGroup(FeatureGroup):
    """Root feature group that generates its declared FEATURES in place via DataCreator.

    All four overridden methods stay overridable by subclasses.
    """

    FEATURES: ClassVar[set[str]] = set()
    COMPUTE_FRAMEWORKS: ClassVar[Optional[set[type[ComputeFramework]]]] = None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.FEATURES)

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(cls.feature_names_supported())

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return cls.get_column_base_feature(str(feature_name)) in cls.feature_names_supported()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return cls.COMPUTE_FRAMEWORKS

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None
