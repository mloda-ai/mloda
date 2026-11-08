"""Feature groups whose declared input is replaced at runtime by a one-row aggregate."""

from typing import Any

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, Options


class SiblingDictRoot(DataCreatorFeatureGroup):
    """Root producing the shared three-row ``sd_a`` column."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sd_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sd_a": [1, 2, 3]}


class SiblingDictAggregate(FeatureGroup):
    """Consumer of ``sd_a`` returning one aggregated row, which replaces the frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sd_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [6]}


class SiblingDictLostInputConsumer(FeatureGroup):
    """Consumes ``sd_a`` and the aggregate whose one-row dict replaces the frame, so ``sd_a`` is gone."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sd_a"), Feature("SiblingDictAggregate")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 1, 1]}
