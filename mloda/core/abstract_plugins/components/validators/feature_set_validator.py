from typing import Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature import Feature


class FeatureSetValidator:
    @staticmethod
    def validate_options_initialized(options: Any, context: str = "FeatureSet") -> None:
        if options is None:
            raise ValueError(f"Options not initialized in {context}")

    @staticmethod
    def validate_equal_options(features: Set["Feature"]) -> None:
        if len(features) <= 1:
            return

        options_list = [feature.options for feature in features]
        first_options = options_list[0]

        for options in options_list[1:]:
            if options != first_options:
                raise ValueError("Features have different options")

    @staticmethod
    def validate_feature_added(feature_name: Optional[str], context: str = "feature") -> None:
        if feature_name is None:
            raise ValueError(f"Feature name is None in {context}")

    @staticmethod
    def validate_filters_not_set(filters: Any) -> None:
        if filters is not None:
            raise ValueError("Filters already set")

    @staticmethod
    def validate_filters_is_set_type(filters: Any) -> None:
        if not isinstance(filters, set):
            raise ValueError("Filters must be a Set type")
