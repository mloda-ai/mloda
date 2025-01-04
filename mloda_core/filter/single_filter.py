from typing import Any, Dict, Tuple, Union
import uuid


from mloda_core.filter.filter_type_enum import FilterTypeEnum


class SingleFilter:
    """
    Represents a single filter with a feature, filter type, and parameters.
    """

    def __init__(
        self,
        filter_feature: Union[str, Any],  # Union[str, Feature]
        filter_type: Union[str, FilterTypeEnum],
        parameter: Dict[str, Any],
    ) -> None:
        """
        Initialize a SingleFilter instance.

        :param filter_feature: The feature to which the filter applies.
        :param filter_type: The type of filter (e.g., 'range', 'zscore', etc.).
        :param parameter: A dictionary of parameters required by the filter.
        """
        self.filter_feature = self.handle_filter_feature(filter_feature)
        self.filter_type = self.handle_filter_type(filter_type)
        self.parameter = self.handle_parameter(parameter)
        self.name = self.filter_feature.name.name

        self.uuid = uuid.uuid4()

    def handle_filter_type(self, filter_type: Union[str, FilterTypeEnum]) -> str:
        if not filter_type:
            raise ValueError(f"Filter type evaluates to false {filter_type}.")

        if isinstance(filter_type, FilterTypeEnum):
            return filter_type.value
        elif isinstance(filter_type, str):
            return filter_type

        raise ValueError(f"Wrong type of Filter. {filter_type}")

    def handle_filter_feature(self, filter_feature: Union[str, Any]) -> Any:  # Union[str, Feature]
        from mloda_core.abstract_plugins.components.feature import Feature

        if isinstance(filter_feature, Feature):
            return filter_feature
        elif isinstance(filter_feature, str):
            return Feature(name=filter_feature)
        else:
            raise ValueError(f"filter_feature is of wrong type {filter_feature}")

    def handle_parameter(self, parameter: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        if not isinstance(parameter, dict):
            raise ValueError(f"Filter parameter is no dictionary: {parameter}.")

        elif not parameter:
            raise ValueError(f"Dictionary is empty: {parameter}.")

        # Convert dictionary to a tuple of sorted key-value pairs for hashability
        return tuple(sorted(parameter.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleFilter):
            return False
        return (
            self.filter_feature == other.filter_feature
            and self.filter_type == other.filter_type
            and self.parameter == other.parameter
        )

    def __hash__(self) -> int:
        # Combine the hashes of the feature, type, and parameter for a unique hash value
        return hash((self.filter_feature, self.filter_type, self.parameter))

    def __repr__(self) -> str:
        return f"<SingleFilter(feature_name={self.filter_feature.name}, type={self.filter_type}, parameters={self.parameter})>"
