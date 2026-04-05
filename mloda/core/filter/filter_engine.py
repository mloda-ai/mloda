from abc import ABC
from typing import Any

from mloda.core.filter.single_filter import SingleFilter


class BaseFilterEngine(ABC):
    @classmethod
    def final_filters(cls) -> bool:
        """This function should return True if the filters should be applied after the feature calculation."""
        return False

    @classmethod
    def apply_filters(cls, data: Any, features: Any) -> Any:
        return cls.apply_single_filters(data, features)

    @classmethod
    def applicable_filters(cls, features: Any) -> list[SingleFilter]:
        """Return the filters that apply_filters will actually process.

        A filter is applicable when its column name appears in features.get_all_names().
        This method is the single source of truth for that decision; both the filter
        engine and the pre-elimination validator in ComputeFramework use it.
        """
        if features.filters is None:
            return []
        all_names = features.get_all_names()
        return [sf for sf in features.filters if sf.filter_feature.name in all_names]

    @classmethod
    def apply_single_filters(cls, data: Any, features: Any) -> Any:
        """This function should be used to apply filters to the data if filters are applied one by one."""
        for single_filter in cls.applicable_filters(features):
            data = cls.do_filter(data, single_filter)

        return data

    @classmethod
    def do_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        if filter_feature.filter_type is None:
            raise ValueError(f"Filter type evaluates to None {filter_feature.filter_feature.name}.")

        if filter_feature.filter_type == "range":
            return cls.do_range_filter(data, filter_feature)
        elif filter_feature.filter_type == "min":
            return cls.do_min_filter(data, filter_feature)
        elif filter_feature.filter_type == "max":
            return cls.do_max_filter(data, filter_feature)
        elif filter_feature.filter_type == "equal":
            return cls.do_equal_filter(data, filter_feature)
        elif filter_feature.filter_type == "regex":
            return cls.do_regex_filter(data, filter_feature)
        elif filter_feature.filter_type == "categorical_inclusion":
            return cls.do_categorical_inclusion_filter(data, filter_feature)
        else:
            return cls.do_custom_filter(data, filter_feature)

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_custom_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def get_min_max_operator(cls, filter_feature: SingleFilter) -> Any:
        """Convenience method to get min, max, and max operator from filter parameters"""

        min_parameter = filter_feature.parameter.min_value
        max_parameter = filter_feature.parameter.max_value
        is_max_exclusive = filter_feature.parameter.max_exclusive

        return min_parameter, max_parameter, is_max_exclusive
